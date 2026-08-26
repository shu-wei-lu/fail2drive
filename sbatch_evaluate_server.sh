#!/usr/bin/env bash
#SBATCH --job-name=f2d_eval_server
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:2
#SBATCH --output=/mnt/bapve/thome/shuwei/fail2drive/results/slurm_%x_%j.out
#SBATCH --error=/mnt/bapve/thome/shuwei/fail2drive/results/slurm_%x_%j.err

set -euo pipefail

# One job runs one policy. Submit separate jobs to compare the two policies:
#
#   sbatch --export=ALL,EVAL_MODE=depth_ttc sbatch_evaluate_server.sh
#   sbatch --export=ALL,EVAL_MODE=vlm sbatch_evaluate_server.sh
#
# Both allocated GPUs stay on one node:
#   first allocated GPU  -> CARLA + Fail2Drive evaluator
#   second allocated GPU -> Depth Anything or Alpamayo server
#
# Common overrides:
#   ROUTES=fail2drive_split
#   SEEDS="1 2 3"
#   RETRIES=1
#   ROUTE_FILTER=PedestriansOnRoad
#   ROUTE_LIMIT=1
#   FORCE=1
#   OUT_ROOT=/absolute/output/path
#   DEPTH_TTC_DEBUG=0  # disable Depth-TTC debug artifacts
#
# VLM mode additionally requires an Alpamayo checkout/venv:
#   ALPAMAYO_ROOT=/path/to/alpamayo1.5
#   ALPAMAYO_PY=/path/to/alpamayo1.5/a1_5_venv/bin/python

ROOT=${ROOT:-/mnt/bapve/thome/shuwei}
FAIL2DRIVE_ROOT=${FAIL2DRIVE_ROOT:-$ROOT/fail2drive}
FAIL2DRIVE_PY=${FAIL2DRIVE_PY:-$ROOT/miniconda3/envs/fail2drive/bin/python}
CARLA_ROOT=${CARLA_ROOT:-$FAIL2DRIVE_ROOT/f2d_carla}

EVAL_MODE=${EVAL_MODE:-depth_ttc}
EVAL_MODE=${EVAL_MODE,,}
case "$EVAL_MODE" in
  depth_ttc|depth-ttc|depthttc)
    EVAL_MODE=depth_ttc
    ;;
  vlm|alpamayo)
    EVAL_MODE=vlm
    ;;
  *)
    echo "EVAL_MODE must be 'depth_ttc' or 'vlm'; got: $EVAL_MODE" >&2
    exit 2
    ;;
esac

JOB_TAG=${SLURM_JOB_ID:-manual}
RESULT_BASE=${OUT_ROOT:-$FAIL2DRIVE_ROOT/results/${EVAL_MODE}_evaluation/$JOB_TAG}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-$FAIL2DRIVE_ROOT/results/server_logs}
mkdir -p "$RESULT_BASE" "$SERVER_LOG_DIR"

if [ ! -x "$FAIL2DRIVE_PY" ]; then
  echo "Missing Fail2Drive Python: $FAIL2DRIVE_PY" >&2
  exit 1
fi
if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
  echo "Missing CARLA executable: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 1
fi

cd "$FAIL2DRIVE_ROOT"

export WORK_DIR="$FAIL2DRIVE_ROOT"
export CARLA_ROOT
export LEADERBOARD_ROOT="$FAIL2DRIVE_ROOT/leaderboard"
export SCENARIO_RUNNER_ROOT="$FAIL2DRIVE_ROOT/scenario_runner"
# Use the CARLA wheel installed in the fail2drive environment. The legacy egg
# under PythonAPI/carla/dist depends on system libjpeg.so.8/libtiff.so.5, which
# are not available on the HPC nodes. PythonAPI/carla itself stays on the path
# for the top-level `agents` helpers, but the dist egg must not be added.
PYBASE="$FAIL2DRIVE_ROOT/team_code:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$FAIL2DRIVE_ROOT"
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  PYBASE="$PYBASE:$EXTRA_PYTHONPATH"
fi
export PYTHONPATH="$PYBASE"

CARLA_PYTHON_MODULE=$("$FAIL2DRIVE_PY" -c 'import carla; print(carla.__file__)')
echo "CARLA Python module: $CARLA_PYTHON_MODULE"

# Preserve Slurm's actual GPU tokens (numeric IDs, UUIDs, or MIG IDs). Each
# subprocess receives exactly one GPU and therefore uses it locally as cuda:0.
ALLOCATED_GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a GPU_IDS <<< "$ALLOCATED_GPUS"
if [ "${#GPU_IDS[@]}" -lt 2 ]; then
  echo "This job requires two visible GPUs; CUDA_VISIBLE_DEVICES=$ALLOCATED_GPUS" >&2
  exit 1
fi
EVAL_GPU=${GPU_IDS[0]}
SERVER_GPU=${GPU_IDS[1]}

echo "Job ID: $JOB_TAG"
echo "Node: $(hostname)"
echo "Mode: $EVAL_MODE"
echo "Allocated GPUs: $ALLOCATED_GPUS"
echo "Evaluator/CARLA GPU token: $EVAL_GPU"
echo "Model server GPU token: $SERVER_GPU"
echo "Result root: $RESULT_BASE"
nvidia-smi || true

free_port() {
  "$FAIL2DRIVE_PY" - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 200):
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tcp.bind(("", port))
        udp.bind(("", port))
    except OSError:
        tcp.close()
        udp.close()
        continue
    tcp.close()
    udp.close()
    print(port)
    raise SystemExit(0)
raise SystemExit(f"No free port found in [{start}, {start + 199}]")
PY
}

ports_available() {
  "$FAIL2DRIVE_PY" - "$@" <<'PY'
import socket
import sys

sockets = []
try:
    for raw_port in sys.argv[1:]:
        port = int(raw_port)
        for socket_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            sock = socket.socket(socket.AF_INET, socket_type)
            sock.bind(("", port))
            sockets.append(sock)
except OSError:
    raise SystemExit(1)
finally:
    for sock in sockets:
        sock.close()
PY
}

# Concurrent jobs can share a compute node. A probe-and-release free-port
# search races when two jobs probe before either CARLA process binds. Reserve
# a node-local port slot with flock for the lifetime of the batch shell. The
# job-id-derived slot is only the first candidate; busy/stale slots are skipped.
# Explicit caller overrides still take precedence. Manual non-Slurm runs retain
# free-port probing.
if [[ "$JOB_TAG" =~ ^[0-9]+$ ]]; then
  PREFERRED_PORT_SLOT=$((JOB_TAG % 2000))
  PORT_SLOT=""
  PORT_LOCK_FD=""
  for ((slot_offset = 0; slot_offset < 2000; slot_offset++)); do
    candidate_slot=$(((PREFERRED_PORT_SLOT + slot_offset) % 2000))
    candidate_carla=$((10000 + candidate_slot * 4))
    candidate_model=$((20000 + candidate_slot))
    candidate_streaming=$((30000 + candidate_slot * 4))
    candidate_tm=$((40000 + candidate_slot * 4))

    exec {candidate_lock_fd}>"/tmp/f2d_eval_port_slot_${candidate_slot}.lock"
    if flock -n "$candidate_lock_fd" && ports_available \
        "$candidate_carla" "$candidate_model" "$candidate_streaming" "$candidate_tm"; then
      PORT_SLOT=$candidate_slot
      PORT_LOCK_FD=$candidate_lock_fd
      break
    fi
    exec {candidate_lock_fd}>&-
  done
  if [ -z "$PORT_SLOT" ]; then
    echo "Unable to reserve a free CARLA/server port slot on node $(hostname)" >&2
    exit 1
  fi
  DEFAULT_CARLA_PORT=$((10000 + PORT_SLOT * 4))
  DEFAULT_MODEL_PORT=$((20000 + PORT_SLOT))
  DEFAULT_STREAMING_PORT=$((30000 + PORT_SLOT * 4))
  DEFAULT_TM_PORT=$((40000 + PORT_SLOT * 4))
else
  DEFAULT_CARLA_PORT=$(free_port 10000)
  DEFAULT_MODEL_PORT=$(free_port 20000)
  DEFAULT_STREAMING_PORT=$(free_port 30000)
  DEFAULT_TM_PORT=$(free_port 40000)
fi

CARLA_PORT=${CARLA_PORT:-$DEFAULT_CARLA_PORT}
TM_PORT=${TM_PORT:-$DEFAULT_TM_PORT}
STREAMING_PORT=${STREAMING_PORT:-$DEFAULT_STREAMING_PORT}

echo "CARLA port: $CARLA_PORT"
echo "Model server port: $DEFAULT_MODEL_PORT"
echo "CARLA streaming port: $STREAMING_PORT"
echo "Traffic Manager port: $TM_PORT"

SERVER_PID=""
SERVER_LOG=""
cleanup() {
  set +e
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local health_url=$1
  local attempts=$2
  local delay_s=$3
  local label=$4
  local attempt

  echo "Waiting for $label health check: $health_url"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl -sf "$health_url" >/dev/null; then
      echo "$label is ready."
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$label exited during startup. Last log lines:" >&2
      tail -100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep "$delay_s"
  done

  echo "$label did not become ready. Last log lines:" >&2
  tail -100 "$SERVER_LOG" >&2 || true
  return 1
}

# Existing TransFuser steering vectors on this HPC checkout, ordered as
# brake,left,right to match the alpha vectors in sensor_agent.py.
export ACTIVATION_VECTOR_PATHS=${ACTIVATION_VECTOR_PATHS:-"$FAIL2DRIVE_ROOT/steering/transfuser/post_process/brake/steering_vector.pt,$FAIL2DRIVE_ROOT/steering/transfuser/post_process/left/steering_vector.pt,$FAIL2DRIVE_ROOT/steering/transfuser/post_process/right/steering_vector.pt"}
export NORMALIZE_STEERING_VECTOR=${NORMALIZE_STEERING_VECTOR:-0}

case "$EVAL_MODE" in
  depth_ttc)
    DEPTH_ROOT=${DEPTH_ROOT:-$ROOT/Depth-Anything-V2}
    DEPTH_CHECKPOINT=${DEPTH_CHECKPOINT:-$DEPTH_ROOT/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth}
    DEPTH_PORT=${DEPTH_PORT:-$DEFAULT_MODEL_PORT}
    SERVER_LOG="$SERVER_LOG_DIR/depth_ttc_server_${JOB_TAG}.log"

    if [ ! -d "$DEPTH_ROOT/metric_depth/depth_anything_v2" ]; then
      echo "Missing Depth Anything metric-depth code under: $DEPTH_ROOT" >&2
      exit 1
    fi
    if [ ! -f "$DEPTH_CHECKPOINT" ]; then
      echo "Missing Depth Anything checkpoint: $DEPTH_CHECKPOINT" >&2
      exit 1
    fi

    export VLM_STEERING=0
    export ACTIVATION_POLICY=depth_ttc
    export DEPTH_TTC_STEERING=1
    export DEPTH_TTC_SERVER_URL="http://127.0.0.1:$DEPTH_PORT/score"
    export DEPTH_TTC_EVERY_N=${DEPTH_TTC_EVERY_N:-5}
    export DEPTH_TTC_HOLD_FRAMES=${DEPTH_TTC_HOLD_FRAMES:-5}
    export DEPTH_TTC_MIN_SPEED_M_S=${DEPTH_TTC_MIN_SPEED_M_S:-0.0}
    export DEPTH_TTC_BRAKE_ALPHA=${DEPTH_TTC_BRAKE_ALPHA:-3.0}
    export DEPTH_TTC_LEFT_ALPHA=${DEPTH_TTC_LEFT_ALPHA:-1.0}
    export DEPTH_TTC_RIGHT_ALPHA=${DEPTH_TTC_RIGHT_ALPHA:-1.0}
    export DEPTH_TTC_VERBOSE=${DEPTH_TTC_VERBOSE:-1}
    DEPTH_TTC_DEBUG=${DEPTH_TTC_DEBUG:-0}
    case "${DEPTH_TTC_DEBUG,,}" in
      0|false|no|off)
        export DEPTH_TTC_DEBUG_DIR=""
        echo "Depth TTC debug artifacts: disabled"
        ;;
      *)
        export DEPTH_TTC_DEBUG_DIR=${DEPTH_TTC_DEBUG_DIR:-$RESULT_BASE/depth_ttc_debug}
        echo "Depth TTC debug artifacts: $DEPTH_TTC_DEBUG_DIR"
        ;;
    esac
    export DEPTH_TTC_DEBUG_SAVE_EVERY=${DEPTH_TTC_DEBUG_SAVE_EVERY:-1}

    CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
      "$FAIL2DRIVE_PY" "$FAIL2DRIVE_ROOT/tools/depth_ttc_server.py" \
        --host 127.0.0.1 \
        --port "$DEPTH_PORT" \
        --device cuda:0 \
        --depth-anything-root "$DEPTH_ROOT" \
        --checkpoint "$DEPTH_CHECKPOINT" \
        --encoder "${DEPTH_TTC_ENCODER:-vitl}" \
        --input-size "${DEPTH_TTC_INPUT_SIZE:-518}" \
        --max-depth "${DEPTH_TTC_MAX_DEPTH:-80}" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    echo "Started Depth TTC server pid=$SERVER_PID log=$SERVER_LOG"
    wait_for_server "http://127.0.0.1:$DEPTH_PORT/health" 120 5 "Depth TTC server"
    ;;

  vlm)
    ALPAMAYO_ROOT=${ALPAMAYO_ROOT:-$ROOT/alpamayo1.5}
    ALPAMAYO_PY=${ALPAMAYO_PY:-$ALPAMAYO_ROOT/a1_5_venv/bin/python}
    ALPAMAYO_PORT=${ALPAMAYO_PORT:-$DEFAULT_MODEL_PORT}
    ALPAMAYO_MODEL=${ALPAMAYO_MODEL:-nvidia/Alpamayo-1.5-10B}
    ALPAMAYO_VERSION=${ALPAMAYO_VERSION:-1.5}
    ALPAMAYO_MAX_NEW_TOKENS=${ALPAMAYO_MAX_NEW_TOKENS:-256}
    SERVER_LOG="$SERVER_LOG_DIR/alpamayo_server_${JOB_TAG}.log"

    if [ ! -x "$ALPAMAYO_PY" ]; then
      echo "Missing Alpamayo Python: $ALPAMAYO_PY" >&2
      echo "Submit with ALPAMAYO_ROOT=... or ALPAMAYO_PY=..." >&2
      exit 1
    fi

    export DEPTH_TTC_STEERING=0
    export VLM_STEERING=1
    export VLM_BACKEND=alpamayo_server
    export VLM_SERVER_URL="http://127.0.0.1:$ALPAMAYO_PORT/generate"
    export ALPAMAYO_SERVER_URL="$VLM_SERVER_URL"
    export VLM_EVERY_N=${VLM_EVERY_N:-5}
    export VLM_DECISION_TTL_FRAMES=${VLM_DECISION_TTL_FRAMES:-5}
    export VLM_SERVER_FRAME_OFFSETS=${VLM_SERVER_FRAME_OFFSETS:--6,-4,-2,0}
    export VLM_MAX_NEW_TOKENS="$ALPAMAYO_MAX_NEW_TOKENS"
    export VLM_VERBOSE=${VLM_VERBOSE:-1}
    export ACTIVATION_POLICY=vlm
    export ALPAMAYO_REPO="$ALPAMAYO_ROOT"
    export ALPAMAYO_RAW_LOG_PATH=${ALPAMAYO_RAW_LOG_PATH:-$RESULT_BASE/alpamayo_raw_responses.txt}
    export ALPAMAYO_INPUT_SAVE_DIR=${ALPAMAYO_INPUT_SAVE_DIR:-$RESULT_BASE/alpamayo_inputs}

    CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
      "$ALPAMAYO_PY" "$FAIL2DRIVE_ROOT/tools/alpamayo_vlm_server.py" \
        --host 127.0.0.1 \
        --port "$ALPAMAYO_PORT" \
        --device cuda:0 \
        --model "$ALPAMAYO_MODEL" \
        --alpamayo-version "$ALPAMAYO_VERSION" \
        --max-new-tokens "$ALPAMAYO_MAX_NEW_TOKENS" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    echo "Started Alpamayo server pid=$SERVER_PID log=$SERVER_LOG"
    wait_for_server "http://127.0.0.1:$ALPAMAYO_PORT/health" 180 10 "Alpamayo server"
    ;;
esac

export LIVE_VISU=${LIVE_VISU:-0}
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}

ROUTES=${ROUTES:-fail2drive_split}
SEEDS=${SEEDS:-1}
RETRIES=${RETRIES:-1}
ROUTE_FILTER=${ROUTE_FILTER:-}
ROUTE_LIMIT=${ROUTE_LIMIT:-0}
FORCE=${FORCE:-1}
CARLA_STARTUP_WAIT=${CARLA_STARTUP_WAIT:-60}
AGENT_FILE=${AGENT_FILE:-team_code/sensor_agent.py}
AGENT_CONFIG=${AGENT_CONFIG:-checkpoints/tfpp}
EVAL_TRACK=${EVAL_TRACK:-SENSORS}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-300}

read -r -a SEED_ARGS <<< "$SEEDS"
EVAL_ARGS=(
  "$FAIL2DRIVE_ROOT/local_evaluate.py"
  --routes "$ROUTES"
  --seeds "${SEED_ARGS[@]}"
  --retries "$RETRIES"
  --out_root "$RESULT_BASE"
  --agent_file "$AGENT_FILE"
  --agent_config "$AGENT_CONFIG"
  --track "$EVAL_TRACK"
  --timeout "$EVAL_TIMEOUT"
  --restart-carla
  --carla-root "$CARLA_ROOT"
  --graphics-adapter 0
  --port "$CARLA_PORT"
  --traffic-manager-port "$TM_PORT"
  --carla-streaming-port "$STREAMING_PORT"
  --carla-startup-wait "$CARLA_STARTUP_WAIT"
)
if [ -n "$ROUTE_FILTER" ]; then
  EVAL_ARGS+=(--route-filter "$ROUTE_FILTER")
fi
if [ "$ROUTE_LIMIT" -gt 0 ]; then
  EVAL_ARGS+=(--route-limit "$ROUTE_LIMIT")
fi
if [ "$FORCE" = "1" ]; then
  EVAL_ARGS+=(--force)
fi

echo "Starting evaluation:"
printf ' %q' "$FAIL2DRIVE_PY" "${EVAL_ARGS[@]}"
printf '\n'

# local_evaluate.py starts CARLA and the leaderboard evaluator as children, so
# both inherit only the first allocated GPU and see it locally as GPU 0.
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$FAIL2DRIVE_PY" -u "${EVAL_ARGS[@]}"

echo "Evaluation finished: $RESULT_BASE"

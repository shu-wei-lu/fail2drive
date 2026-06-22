#!/bin/bash
#SBATCH --job-name=f2d_alpamayo_vlm_test
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24gb
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:2
#SBATCH -o results/slurm/alpamayo_vlm_test_%j.out
#SBATCH -e results/slurm/alpamayo_vlm_test_%j.err

set -euo pipefail

# Submit from the repo root:
#   cd /mnt/beegfs/rhome/shuwei/fail2drive
#   sbatch sbatch_alpamayo_vlm_test.sh
#
# This job runs three processes on the same node:
#   1. Alpamayo VLM server in its uv Python 3.12 env.
#   2. Native CARLA server from f2d_carla.
#   3. Fail2Drive/CARLA evaluator in the fail2drive Python 3.10 env.
#
# Cluster-specific fields to check before submitting:
#   - #SBATCH --partition / --qos / --gres
#   - ALPAMAYO_PY
#   - FAIL2DRIVE_PY
#   - CARLA_ROOT

ROOT=${ROOT:-/mnt/beegfs/rhome/shuwei}
FAIL2DRIVE_ROOT=${FAIL2DRIVE_ROOT:-$ROOT/fail2drive}
ALPAMAYO_ROOT=${ALPAMAYO_ROOT:-$ROOT/alpamayo}

ALPAMAYO_PY=${ALPAMAYO_PY:-$ALPAMAYO_ROOT/ar1_venv/bin/python}
FAIL2DRIVE_PY=${FAIL2DRIVE_PY:-$ROOT/miniforge3/envs/fail2drive/bin/python}

ALPAMAYO_HOST=${ALPAMAYO_HOST:-127.0.0.1}
ALPAMAYO_PORT=${ALPAMAYO_PORT:-8765}
ALPAMAYO_DEVICE=${ALPAMAYO_DEVICE:-cuda:1}
ALPAMAYO_MODEL=${ALPAMAYO_MODEL:-nvidia/Alpamayo-R1-10B}
ALPAMAYO_MAX_NEW_TOKENS=${ALPAMAYO_MAX_NEW_TOKENS:-128}

CARLA_ROOT=${CARLA_ROOT:-$FAIL2DRIVE_ROOT/f2d_carla}
CARLA_STARTUP_SLEEP=${CARLA_STARTUP_SLEEP:-60}

WORLD_PORT_START=${WORLD_PORT_START:-2000}
STREAMING_PORT_START=${STREAMING_PORT_START:-2500}
TM_PORT_START=${TM_PORT_START:-8000}

mkdir -p "$FAIL2DRIVE_ROOT/results/slurm"
cd "$FAIL2DRIVE_ROOT"

echo "JOB ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "FAIL2DRIVE_ROOT=$FAIL2DRIVE_ROOT"
echo "ALPAMAYO_ROOT=$ALPAMAYO_ROOT"
echo "ALPAMAYO_PY=$ALPAMAYO_PY"
echo "FAIL2DRIVE_PY=$FAIL2DRIVE_PY"
nvidia-smi || true

export WORK_DIR="$FAIL2DRIVE_ROOT"
export CARLA_ROOT="$CARLA_ROOT"
export LEADERBOARD_ROOT="$FAIL2DRIVE_ROOT/leaderboard"
export SCENARIO_RUNNER_ROOT="$FAIL2DRIVE_ROOT/scenario_runner"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

if [ ! -x "$ALPAMAYO_PY" ]; then
  echo "Missing Alpamayo python: $ALPAMAYO_PY" >&2
  exit 1
fi
if [ ! -x "$FAIL2DRIVE_PY" ]; then
  echo "Missing fail2drive python: $FAIL2DRIVE_PY" >&2
  exit 1
fi

free_port() {
  local start=$1
  comm -23 <(seq "$start" "$((start + 49))" | sort) \
    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
    | shuf | head -n 1
}

FREE_WORLD_PORT=$(free_port "$WORLD_PORT_START")
FREE_STREAMING_PORT=$(free_port "$STREAMING_PORT_START")
FREE_TM_PORT=$(free_port "$TM_PORT_START")
echo "World Port: $FREE_WORLD_PORT"
echo "Streaming Port: $FREE_STREAMING_PORT"
echo "TM Port: $FREE_TM_PORT"
echo "Alpamayo Server: http://$ALPAMAYO_HOST:$ALPAMAYO_PORT"

ALPAMAYO_LOG="$FAIL2DRIVE_ROOT/results/slurm/alpamayo_server_${SLURM_JOB_ID:-manual}.log"
CARLA_LOG="$FAIL2DRIVE_ROOT/results/slurm/carla_${SLURM_JOB_ID:-manual}.log"

CARLA_PID=""
ALPAMAYO_PID=""
cleanup() {
  set +e
  if [ -n "$ALPAMAYO_PID" ]; then
    kill "$ALPAMAYO_PID" 2>/dev/null || true
    wait "$ALPAMAYO_PID" 2>/dev/null || true
  fi
  if [ -n "$CARLA_PID" ]; then
    kill "$CARLA_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export ALPAMAYO_REPO="$ALPAMAYO_ROOT"
export ALPAMAYO_DEVICE="$ALPAMAYO_DEVICE"
export ALPAMAYO_MODEL="$ALPAMAYO_MODEL"

"$ALPAMAYO_PY" "$FAIL2DRIVE_ROOT/tools/alpamayo_vlm_server.py" \
  --host "$ALPAMAYO_HOST" \
  --port "$ALPAMAYO_PORT" \
  --device "$ALPAMAYO_DEVICE" \
  --model "$ALPAMAYO_MODEL" \
  --max-new-tokens "$ALPAMAYO_MAX_NEW_TOKENS" \
  > "$ALPAMAYO_LOG" 2>&1 &
ALPAMAYO_PID=$!

echo "Started Alpamayo server pid=$ALPAMAYO_PID log=$ALPAMAYO_LOG"
echo "Waiting for Alpamayo server health check..."
for _ in $(seq 1 180); do
  if curl -sf "http://$ALPAMAYO_HOST:$ALPAMAYO_PORT/health" >/dev/null; then
    echo "Alpamayo server is ready."
    break
  fi
  if ! kill -0 "$ALPAMAYO_PID" 2>/dev/null; then
    echo "Alpamayo server exited early. Last log lines:" >&2
    tail -100 "$ALPAMAYO_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done
curl -sf "http://$ALPAMAYO_HOST:$ALPAMAYO_PORT/health" >/dev/null

CARLA_EXE="$CARLA_ROOT/CarlaUE4.sh"
if [ ! -x "$CARLA_EXE" ]; then
  echo "Missing CARLA executable: $CARLA_EXE" >&2
  exit 1
fi
echo "Starting native CARLA from $CARLA_EXE"
"$CARLA_EXE" \
  -carla-rpc-port="$FREE_WORLD_PORT" \
  -nosound -RenderOffScreen -graphicsadapter=0 \
  -carla-primary-port=0 \
  -carla-streaming-port="$FREE_STREAMING_PORT" \
  > "$CARLA_LOG" 2>&1 &
CARLA_PID=$!
echo "Started CARLA pid=$CARLA_PID log=$CARLA_LOG"
sleep "$CARLA_STARTUP_SLEEP"
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "CARLA exited before evaluator started. Last log lines:" >&2
  tail -100 "$CARLA_LOG" >&2 || true
  exit 1
fi

export LIVE_VISU=1
export SAVE_PATH=./viz_vehicle
export DEBUG_CHALLENGE=1

# Alpamayo VLM steering server backend.
export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL="http://$ALPAMAYO_HOST:$ALPAMAYO_PORT/generate"
export VLM_DECISION_TTL_FRAMES=${VLM_DECISION_TTL_FRAMES:-5}
export VLM_EVERY_N=${VLM_EVERY_N:-1}
export VLM_MAX_NEW_TOKENS="$ALPAMAYO_MAX_NEW_TOKENS"
export VLM_VERBOSE=${VLM_VERBOSE:-1}

# Steering/vector settings requested for this test.
export ACTIVATION_POLICY=oracle
export ORACLE_ACTION=auto
export ORACLE_ALPHA=1
export ORACLE_TRIGGER_DISTANCE=35
export ORACLE_HOLD_FRAMES=5
export ORACLE_COOLDOWN_FRAMES=10
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"
export NORMALIZE_STEERING_VECTOR=0

RESULT_DIR="$FAIL2DRIVE_ROOT/results/alpamayo_vlm_test/${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULT_DIR"

"$FAIL2DRIVE_PY" -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint "$RESULT_DIR/simulation_results.json" \
  --debug-checkpoint "$RESULT_DIR/live_results.txt" \
  --timeout=300 \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp \
  --port "$FREE_WORLD_PORT" \
  --traffic-manager-port "$FREE_TM_PORT" \
  --traffic-manager-seed=1

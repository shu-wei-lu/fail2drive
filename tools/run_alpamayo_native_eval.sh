#!/usr/bin/env bash
set -euo pipefail

cd /media/user/data1/shu_wei/fail2drive

source /home/user/miniconda3/etc/profile.d/conda.sh

PYBASE="/media/user/data1/shu_wei/fail2drive/team_code"
PYBASE="$PYBASE:/media/user/data1/shu_wei/fail2drive/f2d_carla/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg"
PYBASE="$PYBASE:/media/user/data1/shu_wei/fail2drive/f2d_carla/PythonAPI/carla"
PYBASE="$PYBASE:/media/user/data1/shu_wei/fail2drive/leaderboard"
PYBASE="$PYBASE:/media/user/data1/shu_wei/fail2drive/scenario_runner"
PYBASE="$PYBASE:/media/user/data1/shu_wei/fail2drive"

export LD_LIBRARY_PATH="/media/user/data1/shu_wei/compat_libs:${LD_LIBRARY_PATH:-}"
export PYTHON_EGG_CACHE="/media/user/data1/shu_wei/.python-eggs"
export WORK_DIR="/media/user/data1/shu_wei/fail2drive"
export CARLA_ROOT="/media/user/data1/shu_wei/fail2drive/f2d_carla"
export LEADERBOARD_ROOT="/media/user/data1/shu_wei/fail2drive/leaderboard"
export SCENARIO_RUNNER_ROOT="/media/user/data1/shu_wei/fail2drive/scenario_runner"
export PYTHONPATH="$PYBASE"

export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export LIVE_VISU=0
export DEBUG_CHALLENGE=0

export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL="${VLM_SERVER_URL:-http://127.0.0.1:8765/generate}"
export ALPAMAYO_SERVER_URL="$VLM_SERVER_URL"
export VLM_EVERY_N="${VLM_EVERY_N:-10}"
export VLM_DECISION_TTL_FRAMES="${VLM_DECISION_TTL_FRAMES:-10}"
export VLM_SERVER_FRAME_OFFSETS="${VLM_SERVER_FRAME_OFFSETS:--6,-4,-2,0}"
export VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-256}"
export VLM_VERBOSE="${VLM_VERBOSE:-0}"

export ACTIVATION_POLICY=vlm
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"

conda run -n llama python local_evaluate.py \
  --routes "${ROUTES:-fail2drive_split}" \
  --seeds "${SEED:-1}" \
  --retries "${RETRIES:-1}" \
  --out_root "${OUT_ROOT:-results/alpamayo_native_traj_steering}" \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter "${GRAPHICS_ADAPTER:-0}" \
  --port "${CARLA_PORT:-2100}" \
  --traffic-manager-port "${TM_PORT:-8100}" \
  --carla-streaming-port "${STREAMING_PORT:-2600}" \
  ${ROUTE_LIMIT:+--route-limit "$ROUTE_LIMIT"} \
  --force

# Installation
clone the repo first: 
```bash
git clone https://github.com/kesai-labs/lead
git worktree add ../lead-cvpr2026 origin/cvpr2026
cd ../lead-cvpr2026
bash scripts/download_one_checkpoint.sh
```

Install the following packages with fail2drive conda env:
```bash
pip install jaxtyping==0.3.2
pip install beartype==0.21
pip install numba
pip install torchmetrics
conda install -c conda-forge ffmpeg   
```
# Running
Export env parameters first:
```bash
export F2D_ROOT=/mnt/bapve/thome/shuwei/fail2drive
export LEAD_ROOT=/mnt/bapve/thome/shuwei/lead-cvpr2026

export CARLA_ROOT="$F2D_ROOT/f2d_carla"
export LEADERBOARD_ROOT="$F2D_ROOT/leaderboard"
export SCENARIO_RUNNER_ROOT="$F2D_ROOT/scenario_runner"

export PYTHONPATH="$LEAD_ROOT:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT"
export IS_BENCH2DRIVE=0
export PLANNER_TYPE=only_traj
export SAVE_PATH="$F2D_ROOT/results/tfv6_resnet34/test"
# for visualization
export LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true slower_for_stop_sign=true produce_debug_video=true produce_debug_image=true produce_input_video=true produce_input_image=true produce_frame_frequency=5"
export BENCHMARK_ROUTE_ID="Base_BadParking_0002"
# without visualization
# export LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true"
mkdir -p "$SAVE_PATH"
```
Note that, should modify lead-cvpr2026/lead/visualization/visualizer.py if you want to save images.

And then run with slurm_quick_validate.py or(copy the following template command):
```bash
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes "$F2D_ROOT/fail2drive_split/Generalization_Animals_1075.xml" \
  --track SENSORS \
  --agent "$LEAD_ROOT/lead/inference/sensor_agent.py" \
  --agent-config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
```

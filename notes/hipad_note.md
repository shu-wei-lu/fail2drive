# installation
install hipad repo first, and 
pip install carla==0.9.15
download ckpt in hipad

# running
export F2D=/mnt/bapve/thome/shuwei/fail2drive
export HIP=/mnt/bapve/thome/shuwei/HiP-AD
export CARLA_ROOT=$F2D/f2d_carla
export PYTHONPATH=$F2D/leaderboard:$F2D/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$HIP:$PYTHONPATH
export IS_BENCH2DRIVE=1

--agent_file $F2D/team_code/hipad_f2d_agent.py \
--agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \

# carla
./f2d_carla/CarlaUE4.sh -RenderOffScreen -graphicsadapter=7

# run
LIVE_VISU=0 SAVE_PATH=./viz_vehicle DEBUG_CHALLENGE=1 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml

# collect behavior
SAVE_HIPAD_PLAN_FEATURES=1 python create_steering_features_all_splits.py \
  --agent-file $F2D/team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad \
  --save-visual-output

# quick debug subset
SAVE_HIPAD_PLAN_FEATURES=1 python create_steering_features.py \
  --agent-file $F2D/team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad_new/Left \
  --routes steering_split/left \
  --max-frames 200 \
  --save-visual-output 


# run local evaluate
export CUDA_VISIBLE_DEVICES=6
export ACTIVATION_POLICY=pdm_oracle
export PDM_ORACLE_ACTION=auto
export PDM_ORACLE_ALPHA=1.0

# PDM oracle trigger gates. These are activation-steering gates, not exact
# PDM-Lite route-planner internals.
# - TRIGGER_DISTANCE: default active-scenario distance gate in meters.
# - HOLD_FRAMES / COOLDOWN_FRAMES: keep one trigger stable, then prevent repeat firing.
# - TWO_WAY_CLEAR_DISTANCE / LANE_KEY_SEARCH_DISTANCE: simplified clearance check for opposite-lane actions.
# - GENERAL_BRAKE=0: only use scenario-runner triggers; set 1 for generic actor brake fallback.
export ACTIVATION_POLICY=pdm_oracle
export PDM_ORACLE_ACTION=auto
export PDM_ORACLE_ALPHA=1.0
export PDM_ORACLE_TRIGGER_DISTANCE=20
export PDM_ORACLE_HOLD_FRAMES=30
export PDM_ORACLE_COOLDOWN_FRAMES=50
export PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE=70
export PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE=100
export PDM_ORACLE_SIDE_HAZARD_DISTANCE=25
export PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE=10
export PDM_ORACLE_ROADBLOCKED_DISTANCE=40
export PDM_ORACLE_PRIORITY_DISTANCE=25
export PDM_ORACLE_YIELD_EMERGENCY_DISTANCE=50
export PDM_ORACLE_GENERAL_BRAKE=1
export ACTIVATION_VECTOR_PATHS="$F2D/steering/hipad_new/post_process/brake_align_q_v5/steering_vector.pt,$F2D/steering/hipad_new/post_process/left_change_lane_align_q/steering_vector.pt,$F2D/steering/hipad_new/post_process/right_align_q/steering_vector.pt"

export BRAKE_ACTIVATION_ALPHA_SCALE=1.0
export LEFT_ACTIVATION_ALPHA_SCALE=3.0
export RIGHT_ACTIVATION_ALPHA_SCALE=2.0

#### 

python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/hipad_pdm_oracle \
  --agent_file ./team_code/hipad_f2d_agent.py \
  --agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 4 \

# features
python post_process_steering_features.py   --adapter hipad_plan   --collection-root steering/hipad_new   --output-dir steering/hipad_new/post_process/brake_align_q_v3   --action brake   --feature-name align_query   --layer-name layer_00   --positive-include-pattern Brake   --negative-include-pattern Normal --manual

python post_process_steering_features.py   --adapter hipad_plan   --collection-root steering/hipad_new   --output-dir steering/hipad_new/post_process/right_align_q   --action right   --feature-name align_query   --layer-name layer_00   --positive-include-pattern Right   --negative-include-pattern Normal --manual

python post_process_steering_features.py   --adapter hipad_plan   --collection-root steering/hipad_new   --output-dir steering/hipad_new/post_process/left_change_lane_align_q   --action left_change_lane   --feature-name align_query   --layer-name layer_00   --positive-include-pattern Left   --negative-include-pattern Normal --manual

python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad_new \
  --output-dir steering/hipad_new/post_process/brake_matchedv2 \
  --action brake \
  --feature-name align_query \
  --layer-name layer_00 \
  --positive-include-pattern Brake \
  --negative-include-pattern Brake \
  --manual \
  --manual-negative-frames steering/hipad_new/Brake/negative_picked_frames.json

  python slurm_evaluate.py   --routes fail2drive_split   --out_root results/hipad_oracle_new_6   --seeds 1   --retries 2  --agent_file ./team_code/hipad_f2d_agent.py   --agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d"
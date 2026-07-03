# installation
install hipad repo first, and 
pip install carla==0.9.15
download ckpt in hipad

# running
export F2D=/media/user/data1/shu_wei/fail2drive
export HIP=/media/user/data1/shu_wei/hip-ad
export CARLA_ROOT=$F2D/f2d_carla
export PYTHONPATH=$F2D/leaderboard:$F2D/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$HIP:$PYTHONPATH
export IS_BENCH2DRIVE=1

--agent_file $F2D/team_code/hipad_f2d_agent.py \
--agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \

LIVE_VISU=0 SAVE_PATH=./viz_vehicle DEBUG_CHALLENGE=1 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml

# collect behavior
SAVE_HIPAD_PLAN_FEATURES=1 python create_steering_features_all_splits.py \
  --agent-file $F2D/team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad \
  --save-visual-output

# quick debug subset
SAVE_HIPAD_PLAN_FEATURES=1 python create_steering_features_all_splits.py \
  --agent-file $F2D/team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad_debug \
  --splits LaneChangeLeft \
  --max-frames 400 \
  --save-visual-output

# run local evaluate
export CUDA_VISIBLE_DEVICES=6
export ACTIVATION_POLICY=oracle
export ORACLE_ACTION=auto
export ORACLE_ALPHA=1.0
export ORACLE_TRIGGER_DISTANCE=35
export ORACLE_HOLD_FRAMES=30
export ORACLE_COOLDOWN_FRAMES=30

export ACTIVATION_VECTOR_PATHS="$F2D/steering/hipad/post_process/brake_strong/steering_vector.pt,$F2D/steering/hipad/post_process/left_change_lane_meta/steering_vector.pt,$F2D/steering/hipad/post_process/right_change_lane_meta/steering_vector.pt"
export BRAKE_ACTIVATION_ALPHA_SCALE=2.0
export LEFT_ACTIVATION_ALPHA_SCALE=1.0
export RIGHT_ACTIVATION_ALPHA_SCALE=1.0

python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/hipad_oracle \
  --agent_file ./team_code/hipad_f2d_agent.py \
  --agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 6 \
  --force
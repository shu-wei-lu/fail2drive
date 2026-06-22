# Running the PDM-Lite expert policy

python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/visu_agent.py \
  --track MAP \
  --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml

# Running the TransFuser++ model
LIVE_VISU=1 \
SAVE_PATH=./viz_vehicle \
DEBUG_CHALLENGE=1 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Generalization_VehicleOnRoad_1085.xml \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp

#

LIVE_VISU=0 \
DEBUG_CHALLENGE=1 \
SAVE_PATH=./viz_vehicle \
python leaderboard/leaderboard/leaderboard_evaluator_local.py   --routes ./fail2drive_split/Base_BadParking_0001.xml   --agent ./team_code/sensor_agent.py   --agent-config ./checkpoints/tfpp 

LIVE_VISU=0 SAVE_PATH=./viz_vehicle DEBUG_CHALLENGE=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/sensor_agent.py   --agent-config ./checkpoints/tfpp --routes 

# local pc evaluation
cd /media/hcis-s20/SRL/osu_fail2drive/fail2drive
conda activate fail2drive
source env_vars.sh

VLM_STEERING=True \
VLM_BACKEND=internvl  \
VLM_MODEL=OpenGVLab/InternVL3_5-8B \
VLM_DEVICE=cuda:1 \
VLM_LOAD_IN_8BIT=1 \
VLM_MAX_NEW_TOKENS=256 \
VLM_EVERY_N=1 \
VLM_DECAY_FRAMES=15 \
VLM_SAVE_INPUTS=0 \
VLM_VERBOSE=1 \
VLM_ALPHA_MAX=5 \
VLM_ALPHA_MAPPING=piecewise \
VLM_ALPHA_POINTS="0:0,0.3:1.0,0.5:3.0,0.7:4.5,1.0:5.0" \
ACTIVATION_VECTOR_PATH=./steering_feats/brake_minus_normal.pt \
NORMALIZE_STEERING_VECTOR=0 \
LIVE_VISU=0 \
DEBUG_CHALLENGE=0 \
python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/transfuser \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp

  # collect steering 
  NO_OTHER_VEHICLES=1 python create_steering_features.py   --routes steering_split/vehicle_on_road   --agent-file team_code/sensor_agent.py   --agent-config checkpoints/tfpp   --output-root steering_results/steering_features_vehicle_on_road --save-visual-output --max-frames {}

  # post processing
  python post_process_steering_features.py \
  --collection-root steering_results/steering_features_vehicle_on_road \
  --exclude-stop-sign \
  --min-speed 0.5 \
  --brake-threshold 0.5 \
  --normal-throttle-threshold 0.1 \
  --output-dir ./steering_feats

  python post_process_steering_features.py \
  --collection-root steering_results/steering_features_vehicle_on_road \
  --output-dir steering_feats \
  --brake-include-pattern Steering_VehicleOnRoad \
  --normal-include-pattern Steering_Normal \
  --brake-min-target-speed-drop 1.5 \
  --brake-min-speed-ratio 1.15 \
  --brake-min-run-length 3

  python post_process_steering_features.py \
  --collection-root ./steering/transfuser \
  --output-dir ./steering/transfuser/post_process/right_change_lane \
  --action right_change_lane \
  --positive-include-pattern LaneChangeRight \
  --negative-include-pattern normal \
  --min-speed 5

# calibrate alpha
python calibrate_steering_alpha.py \
  --adapter transfuser_target_speed \
  --routes steering_split/normal/Steering_Normal_1085.xml \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --activation-vector-path steering_feats/brake_minus_normal.pt \
  --starting-alpha 1.0 \
  --alpha-interval 1.0 \
  --start-steering-frame 100 \
  --fixed-eval-frames 100 \
  --success-metric target_speed \
  --target-speed-threshold 1.0 \
  --success-ratio 0.95
  --live-visu
  
# VAD
LIVE_VISU=0 DEBUG_CHALLENGE=1 SAVE_PATH=./viz_vehicle python leaderboard/leaderboard/leaderboard_evaluator_local.py   --routes ./steering_split/vehicle_on_road/Steering_VehicleOnRoad_1086.xml   --agent ./team_code/vad_f2d_agent.py --agent-config "../Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+./checkpoints/vad/vad_b2d_base.pth+vad_f2d"

# Oracle policy
ACTIVATION_POLICY=oracle \
ORACLE_ACTION=auto \
ORACLE_ALPHA=1 \
ORACLE_TRIGGER_DISTANCE=35 \
ORACLE_HOLD_FRAMES=5 \
ORACLE_COOLDOWN_FRAMES=10 \
ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt" \
NORMALIZE_STEERING_VECTOR=0 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py ...

# Slurm evaluate
CARLA_ROOT="$PWD/f2d_carla" \
LIVE_VISU=0 \
DEBUG_CHALLENGE=0 \
python -u slurm_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/transfuser \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --lb_script leaderboard/leaderboard/leaderboard_evaluator_local.py
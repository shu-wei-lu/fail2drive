# Alpamayo server
cd /media/user/data1/shu_wei/fail2drive

export ALPAMAYO_ROOT=/media/user/data1/shu_wei/alpamayo1.5
export ALPAMAYO_REPO=$ALPAMAYO_ROOT
export ALPAMAYO_RAW_LOG_PATH=./results/alpamayo_vlm_test/local/alpamayo_raw_responses.txt
export ALPAMAYO_INPUT_SAVE_DIR=./results/alpamayo_vlm_test/local/alpamayo_inputs
# export ALPAMAYO_RAW_OUTPUT_ONLY=1  # debug prompts without Alpamayo action parser/Invalid reason
export CUDA_VISIBLE_DEVICES=7

$ALPAMAYO_ROOT/a1_5_venv/bin/python ./tools/alpamayo_vlm_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --alpamayo-version 1.5 \
  --model nvidia/Alpamayo-1.5-10B \
  --max-new-tokens 256

# Carla server
./f2d_carla/CarlaUE4.sh -RenderOffScreen -graphicsadapter=6

# Planner + evaluation
export CUDA_VISIBLE_DEVICES=6
source env_vars.sh

export LIVE_VISU=1
export SAVE_PATH=./viz_vehicle
export DEBUG_CHALLENGE=1

export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL=http://127.0.0.1:8765/generate
export VLM_DECISION_TTL_FRAMES=5
export VLM_EVERY_N=1
export VLM_SERVER_FRAME_OFFSETS=-6,-4,-2,0  # 20Hz planner: -0.3s,-0.2s,-0.1s,0.0s
export VLM_MAX_NEW_TOKENS=128
export VLM_VERBOSE=1

export ACTIVATION_POLICY=vlm
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"

python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Base_BadParking_0004.xml \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint ./results/alpamayo_vlm_test/local/simulation_results.json \
  --debug-checkpoint ./results/alpamayo_vlm_test/local/live_results.txt \
  --timeout=300 \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp \
  --port 2000 \
  --traffic-manager-port 8000 \
  --traffic-manager-seed=1

# local evaluation

export CUDA_VISIBLE_DEVICES=6

export LIVE_VISU=0
export DEBUG_CHALLENGE=0

export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL=http://127.0.0.1:8765/generate
export VLM_EVERY_N=5
export VLM_DECISION_TTL_FRAMES=5
export VLM_SERVER_FRAME_OFFSETS=-6,-4,-2,0
export VLM_MAX_NEW_TOKENS=256
export VLM_VERBOSE=0

export ACTIVATION_POLICY=vlm
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"

python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/alpamayo_vlm_steering \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 6 \
  --force

  # 1st evaluation
  Per-Scenario HM Overview (scenario classes):
  Class      | Scenario                 | Base HM | Gen HM | Rel %
  -----------+--------------------------+---------+--------+--------
  Behavior   | Wall                     |    84.8 |    0.0 | -100.0%
  Behavior   | FullyBlocked             |    85.2 |   50.1 |  -41.2%
  Behavior   | PedestriansOnRoad        |    82.7 |   29.1 |  -64.8%
  Behavior   | ConstructionPedestrian   |    28.3 |   44.0 |   55.2%
  Visual-lat | BadParking               |    88.6 |   65.7 |  -25.9%
  Visual-lat | ConstructionPermutations |    27.4 |   27.4 |   -0.0%
  Visual-lat | CustomObstacles          |    38.2 |   40.5 |    6.1%
  Visual-lon | ObscuredStop             |   100.0 |   87.3 |  -12.7%
  Visual-lon | HardBrake                |    30.0 |   30.5 |    1.6%
  Visual-lon | RightOfWay               |    83.5 |  100.0 |   19.8%
  Visual-lon | Animals                  |    83.0 |   82.3 |   -0.9%
  Visual-lon | PedestrianOtherBlocker   |    97.0 |   82.3 |  -15.2%
  Robustness | RightConstruction        |    99.6 |   69.3 |  -30.4%
  Robustness | OppositeConstruction     |    99.8 |   85.9 |  -13.9%
  Robustness | ImageOnObject            |    95.3 |   77.5 |  -18.7%
  Robustness | PassableObstacles        |    98.0 |   98.0 |   -0.1%
  Robustness | PedestrianCrowd          |    79.0 |   82.2 |    4.0%

  Per-Class HM Overview (scenario classes):
  Class      | Base HM | Gen HM | Rel %
  -----------+---------+--------+-------
  Visual-lon |    80.9 |   79.0 |  -2.3%
  Visual-lat |    57.6 |   48.8 | -15.3%
  Behavior   |    71.3 |   35.0 | -50.9%
  Robustness |    94.4 |   83.1 | -12.0%

  Parsed rows: 199
  Seeds: 1

  WARNING: no records found in results/alpamayo_vlm_steering/1/res/1041_res.json
  WARNING: seed 1 is missing 1 results: 1041

  Summary:
  B2D: TODO
  DS     : Base 80.8   Gen 74.8 (-7.5%)
  Success: Base 73.0   Gen 56.6 (-22.5%)
  HM     : Base 76.7   Gen 64.4 (-16.1%)

# Steering Flags Note

This is a quick reference for activation steering flags in this repo.

## Runtime Basics

Activation steering has two separate parts:

- Policy: decides when alpha is nonzero.
- Injector: decides which activation vector is applied.

Main policy selector:

```bash
ACTIVATION_POLICY=pdm_oracle    # or vlm, depth_ttc
STEERING_POLICY=pdm_oracle      # alias used when ACTIVATION_POLICY is unset
```

If no named policy is selected, setting `STEERING_ALPHA` or `ACTIVATION_ALPHA` enables `FixedAfterFramePolicy`.

## Fixed Frame Policy

This policy only opens/closes steering by frame id. It does not choose left/right/brake by itself.

```bash
STEERING_ALPHA=1.0
START_STEERING_FRAME=100
END_STEERING_FRAME=180
```

Aliases:

```bash
export ACTIVATION_ALPHA=1.0
export ACTIVATION_START_FRAME=100
export ACTIVATION_END_FRAME=180
```

Behavior:

- before `START_STEERING_FRAME`: alpha is 0
- from `START_STEERING_FRAME` through `END_STEERING_FRAME`: alpha is active
- after `END_STEERING_FRAME`: alpha is 0
- if `END_STEERING_FRAME` is unset, alpha stays active until the route ends

For left/right with fixed frames, use single-vector mode:

```bash
STEERING_ALPHA=1.0 \
START_STEERING_FRAME=100 \
END_STEERING_FRAME=180 \
ACTIVATION_VECTOR_PATH=./steering/transfuser/post_process/left_change_lane/steering_vector.pt \
python leaderboard/leaderboard/leaderboard_evaluator_local.py ...
```

## Activation Vector Paths

Single-vector mode:

```bash
ACTIVATION_VECTOR_PATH=./steering/transfuser/post_process/left_change_lane/steering_vector.pt
```

Action-vector mode uses fixed order `brake,left,right`:

```bash
ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"
```

Equivalent per-action flags:

```bash
BRAKE_ACTIVATION_VECTOR_PATH=...
LEFT_ACTIVATION_VECTOR_PATH=...
RIGHT_ACTIVATION_VECTOR_PATH=...
```

Aliases:

```bash
ACTIVATION_VECTOR_PATH_BRAKE=...
ACTIVATION_VECTOR_PATH_LEFT=...
ACTIVATION_VECTOR_PATH_RIGHT=...
```

Other injector flags:

```bash
ENABLE_ACTIVATION_STEERING=1       # use model default vector path if no vector path is set
ENABLE_ACTIVATION_INJECTOR=1       # alias
ACTIVATION_INJECTOR_VERBOSE=1
```

Action-vector alpha scales:

```bash
BRAKE_ACTIVATION_ALPHA_SCALE=2.0
LEFT_ACTIVATION_ALPHA_SCALE=1.0
RIGHT_ACTIVATION_ALPHA_SCALE=1.0
```

Equivalent compact form, in `brake,left,right` order:

```bash
ACTIVATION_ACTION_ALPHA_SCALES=2.0,1.0,1.0
```

These are injector-side multipliers on the policy alpha. For example, PDM
oracle with `PDM_ORACLE_ALPHA=1.0` and `BRAKE_ACTIVATION_ALPHA_SCALE=2.0`
applies the brake vector with effective alpha `2.0`. If a policy already
outputs brake alpha `2.0`, the same scale makes the effective brake alpha
`4.0`.

Important: with `ACTIVATION_VECTOR_PATHS`, scalar fixed-frame alpha becomes `[alpha,0,0]`, so it maps to brake. For fixed-frame left/right, use single-vector mode unless the policy returns `[brake,left,right]`.

## PDM Oracle Policy

This is the only oracle policy path. The old `ACTIVATION_POLICY=oracle` and
`ORACLE_*` flags still route to `PDMOraclePolicy` for compatibility, but new
runs should use `pdm_oracle` and `PDM_ORACLE_*`.

Enable:

```bash
ACTIVATION_POLICY=pdm_oracle
PDM_ORACLE_STEERING=1
PDM_ORACLE_POLICY=1
```

Core flags:

```bash
PDM_ORACLE_ACTION=auto    # auto, brake, left, right
ACTIVATION_ACTION=auto    # fallback alias
PDM_ORACLE_ALPHA=1.0
STEERING_ALPHA=1.0        # fallback
ACTIVATION_ALPHA=1.0      # fallback
BRAKE_ACTIVATION_ALPHA_SCALE=2.0
LEFT_ACTIVATION_ALPHA_SCALE=1.0
RIGHT_ACTIVATION_ALPHA_SCALE=1.0
PDM_ORACLE_TRIGGER_DISTANCE=50.0
PDM_ORACLE_MIN_DISTANCE=0.0
PDM_ORACLE_HOLD_FRAMES=8
PDM_ORACLE_COOLDOWN_FRAMES=20
PDM_ORACLE_ALLOW_MULTI_ACTION=0
PDM_ORACLE_VERBOSE=0
```

PDM scenario tuning flags:

```bash
PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE=70.0
PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE=90.0
PDM_ORACLE_SIDE_HAZARD_DISTANCE=25.0
PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE=10.0
PDM_ORACLE_ROADBLOCKED_DISTANCE=40.0
PDM_ORACLE_PRIORITY_DISTANCE=50.0
PDM_ORACLE_YIELD_EMERGENCY_DISTANCE=50.0
PDM_ORACLE_GENERAL_BRAKE=0
```

Meaning of the common PDM oracle gates:

- `PDM_ORACLE_TRIGGER_DISTANCE`: default active-scenario distance gate in meters.
- `PDM_ORACLE_HOLD_FRAMES`: keep a trigger active for this many frames after it fires.
- `PDM_ORACLE_COOLDOWN_FRAMES`: wait this many frames after a held trigger before the same action can fire again.
- `PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE`: vehicle-clearance lookahead distance for two-way/overtaking lateral actions.
- `PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE`: distance used to collect target lane ids around ego for the two-way clearance check.
- `PDM_ORACLE_GENERAL_BRAKE`: enable a non-scenario fallback brake trigger for any nearby vehicle/walker satisfying the brake-hazard geometry.

Original PDM-Lite does not use `hold_frames` or `cooldown_frames`; those are
activation-steering stabilizers. The closest original `AutoPilot`/PDM-Lite
parameters are:

```bash
detection_radius=50.0
default_max_distance_to_process_scenario=50
max_distance_to_process_hazard_at_side_lane=25
max_distance_to_process_hazard_at_side_lane_two_ways=10
previous_road_lane_retrieve_distance=100
check_path_free_safety_distance=10
check_path_free_safety_time=0.2
max_distance_to_overtake_two_way_scnearios=8
distance_to_delete_scenario_in_two_ways=2
default_overtake_speed=50/3.6
overtake_speed_vehicle_opens_door_two_ways=40/3.6
idm_two_way_scenarios_minimum_distance=2.0
idm_two_way_scenarios_time_headway=0.1
braking_distance_calculation_safety_distance=10
lane_shift_extension_length_for_yield_to_emergency_vehicle=20
```

PDM oracle brake hazard flags:

```bash
PDM_ORACLE_BRAKE_HAZARD_DISTANCE=20.0
PDM_ORACLE_BRAKE_HAZARD_LATERAL_MARGIN=2.5
PDM_ORACLE_BRAKE_REACTION_TIME=0.4
PDM_ORACLE_BRAKE_DECELERATION=6.0
PDM_ORACLE_BRAKE_DISTANCE_MARGIN=2.0
PDM_ORACLE_BRAKE_TTC_THRESHOLD=2.0
PDM_ORACLE_BRAKE_MIN_CLOSING_SPEED=0.5
```

## Depth TTC Steering

Enable:

```bash
ACTIVATION_POLICY=depth_ttc
DEPTH_TTC_STEERING=1
```

Action alpha flags:

```bash
DEPTH_TTC_BRAKE_ALPHA=3.0
DEPTH_TTC_LATERAL_ALPHA=1.0
```

Scheduling flags:

```bash
DEPTH_TTC_EVERY_N=1
DEPTH_TTC_HOLD_FRAMES=1
DEPTH_TTC_MIN_SPEED_M_S=0.0
```

Depth TTC uses action-vector alpha candidates:

- original: `[0,0,0]`
- brake: `[DEPTH_TTC_BRAKE_ALPHA,0,0]`
- left: `[0,DEPTH_TTC_LATERAL_ALPHA,0]`
- right: `[0,0,DEPTH_TTC_LATERAL_ALPHA]`

## VLM Steering

Enable:

```bash
VLM_STEERING=1
ACTIVATION_POLICY=vlm
```

Decision lifetime:

```bash
VLM_DECISION_TTL_FRAMES=10
VLM_DECAY_FRAMES=10       # fallback alias
```

Trajectory projection:

```bash
VLM_TRAJECTORY_Z_M=0.0
VLM_ALPAMAYO_HISTORY_STEPS=16
VLM_ALPAMAYO_HISTORY_STRIDE_FRAMES=2
```

See `notes/VLM_note.md` and `notes/alpamayo.md` for backend/model/server flags.

## Logging And Feature Collection Env

These are usually set by `create_steering_features.py`:

```bash
SAVE_PATH=<output-root>/logs
SAVE_FUSED_FEATURES=True
FUSED_FEATURES_PATH=<output-root>/features
```

Visual/debug flags:

```bash
DEBUG_CHALLENGE=1
VISUAL_OUTPUT_PATH=<output-root>/images
LIVE_VISU=True
```

Other useful run flags:

```bash
STOP_CONTROL=0        # disable stop-sign controller, used by calibration unless --keep-stop-control
NO_OTHER_VEHICLES=1   # remove background actors, used by calibration unless --keep-other-vehicles
```

## create_steering_features.py Flags

Use `create_steering_features_all_splits.py` when collecting the standard steering set:

```bash
python create_steering_features_all_splits.py \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser
```

The wrapper forwards unknown arguments to `create_steering_features.py`, so flags such as `--max-frames`, `--route-filter`, `--save-visual-output`, `--host`, `--port`, and `--env KEY=VALUE` can be appended to the wrapper command. Wrapper-specific flags are:

```bash
--steering-split-root
--feature-script
--splits
--keep-going
```

Use `create_steering_features.py` directly when collecting a single route file or split:

```bash
python create_steering_features.py \
  --routes steering_split/LaneChangeLeft \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/LaneChangeLeft
```

Important flags:

```bash
--routes
--agent-file
--agent-config
--output-root
--lb-script
--track
--host
--port
--timeout
--traffic-manager-seed
--repetitions
--route-limit
--route-filter
--max-frames
--save-visual-output
--live-visual-output
--no-other-vehicles
--env KEY=VALUE
```

Note: rerunning the same route/output root does not skip existing runs; it appends stdout/stderr logs and creates new timestamped feature/log folders.

## post_process_steering_features.py Flags

Common flags:

```bash
--adapter transfuser_target_speed
--collection-root steering/transfuser
--logs-root
--features-root
--output-dir
--folder-name
--action brake|left_change_lane|right_change_lane
--model-index
--max-frames-per-class
--flatten
--steer-threshold
--normal-max-abs-steer
--positive-include-pattern
--positive-exclude-pattern
--negative-include-pattern
--negative-exclude-pattern
```

TransFuser adapter post-process flags:

```bash
--brake-threshold
--brake-min-target-speed-drop
--brake-min-speed-ratio
--brake-min-run-length
--normal-throttle-threshold
--min-speed
--exclude-stop-sign
--brake-include-pattern
--brake-exclude-pattern
--normal-include-pattern
--normal-exclude-pattern
```

Left/right lane-change post-process currently uses final control steer:

```bash
--action left_change_lane --steer-threshold 0.2
--action right_change_lane --steer-threshold 0.2
```

This is best interpreted as an initiate-frame proxy, not a full lane-change trajectory label.

## calibrate_steering_alpha.py Flags

Common flags:

```bash
--adapter transfuser_target_speed
--routes
--agent-file
--agent-config
--output-root
--activation-vector-path
--lb-script
--track
--host
--port
--timeout
--traffic-manager-seed
--starting-alpha
--alpha-interval
--max-alpha
--start-steering-frame
--fixed-eval-frames
--min-eval-frames
--poll-interval
--stop-timeout
--live-visu
--binary-refine-steps
--keep-other-vehicles
--keep-stop-control
--env KEY=VALUE
```

TransFuser adapter calibration flags:

```bash
--success-ratio
--success-metric target_speed|target_speed_drop|target_speed_ratio|brake_or_target_speed
--target-speed-threshold
--target-speed-drop-threshold
--target-speed-ratio-threshold
--baseline-alpha
--brake-threshold
```

Example fixed-frame left activation:

```bash
STEERING_ALPHA=1.0 \
START_STEERING_FRAME=100 \
END_STEERING_FRAME=180 \
ACTIVATION_VECTOR_PATH=./steering/transfuser/post_process/left_change_lane/steering_vector.pt \
python leaderboard/leaderboard/leaderboard_evaluator_local.py ...
```

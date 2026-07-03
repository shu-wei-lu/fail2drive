# Activation Steering Spec

This note is the high-level workflow for activation steering. Planner-specific details, thresholds, and policy-specific debugging should live in separate notes, for example:

- `notes/steering_features_creation.md`: steering split, collection, post-process details.
- `notes/alpamayo.md`: Alpamayo VLM server and VLM evaluation commands.
- `notes/depth_ttc.md`: Depth Anything / occupancy TTC policy.
- `notes/hipad_note.md`: HiP-AD setup and feature collection notes.

## Goal

For a given planner, collect activations for normal driving and three target actions, convert them into action steering vectors, then evaluate the planner with a policy that decides when to apply each vector.

The action vector order is:

```text
brake,left,right
```

At evaluation time this is passed as:

```bash
export ACTIVATION_VECTOR_PATHS="./steering/<planner>/post_process/Brake/steering_vector.pt,./steering/<planner>/post_process/left_change_lane/steering_vector.pt,./steering/<planner>/post_process/right_change_lane/steering_vector.pt"
```

Optional per-action alpha scales are applied by the injector after the policy
chooses an action. This is useful when the same policy alpha should map to a
stronger brake vector than lane-change vectors:

```bash
export BRAKE_ACTIVATION_ALPHA_SCALE=2.0
export LEFT_ACTIVATION_ALPHA_SCALE=1.0
export RIGHT_ACTIVATION_ALPHA_SCALE=1.0
```

Equivalent compact form, also in `brake,left,right` order:

```bash
export ACTIVATION_ACTION_ALPHA_SCALES=2.0,1.0,1.0
```

These values multiply the policy output. For example, oracle with
`ORACLE_ALPHA=1.0` and `BRAKE_ACTIVATION_ALPHA_SCALE=2.0` applies brake with an
effective alpha of `2.0`.

## 1. Start CARLA

Run from the Fail2Drive root:

```bash
cd /media/user/data1/shu_wei/fail2drive
source env_vars.sh
```

Basic local CARLA launch:

```bash
./f2d_carla/CarlaUE4.sh -RenderOffScreen -graphicsadapter=6
```

Useful CARLA arguments:

```bash
./f2d_carla/CarlaUE4.sh \
  -carla-rpc-port=2000 \
  -nosound \
  -RenderOffScreen \
  -graphicsadapter=6 \
  -carla-primary-port=0 \
  -carla-streaming-port=2001
```

Notes:

- `-RenderOffScreen`: use this when the agent needs RGB cameras.
- `-graphicsadapter=<gpu_id>`: choose which GPU CARLA renders on.
- `-carla-rpc-port=<port>`: evaluator connects through this port.
- `-carla-streaming-port=<port>`: useful when running multiple CARLA servers.
- `-nullrhi`: headless/no rendering mode. Do not use this when the policy/planner needs RGB images.

`local_evaluate.py --restart-carla` can start CARLA per route:

```bash
python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/<run_name> \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 6
```

## 2. Collect Steering Features

For a given planner, run `create_steering_features_all_splits.py` once. It wraps `create_steering_features.py` and runs each steering split:

```text
steering_split/Normal
steering_split/Brake
steering_split/LaneChangeLeft
steering_split/LaneChangeRight
```

Template:

```bash
export PLANNER_NAME=transfuser
export AGENT_FILE=team_code/sensor_agent.py
export AGENT_CONFIG=checkpoints/tfpp

python create_steering_features_all_splits.py \
  --agent-file "$AGENT_FILE" \
  --agent-config "$AGENT_CONFIG" \
  --output-root "steering/$PLANNER_NAME"
```

What this produces under each output root:

```text
logs/<route_run>/activation_actions.jsonl
features/<route_run>/<frame>.pt
results/
stdout/
stderr/
debug/
```

The wrapper writes one output root per split:

```text
steering/<planner>/Normal
steering/<planner>/Brake
steering/<planner>/LaneChangeLeft
steering/<planner>/LaneChangeRight
```

Use `--splits Normal LaneChangeLeft` to run a subset, or call `create_steering_features.py --routes steering_split/<Split>` directly when you need to rerun only one route folder by hand.

Useful collection flags:

```bash
--max-frames 400
--route-filter 1085
--route-limit 1
--save-visual-output
--live-visual-output
--traffic-manager-seed 1
```

Planner-specific feature hooks belong in planner notes. Example: HiP-AD uses `SAVE_HIPAD_PLAN_FEATURES=1` and usually needs `--save-visual-output` for meta-based lane-change labels.

## 3. Post-process Steering Vectors

After collecting the driving logs and features, run `post_process_steering_features.py` to build three action-vs-normal vectors:

```text
Brake/steering_vector.pt
left_change_lane/steering_vector.pt
right_change_lane/steering_vector.pt
```

Template:

```bash
export PLANNER_NAME=transfuser
export ADAPTER=transfuser_target_speed

python post_process_steering_features.py \
  --adapter "$ADAPTER" \
  --collection-root "steering/$PLANNER_NAME" \
  --output-dir "steering/$PLANNER_NAME/post_process/Brake" \
  --action brake \
  --positive-include-pattern Brake \
  --negative-include-pattern Normal

python post_process_steering_features.py \
  --adapter "$ADAPTER" \
  --collection-root "steering/$PLANNER_NAME" \
  --output-dir "steering/$PLANNER_NAME/post_process/left_change_lane" \
  --action left_change_lane \
  --positive-include-pattern LaneChangeLeft \
  --negative-include-pattern Normal

python post_process_steering_features.py \
  --adapter "$ADAPTER" \
  --collection-root "steering/$PLANNER_NAME" \
  --output-dir "steering/$PLANNER_NAME/post_process/right_change_lane" \
  --action right_change_lane \
  --positive-include-pattern LaneChangeRight \
  --negative-include-pattern Normal
```

Each post-process run writes:

```text
positive_mean.pt
negative_mean.pt
steering_vector.pt
selected_frames.jsonl
summary.json
```

Vector formula:

```text
steering_vector = positive_mean - negative_mean
```

Use adapter-specific notes for thresholds:

- `transfuser_target_speed`: TransFuser / TF++ target-speed-token steering.
- `hipad_plan` or `hipad`: HiP-AD plan feature steering.
- `vad_traj`, `uniad_traj`: placeholders until those planner adapters are implemented.

## 4. Policy Layer

The policy decides which action vector to apply at each frame. All policies share the same three-vector interface:

```bash
export ACTIVATION_VECTOR_PATHS="./steering/<planner>/post_process/Brake/steering_vector.pt,./steering/<planner>/post_process/left_change_lane/steering_vector.pt,./steering/<planner>/post_process/right_change_lane/steering_vector.pt"
```

The injector can rescale action alphas without changing the policy:

```bash
export BRAKE_ACTIVATION_ALPHA_SCALE=2.0
export LEFT_ACTIVATION_ALPHA_SCALE=1.0
export RIGHT_ACTIVATION_ALPHA_SCALE=1.0
```

This is a multiplier on the policy alpha. If a policy already emits brake
alpha `2.0`, setting `BRAKE_ACTIVATION_ALPHA_SCALE=2.0` makes the effective
brake alpha `4.0`.

### Oracle

Uses scenario-runner / CARLA privileged state to trigger action vectors.

```bash
export ACTIVATION_POLICY=oracle
export ORACLE_ALPHA=1.0
export BRAKE_ACTIVATION_ALPHA_SCALE=2.0
export LEFT_ACTIVATION_ALPHA_SCALE=1.0
export RIGHT_ACTIVATION_ALPHA_SCALE=1.0
export ORACLE_TRIGGER_DISTANCE=35.0
export ORACLE_HOLD_FRAMES=12
export ORACLE_COOLDOWN_FRAMES=30
```

Use this to verify whether the vectors can help when the trigger is clean.

### VLM: Alpamayo

Uses the Alpamayo VLM server to select `brake`, `left`, or `right`.

Server:

```bash
export ALPAMAYO_ROOT=/media/user/data1/shu_wei/alpamayo1.5
export CUDA_VISIBLE_DEVICES=7

$ALPAMAYO_ROOT/a1_5_venv/bin/python ./tools/alpamayo_vlm_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --alpamayo-version 1.5 \
  --model nvidia/Alpamayo-1.5-10B \
  --max-new-tokens 256
```

Planner environment:

```bash
export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL=http://127.0.0.1:8765/generate
export VLM_EVERY_N=5
export VLM_DECISION_TTL_FRAMES=5
export VLM_SERVER_FRAME_OFFSETS=-6,-4,-2,0
export VLM_MAX_NEW_TOKENS=256
export VLM_VERBOSE=0
export ACTIVATION_POLICY=vlm
```

### Occupancy: Depth Anything / Depth TTC

Uses Depth Anything metric depth and swept-corridor TTC to score candidates.

Server:

```bash
export CUDA_VISIBLE_DEVICES=7

python tools/depth_ttc_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --device cuda:0 \
  --depth-anything-root /media/user/data1/shu_wei/Depth-Anything-V2 \
  --checkpoint /media/user/data1/shu_wei/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth \
  --encoder vitl \
  --max-depth 80
```

Planner environment:

```bash
export ACTIVATION_POLICY=depth_ttc
export DEPTH_TTC_STEERING=1
export DEPTH_TTC_SERVER_URL=http://127.0.0.1:8766/score
export DEPTH_TTC_EVERY_N=5
export DEPTH_TTC_HOLD_FRAMES=5
export DEPTH_TTC_VERBOSE=0
```

## 5. Run Evaluation

### Local PC

Use `local_evaluate.py` for sequential local evaluation. It can either connect to an existing CARLA server or restart CARLA for every route.

```bash
cd /media/user/data1/shu_wei/fail2drive
source env_vars.sh

export CUDA_VISIBLE_DEVICES=6
export LIVE_VISU=0
export DEBUG_CHALLENGE=0

export ACTIVATION_POLICY=<oracle|vlm|depth_ttc>
export ACTIVATION_VECTOR_PATHS="./steering/<planner>/post_process/Brake/steering_vector.pt,./steering/<planner>/post_process/left_change_lane/steering_vector.pt,./steering/<planner>/post_process/right_change_lane/steering_vector.pt"

python local_evaluate.py \
  --routes fail2drive_split \
  --seeds 1 \
  --retries 1 \
  --out_root results/<run_name> \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 6 \
  --force
```

For a smoke test:

```bash
python local_evaluate.py \
  --routes fail2drive_split \
  --route-limit 1 \
  --seeds 1 \
  --retries 1 \
  --out_root results/<run_name>_smoke \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --restart-carla \
  --carla-root ./f2d_carla \
  --graphics-adapter 6 \
  --force
```

### SLURM / HPC

Use the SLURM runner for large sweeps. The runner creates one job script per route, starts CARLA inside Apptainer, finds free world/streaming/TM ports, and writes the same `seed/res`, `seed/out`, `seed/err` layout.

```bash
python slurm_evaluate.py \
  --routes fail2drive_split \
  --out_root results/<run_name> \
  --seeds 1 2 3 \
  --retries 3 \
  --agent_file team_code/sensor_agent.py \
  --agent_config checkpoints/tfpp \
  --carla_image ../carla-wrapper-native.sif \
  --carla_bind_source /media/user/data1/shu_wei/fail2drive
```

Before running on a new cluster, check:

- SLURM partition, QoS, GPU type, memory, and time limit inside `slurm_evaluate.py`.
- `eval_num_jobs.txt` for maximum parallel jobs.
- Whether RGB is needed. If the planner/policy uses cameras, do not pass `--no_rgb`.
- Policy server availability. Alpamayo and Depth TTC usually need a persistent server or a job script that starts the server before evaluation.

## Checklist

1. Choose planner and adapter.
2. Start CARLA with RGB rendering enabled.
3. Collect `Normal`, `Brake`, `LaneChangeLeft`, and `LaneChangeRight`.
4. Post-process three vectors against `Normal`.
5. Set `ACTIVATION_VECTOR_PATHS` in `brake,left,right` order.
6. Choose policy: `oracle`, `vlm`, or `depth_ttc`.
7. Run a one-route smoke test.
8. Run full local or SLURM evaluation.
9. Parse results and inspect failures/logs.

# Steering Features Creation

This documents the recreated steering split and the collection/post-processing flow for TransFuser and HiP-AD steering vectors.

## Recreated Split

The route ids came from `steering/transfuser/post_process/*/selected_frames.jsonl`. They are original `fail2drive_split` route ids.

Generated files live under `steering_split`:

- `Brake`: `SteeringVehicleOnRoad_0000`, `0037`, `0038`, `1085`
- `LaneChangeLeft`: `SteeringChangeLaneLeft_0000`-`0004`, `0035`-`0039`, `1085`-`1089`
- `LaneChangeRight`: `SteeringChangeLaneRight_0000`-`0004`, `0035`-`0039`, `1085`-`1089`
- `Normal`: `Normal_0000`, `1085`

The repo ignores `*.xml` globally, so `.gitignore` explicitly unignores `steering_split/**/*.xml`.

Run this to recreate the XML files:

```bash
cd fail2drive
python create_steering_split.py
```

Implementation details:

- `Brake` uses a `VehicleOnRoad` scenario with a stationary vehicle on the route and `lateral_offset=0`.
- `LaneChangeLeft` uses `VehicleOnRoad` with `distance=30` and `lateral_offset=1.5`, placing the stationary vehicle to the right so the ego steers left.
- `LaneChangeRight` uses `VehicleOnRoad` with `distance=30` and `lateral_offset=-1.5`, placing the stationary vehicle to the left so the ego steers right.
- `Normal` strips all scenarios and keeps only the original route/weather/waypoints. Do not set `NO_OTHER_VEHICLES=1` when collecting Normal if you want normal background actors; `RouteScenario` adds `BackgroundBehavior` by default.

## Collect Features

Start CARLA and source the normal Fail2Drive environment first, then collect all four groups into sibling folders under `steering/transfuser`.

```bash
cd fail2drive
source env_vars.sh

python create_steering_features_all_splits.py \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser
```

This writes:

```text
steering/transfuser/Normal
steering/transfuser/Brake
steering/transfuser/LaneChangeLeft
steering/transfuser/LaneChangeRight
```

To run only one split, call the underlying script directly:

```bash
python create_steering_features.py \
  --routes steering_split/Normal \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/Normal
```

Useful collection options:

- `--max-frames N`: stop each route early for quick debug runs.
- `--route-filter 1085`: run only XML filenames containing that substring.
- `--save-visual-output`: save debug images under `<output-root>/images`.
- `--traffic-manager-seed N`: change background traffic deterministically.
- `--no-other-vehicles`: set `NO_OTHER_VEHICLES=1` inside the evaluator subprocess to disable background and parked vehicles.
- `--splits Normal LaneChangeLeft`: with `create_steering_features_all_splits.py`, run only the listed split folders.
- `--keep-going`: with `create_steering_features_all_splits.py`, continue remaining splits after one split fails.

## Collect HiP-AD Features

HiP-AD uses the same steering split, but feature collection needs two extra pieces:

- `SAVE_HIPAD_PLAN_FEATURES=1` writes the selected HiP-AD plan feature tensor.
- `--save-visual-output` writes `images/<run>/metas/*.json`, which contains `plan_spat`, `plan_temp`, `aim`, and `angle_final`. The HiP-AD post-process rule uses these meta files to label lane-change intention frames.

The default saved feature is the last plan refine layer (`HIPAD_PLAN_FEATURE_LAYER=-1`, equivalent to layer 5 for the 6-layer stage2 decoder).

```bash
cd fail2drive
source env_vars.sh
export F2D=/media/user/data1/shu_wei/fail2drive
export HIP=/media/user/data1/shu_wei/hip-ad

SAVE_HIPAD_PLAN_FEATURES=1 python create_steering_features_all_splits.py \
  --agent-file $F2D/team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad \
  --save-visual-output
```

For quick debug runs, add `--max-frames N` or `--route-filter 0000`. If `--save-visual-output` is omitted, HiP-AD still logs `activation_actions.jsonl`, but the meta-driven lane-change labels cannot be used.

## Post Process

The post-process script can scan all four collection roots through the parent `steering/transfuser`. Include/exclude patterns are case-sensitive substrings matched against the run name and log path.

```bash
python post_process_steering_features.py \
  --collection-root steering/transfuser \
  --output-dir steering/transfuser/post_process/Brake \
  --action brake \
  --positive-include-pattern Brake \
  --negative-include-pattern Normal \
  --brake-threshold 0.5 \
  --brake-min-target-speed-drop 1.5 \
  --brake-min-speed-ratio 1.15 \
  --brake-min-run-length 3 \
  --normal-throttle-threshold 0.1 \
  --min-speed 5.0

python post_process_steering_features.py \
  --collection-root steering/transfuser \
  --output-dir steering/transfuser/post_process/left_change_lane \
  --action left_change_lane \
  --positive-include-pattern LaneChangeLeft \
  --negative-include-pattern Normal \
  --steer-threshold 0.2 \
  --normal-max-abs-steer 0.05 \
  --normal-throttle-threshold 0.1 \
  --min-speed 5.0

python post_process_steering_features.py \
  --collection-root steering/transfuser \
  --output-dir steering/transfuser/post_process/right_change_lane \
  --action right_change_lane \
  --positive-include-pattern LaneChangeRight \
  --negative-include-pattern Normal \
  --steer-threshold 0.2 \
  --normal-max-abs-steer 0.05 \
  --normal-throttle-threshold 0.1 \
  --min-speed 5.0
```

Each post-process run writes:

- `positive_mean.pt`
- `negative_mean.pt`
- `steering_vector.pt`
- `selected_frames.jsonl`
- `summary.json`

The vector formula is `positive_mean - negative_mean`.

## Post Process HiP-AD

Use the `hipad_plan` adapter for HiP-AD. For lane-change actions, this adapter uses `images/<run>/metas/*.json` rather than only `activation_actions.jsonl`.
For newer HiP-AD runs that save multiple tensors under
`features/<run>/<feature-name>/<layer-name>/<frame>.pt`, choose the tensor with
`--feature-name` and `--layer-name`. Older runs that save directly under
`features/<run>/<frame>.pt` keep working without these flags.

The lane-change positive rule is based on planning intention:

- left: `plan_spat` x moves negative enough, `aim.x` is negative, and `angle_final` is negative.
- right: the same rule with positive signs.

The default thresholds are:

- `--hipad-plan-spat-mean-threshold 1.0`
- `--hipad-plan-spat-last-threshold 0.8`
- `--hipad-plan-spat-first-threshold 0.0`
- `--hipad-plan-aim-threshold 0.5`
- `--hipad-plan-angle-threshold 0.12`
- `--hipad-plan-min-desired-speed 0.0`

Normal frames require near-straight planning:

- `--hipad-normal-plan-spat-mean-max 0.25`
- `--hipad-normal-plan-spat-last-max 0.4`
- `--hipad-normal-aim-max 0.2`
- `--hipad-normal-angle-max 0.05`

Brake positives normally use the brake/target-speed labels from the action log. Add
`--hipad-brake-require-neutral-plan` when building a cleaner brake vector; this requires
both brake positives and normal negatives to pass the same near-straight planning
thresholds above.

Example:

```bash
python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad \
  --output-dir steering/hipad/post_process/brake \
  --action brake \
  --feature-name align_query \
  --layer-name layer_02 \
  --positive-include-pattern Brake \
  --negative-include-pattern Normal \
  --hipad-brake-require-neutral-plan

python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad \
  --output-dir steering/hipad/post_process/left_change_lane_meta \
  --action left_change_lane \
  --feature-name align_query \
  --layer-name layer_02 \
  --positive-include-pattern LaneChangeLeft \
  --negative-include-pattern Normal

python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad \
  --output-dir steering/hipad/post_process/right_change_lane_meta \
  --action right_change_lane \
  --feature-name align_query \
  --layer-name layer_02 \
  --positive-include-pattern LaneChangeRight \
  --negative-include-pattern Normal
```

For a stricter left-lane-change vector that requires near-point lateral intent and avoids low-speed/stop signatures, use:

```bash
python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad \
  --output-dir steering/hipad/post_process/left_change_lane_meta_fast_aim \
  --action left_change_lane \
  --feature-name align_query \
  --layer-name layer_02 \
  --positive-include-pattern LaneChangeLeft \
  --negative-include-pattern Normal \
  --hipad-plan-aim-threshold 0.4 \
  --hipad-plan-spat-first-threshold 0.4 \
  --hipad-plan-min-desired-speed 3.0
```

The meta-driven rule skips frames without a matching `metas/<frame>.json`. HiP-AD visualization currently saves every 2 frames, so selected lane-change frames are usually even-numbered. For the observed left-lane-change run, frames `0170` through `0220` match the default positive rule.

To fall back to the old action-log steer-threshold labels, add:

```bash
--hipad-disable-plan-meta-labels
```

# Steering Features Creation

This documents the recreated steering split and the collection/post-processing flow for TransFuser steering vectors.

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

Start CARLA and source the normal Fail2Drive environment first, then collect each group into a sibling folder under `steering/transfuser`.

```bash
cd fail2drive
source env_vars.sh

python create_steering_features.py \
  --routes steering_split/Normal \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/Normal

python create_steering_features.py \
  --routes steering_split/Brake \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/Brake

python create_steering_features.py \
  --routes steering_split/LaneChangeLeft \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/LaneChangeLeft

python create_steering_features.py \
  --routes steering_split/LaneChangeRight \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/LaneChangeRight
```

Useful collection options:

- `--max-frames N`: stop each route early for quick debug runs.
- `--route-filter 1085`: run only XML filenames containing that substring.
- `--save-visual-output`: save debug images under `<output-root>/images`.
- `--traffic-manager-seed N`: change background traffic deterministically.

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

delte left: 0004, 0035, 0036
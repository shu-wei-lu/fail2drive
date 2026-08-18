# Steering Features: Manual-Pick Workflow

Positive frames are selected manually with `picked_frames.json`. Automatic
positive-frame thresholds are optional and listed at the end.

## 1. Collect Features

Use one output directory per split:

| Routes | Output directory |
|---|---|
| `steering_split/Brake` | `steering/<planner>/Brake` |
| `steering_split/Normal` | `steering/<planner>/Normal` |
| `steering_split/left` | `steering/<planner>/left` or `Left` |
| `steering_split/right` | `steering/<planner>/right` or `Right` |

The Slurm wrapper submits one job and starts one CARLA instance for every XML
inside `--routes`:

```bash
python slurm_create_steering_features.py \
  --routes steering_split/Brake \
  --agent-file team_code/sensor_agent.py \
  --agent-config checkpoints/tfpp \
  --output-root steering/transfuser/Brake \
  --conda-env fail2drive \
  --save-visual-output
```

For HiP-AD, export the HiP-AD environment first, including
`SAVE_HIPAD_PLAN_FEATURES=1`, then change the agent:

```bash
python slurm_create_steering_features.py \
  --routes steering_split/Brake \
  --agent-file "$F2D/team_code/hipad_f2d_agent.py" \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --output-root steering/hipad_new/Brake \
  --conda-env hipad \
  --save-visual-output
```

The collection layout remains:

```text
<collection-root>/<Split>/
├── features/<run>/...
├── images/<run>/...
├── logs/<run>/activation_actions.jsonl
├── results/
├── stderr/
└── stdout/
```

## 2. Create `picked_frames.json`

After inspecting the images, metadata, and action log, put a
`picked_frames.json` in each positive split:

```text
steering/<planner>/Brake/picked_frames.json
steering/<planner>/left/picked_frames.json
steering/<planner>/right/picked_frames.json
```

Use the actual directory capitalization (`Left`/`Right` when applicable). The
JSON key must exactly match a run name under `features/` and `logs/`:

```json
{
  "SteeringVehicleOnRoad_0000_route0_07_29_16_28_55": [128, 129, 130]
}
```

Frame values are integers without `.pt`. With `--manual`, these are the only
positive frames. Negative frames still come from the Normal split unless
`--manual-negative-frames` is also supplied.

## 3. Manual TransFuser Post-Process

Choose one action:

| Action | `ACTION` | `POSITIVE_SPLIT` | `OUTPUT_NAME` |
|---|---|---|---|
| Brake | `brake` | `Brake` | `brake_manual` |
| Left | `left_change_lane` | `left` | `left_manual` |
| Right | `right_change_lane` | `right` | `right_manual` |

Example for Brake; change the three variables using the table:

```bash
ACTION=brake
POSITIVE_SPLIT=Brake
OUTPUT_NAME=brake_manual

python post_process_steering_features.py \
  --adapter transfuser_target_speed \
  --collection-root steering/transfuser \
  --output-dir "steering/transfuser/post_process/$OUTPUT_NAME" \
  --action "$ACTION" \
  --manual \
  --positive-include-pattern "$POSITIVE_SPLIT" \
  --negative-include-pattern Normal
```

Pattern matching is case-sensitive. Set `POSITIVE_SPLIT=Left` instead of
`left` when the actual collection directory is `Left`.

## 4. Manual HiP-AD Post-Process

The current `steering/hipad_new` runs use `align_query/layer_00`.

| Action | `ACTION` | `POSITIVE_SPLIT` | `OUTPUT_NAME` |
|---|---|---|---|
| Brake | `brake` | `Brake` | `brake_manual` |
| Left | `left_change_lane` | `Left` | `left_manual` |
| Right | `right_change_lane` | `Right` | `right_manual` |

Example for Brake; change the three variables using the table:

```bash
ACTION=brake
POSITIVE_SPLIT=Brake
OUTPUT_NAME=brake_manual

python post_process_steering_features.py \
  --adapter hipad_plan \
  --collection-root steering/hipad_new \
  --output-dir "steering/hipad_new/post_process/$OUTPUT_NAME" \
  --action "$ACTION" \
  --feature-name align_query \
  --layer-name layer_00 \
  --manual \
  --positive-include-pattern "$POSITIVE_SPLIT" \
  --negative-include-pattern Normal
```

## 5. Optional Manual Negatives

To manually choose negatives too, create a JSON file with the same format:

```text
steering/hipad_new/Brake/negative_picked_frames.json
```

Add these arguments:

```bash
--manual-negative-frames steering/hipad_new/Brake/negative_picked_frames.json \
--negative-include-pattern Brake
```

Remove the earlier `--negative-include-pattern Normal`. A frame cannot appear
in both the positive and negative JSON files.

## 6. Verify the Result

Every successful run writes:

```text
positive_mean.pt
negative_mean.pt
steering_vector.pt
selected_frames.jsonl
summary.json
```

The formula is:

```text
steering_vector = positive_mean - negative_mean
```

Check `summary.json`:

- `manual_positive_frames_path` points to the intended file.
- `manual_positive_frame_count` equals the number of picked frames.
- `positive_count` and `negative_count` are non-zero.
- `missing_features` is normally zero.
- HiP-AD uses the intended `feature_name` and `layer_name`.

Use `selected_frames.jsonl` to audit every frame used in the final vector.

## 7. Optional Filters and Automatic Selection

Feature-collection route selection is optional:

- `--routes steering_split/Brake`: run every XML in the split.
- `--routes path/to/route.xml`: run only that route.
- `--route-filter 1085`: keep only filenames containing `1085`.
- `--route-limit N`: run only the first `N` selected XML files.
- `--max-frames N`: stop each route after `N` evaluator frames.

These case-sensitive patterns restrict which runs may contribute:

- `--positive-include-pattern` / `--positive-exclude-pattern`
- `--negative-include-pattern` / `--negative-exclude-pattern`

With `--manual`, positive thresholds are disabled. The patterns only restrict
where manually listed frames may come from.

Automatically selected Normal negatives can still be controlled with:

- `--normal-max-abs-steer`
- `--normal-throttle-threshold`
- `--min-speed`
- `--exclude-stop-sign`
- `--normal-include-pattern` / `--normal-exclude-pattern`
- `--max-frames-per-class`

For fully automatic positive selection, omit `--manual`. TransFuser then uses:

- Lane change: `--steer-threshold`.
- Brake: `--brake-threshold`, `--brake-min-target-speed-drop`,
  `--brake-min-speed-ratio`, `--brake-min-run-length`, and
  `--brake-target-speed-state-threshold`.
- Optional run filtering: `--brake-include-pattern` and
  `--brake-exclude-pattern`.

HiP-AD automatic lane-change selection additionally uses the
`--hipad-plan-*` thresholds, while automatic Normal selection uses the
`--hipad-normal-*` thresholds. These are not required for manually picked
positives unless an extra adapter filter such as
`--hipad-brake-require-neutral-plan` is explicitly enabled.

# Depth TTC policy

This policy uses the Alpamayo-server style split:

1. The planner runs four forward passes: `original`, `brake`, `left`, `right`.
2. The agent sends each candidate's `pred_checkpoint` and target-speed probabilities to `tools/depth_ttc_server.py`.
3. The server computes steer from `pred_checkpoint`, target speed from the speed probabilities, then rolls out a kinematic bicycle model to produce true time-stamped trajectories.
4. The server runs Depth Anything V2 metric depth on the front image.
5. The server back-projects metric depth to an ego-frame point cloud, computes swept-corridor TTC for each rollout trajectory, and returns the selected action.
6. Ties prefer `original`.

## TTC time calculation

The server checks every point-cloud point against the swept ego corridor of each trajectory segment. For a point inside segment `i`, TTC is linearly interpolated between the segment start/end timestamps:

```text
ttc = t_i + along_segment_ratio * (t_{i+1} - t_i)
```

Trajectory timestamps come from the server-side kinematic bicycle rollout. The agent does not convert checkpoints into trajectories.

If `trajectory_times` is missing from an old request, the server falls back to the old behavior: uniform spacing over `DEPTH_TTC_HORIZON_S`.

## Server

```bash
cd /media/user/data1/shu_wei/fail2drive

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

## Planner

```bash
export DEPTH_TTC_STEERING=1
# or: export ACTIVATION_POLICY=depth_ttc
export DEPTH_TTC_SERVER_URL=http://127.0.0.1:8766/score
export DEPTH_TTC_VERBOSE=1
export DEPTH_TTC_EVERY_N=5
export DEPTH_TTC_HOLD_FRAMES=5

export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"
export NORMALIZE_STEERING_VECTOR=0
```

`DEPTH_TTC_EVERY_N` controls how often the planner runs the four-candidate scoring request. `DEPTH_TTC_HOLD_FRAMES` controls how long the selected `brake/left/right` action remains active after a scoring response. Non-scoring frames only forward the held candidate, so this avoids scoring and four-way planner inference every tick.

Useful tuning knobs:

```bash
export DEPTH_TTC_POINT_STRIDE=2
export DEPTH_TTC_ROLLOUT_DT=0.1
export DEPTH_TTC_MIN_HEIGHT_M=0.2
export DEPTH_TTC_CORRIDOR_MARGIN_M=0.35
export DEPTH_TTC_TIE_EPSILON_S=0.05
export DEPTH_TTC_TIMEOUT_S=30
```

### Ego hood mask

The front camera may see the ego hood in the lower center of the image. Those pixels are ignored before converting metric depth to ego-frame points:

```bash
export DEPTH_TTC_EGO_MASK=1
export DEPTH_TTC_EGO_MASK_BOTTOM_FRAC=0.22
export DEPTH_TTC_EGO_MASK_CENTER_WIDTH_FRAC=0.55
```

`front_ego_mask_debug.png` shows the ignored image region. Increase `BOTTOM_FRAC` if the hood extends higher; increase `CENTER_WIDTH_FRAC` if it extends wider.

### Server-side rollout

The debug policy no longer treats `pred_checkpoint` as a temporal future trajectory. For each candidate, the server:

1. Computes steering using the same lateral PID formula as TF++ direct control.
2. Computes target speed from the candidate speed probabilities and configured target-speed bins.
3. Computes throttle/brake with the same longitudinal controller formula.
4. Rolls out the tuned f2d kinematic bicycle dynamics over `DEPTH_TTC_HORIZON_S` at `DEPTH_TTC_ROLLOUT_DT`.

The f2d repo already has this model in `team_code/kinematic_bicycle_model.py`; the server uses a lightweight local copy of the same equations and constants to avoid importing CARLA-dependent modules.

During rollout, the server treats the predicted checkpoints as a spatial reference path. It extends the path forward along the last checkpoint tangent and tracks a lookahead point ahead of the current rolled-out ego position. This avoids the vehicle passing the last checkpoint and turning back toward it.

## Debug visualization

Enable server-side artifacts:

```bash
export DEPTH_TTC_DEBUG_DIR=./results/depth_ttc_debug
export DEPTH_TTC_DEBUG_SAVE_EVERY=1
export DEPTH_TTC_DEBUG_MAX_POINTS=80000
```

Each request directory contains:

- `front.png`: raw front camera image received by the server.
- `depth_vis.png` and `metric_depth.npy`: Depth Anything metric depth output.
- `points_ego.ply` and `points_ego.npz`: filtered ego-frame point cloud.
- `trajectories.json`: four candidate trajectories in ego frame.
- `trajectory_times.json`: timestamps used for each trajectory point.
- `bev_debug.png`: top-down ego-frame point cloud with trajectory corridors and TTC labels.
- `front_traj_debug.png`: projected trajectory corridors on the front image.
- `front_ego_mask_debug.png`: ignored ego-hood image mask.
- `metadata.json`: request camera intrinsics, ego extent, rollout controls, scores, selected action, and collision details.

Quick checks:

- If `front_traj_debug.png` projects trajectories outside the lane/image, inspect camera intrinsics and crop scaling.
- If `bev_debug.png` point cloud is mirrored left/right, inspect the image-to-ego lateral sign.
- If TTC is too long, inspect `trajectory_times.json` and `metadata.json -> scores -> collision_detail`. If collision `point_ego.x` is too large, metric depth is likely overestimated; if `point_ego.x` is correct but timestamps are large, the trajectory time model is wrong.
- If TTC is always unsafe, reduce `DEPTH_TTC_CORRIDOR_MARGIN_M` or increase `DEPTH_TTC_MIN_COLLISION_POINTS`.
- If point cloud is empty, loosen `DEPTH_TTC_MIN_HEIGHT_M`, `DEPTH_TTC_MAX_HEIGHT_M`, or depth/forward filters.

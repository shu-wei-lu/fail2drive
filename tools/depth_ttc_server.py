#!/usr/bin/env python3
"""HTTP server that scores candidate trajectories with Depth Anything metric depth."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

cv2 = None
np = None
Image = None
torch = None


def _strtobool(value) -> bool:
  return str(value).lower() in ("1", "true", "t", "yes", "y", "on")


class DepthTTCEvaluator:
  ACTION_ORDER = ("original", "brake", "left", "right")
  ACTION_RESPONSE = {
      "original": "none",
      "brake": "brake",
      "left": "left",
      "right": "right",
  }

  def __init__(self, args):
    self._ensure_runtime_deps()
    self.args = args
    self.device = args.device
    self.input_size = int(args.input_size)
    self.max_depth = float(args.max_depth)
    self.point_stride = max(1, int(args.point_stride))
    self.min_depth = float(args.min_depth)
    self.max_forward = float(args.max_forward)
    self.max_lateral = float(args.max_lateral)
    self.min_height = float(args.min_height)
    self.max_height = float(args.max_height)
    self.corridor_margin = float(args.corridor_margin)
    self.min_collision_points = max(1, int(args.min_collision_points))
    self.tie_epsilon = float(args.tie_epsilon)
    self.ego_mask_enabled = _strtobool(args.ego_mask)
    self.ego_mask_bottom_frac = max(0.0, min(1.0, float(args.ego_mask_bottom_frac)))
    self.ego_mask_center_width_frac = max(0.0, min(1.0, float(args.ego_mask_center_width_frac)))
    self.debug_dir = Path(args.debug_dir) if args.debug_dir else None
    self.debug_save_every = max(1, int(args.debug_save_every))
    self.debug_max_points = max(1, int(args.debug_max_points))
    self.rollout_dt = max(float(args.rollout_dt), 1e-3)
    self.rollout_horizon_s = float(args.horizon_s)
    self.brake_deceleration = max(float(args.brake_deceleration), 1e-3)
    self._lock = threading.Lock()
    self._debug_lock = threading.Lock()
    self._debug_counter = 0
    if self.debug_dir is not None:
      self.debug_dir.mkdir(parents=True, exist_ok=True)
      print(f"Depth TTC debug outputs will be saved to {self.debug_dir}", flush=True)
    self.model = self._load_model(args)

  def _load_model(self, args):
    metric_dir = Path(args.depth_anything_root).resolve() / "metric_depth"
    if not metric_dir.exists():
      raise FileNotFoundError(f"Depth Anything metric_depth directory not found: {metric_dir}")
    metric_dir_str = str(metric_dir)
    if metric_dir_str not in sys.path:
      sys.path.insert(0, metric_dir_str)

    from depth_anything_v2.dpt import DepthAnythingV2  # pylint: disable=import-outside-toplevel

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], "max_depth": self.max_depth})
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(self.device).eval()
    print(
        f"Depth TTC loaded Depth Anything V2 encoder={args.encoder} "
        f"checkpoint={args.checkpoint} device={self.device}",
        flush=True)
    return model

  @staticmethod
  def _ensure_runtime_deps():
    global cv2, torch  # pylint: disable=global-statement
    global np, Image  # pylint: disable=global-statement
    if cv2 is None:
      import cv2 as cv2_module  # pylint: disable=import-outside-toplevel
      cv2 = cv2_module
    if np is None:
      import numpy as np_module  # pylint: disable=import-outside-toplevel
      np = np_module
    if Image is None:
      from PIL import Image as image_module  # pylint: disable=import-outside-toplevel
      Image = image_module
    if torch is None:
      import torch as torch_module  # pylint: disable=import-outside-toplevel
      torch = torch_module

  def score(self, payload: dict) -> dict:
    frame_id = payload.get("frame_id")
    image_rgb = self._decode_image(payload["image_base64"])
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with self._lock, torch.inference_mode():
      depth = self.model.infer_image(image_bgr, self.input_size)
    if depth.shape[:2] != image_rgb.shape[:2]:
      depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    camera = payload.get("camera", {}) or {}
    ego_extent = payload.get("ego_extent", {}) or {}
    points = self._depth_to_ego_points(depth, image_rgb.shape[1], image_rgb.shape[0], camera)
    half_width = float(ego_extent.get("y", 1.1)) + self.corridor_margin
    half_length = float(ego_extent.get("x", 2.4))
    horizon_s = float(payload.get("horizon_s", self.args.horizon_s))
    trajectories = payload.get("trajectories", {}) or {}
    trajectory_times = payload.get("trajectory_times", {}) or {}
    rollout_controls = {}
    planner_candidates = payload.get("planner_candidates", {}) or {}
    if planner_candidates:
      rollout = self._rollout_planner_candidates(
          planner_candidates,
          current_speed=float(payload.get("speed", 0.0)),
          horizon_s=horizon_s)
      if rollout["trajectories"]:
        trajectories = rollout["trajectories"]
        trajectory_times = rollout["trajectory_times"]
        rollout_controls = rollout["controls"]

    scores = {}
    for name in self.ACTION_ORDER:
      trajectory = trajectories.get(name)
      if trajectory is None:
        scores[name] = {
            "ttc_s": None,
            "score": -1.0,
            "collision": True,
            "reason": "missing trajectory",
        }
        continue
      score = self._score_trajectory(
          np.asarray(trajectory, dtype=np.float32),
          points,
          horizon_s=horizon_s,
          half_width=half_width,
          half_length=half_length,
          trajectory_times=trajectory_times.get(name))
      scores[name] = score

    selected = self._select(scores)
    action = self.ACTION_RESPONSE[selected]
    result = {
        "frame_id": frame_id,
        "action": action,
        "selected_trajectory": selected,
        "scores": scores,
        "point_count": int(len(points)),
        "reason": self._format_reason(selected, scores),
    }
    debug_path = self._save_debug_artifacts(
        payload=payload,
        image_rgb=image_rgb,
        depth=depth,
        points=points,
        trajectories=trajectories,
        trajectory_times=trajectory_times,
        rollout_controls=rollout_controls,
        scores=scores,
        result=result,
        camera=camera,
        ego_extent=ego_extent,
        horizon_s=horizon_s,
        half_width=half_width,
        half_length=half_length)
    if debug_path is not None:
      result["debug_dir"] = str(debug_path)
    return result

  @staticmethod
  def _decode_image(encoded: str) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return np.asarray(image)

  def _depth_to_ego_points(self, depth: np.ndarray, width: int, height: int, camera: dict) -> np.ndarray:
    stride = self.point_stride
    v_coords = np.arange(0, height, stride, dtype=np.float32)
    u_coords = np.arange(0, width, stride, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    z = depth[0:height:stride, 0:width:stride].astype(np.float32)

    fx = camera.get("fx")
    fy = camera.get("fy")
    if fx is None or fy is None:
      fov = float(camera.get("fov", 110.0))
      focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
      fx = focal
      fy = focal
    cx = float(camera.get("cx", (width - 1) / 2.0))
    cy = float(camera.get("cy", (height - 1) / 2.0))
    camera_pos = np.asarray(camera.get("position", [-1.5, 0.0, 2.0]), dtype=np.float32).reshape(3)

    pinhole_x = (u_grid - cx) / float(fx) * z
    pinhole_y = (v_grid - cy) / float(fy) * z
    ego_x = z + camera_pos[0]
    ego_y = pinhole_x + camera_pos[1]
    ego_z = -pinhole_y + camera_pos[2]
    points = np.stack((ego_x, ego_y, ego_z), axis=-1).reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1)
    valid &= points[:, 0] >= self.min_depth
    valid &= points[:, 0] <= self.max_forward
    valid &= np.abs(points[:, 1]) <= self.max_lateral
    valid &= points[:, 2] >= self.min_height
    valid &= points[:, 2] <= self.max_height
    valid &= self._ego_hood_valid_mask(u_grid, v_grid, width, height).reshape(-1)
    return points[valid].astype(np.float32)

  def _ego_hood_valid_mask(self, u_grid, v_grid, width: int, height: int):
    if not self.ego_mask_enabled or self.ego_mask_bottom_frac <= 0.0 or self.ego_mask_center_width_frac <= 0.0:
      return np.ones_like(u_grid, dtype=bool)

    bottom_start = float(height) * (1.0 - self.ego_mask_bottom_frac)
    center = (float(width) - 1.0) * 0.5
    half_width = float(width) * self.ego_mask_center_width_frac * 0.5
    hood = (v_grid >= bottom_start) & (np.abs(u_grid - center) <= half_width)
    return ~hood

  def _score_trajectory(
      self,
      trajectory: np.ndarray,
      points: np.ndarray,
      horizon_s: float,
      half_width: float,
      half_length: float,
      trajectory_times=None) -> dict:
    if trajectory.ndim == 3:
      trajectory = trajectory[0]
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) == 0:
      return {"ttc_s": None, "score": -1.0, "collision": True, "reason": "invalid trajectory"}
    if len(points) == 0:
      return {"ttc_s": None, "score": horizon_s + 1.0, "collision": False, "reason": "empty point cloud"}

    centers = np.asarray(trajectory[:, :2], dtype=np.float32)
    path = np.vstack((np.zeros((1, 2), dtype=np.float32), centers))
    times = self._trajectory_times(trajectory_times, len(path), horizon_s)
    cloud_xy = points[:, :2]
    min_ttc = None
    min_detail = None

    for index in range(len(path) - 1):
      p0 = path[index]
      p1 = path[index + 1]
      segment = p1 - p0
      length = float(np.linalg.norm(segment))
      if length < 1e-3:
        continue
      tangent = segment / length
      rel = cloud_xy - p0.reshape(1, 2)
      along = rel @ tangent
      lateral_vec = rel - along.reshape(-1, 1) * tangent.reshape(1, 2)
      lateral = np.linalg.norm(lateral_vec, axis=1)
      inside = (
          (along >= -half_length) &
          (along <= length + half_length) &
          (lateral <= half_width)
      )
      if int(np.count_nonzero(inside)) < self.min_collision_points:
        continue
      clipped_along = np.clip(along[inside], 0.0, length)
      collision_time = times[index] + (clipped_along / length) * (times[index + 1] - times[index])
      min_index = int(np.argmin(collision_time))
      segment_ttc = float(collision_time[min_index])
      if min_ttc is None or segment_ttc < min_ttc:
        min_ttc = segment_ttc
        inside_indices = np.nonzero(inside)[0]
        point_index = int(inside_indices[min_index])
        min_detail = {
            "segment_index": int(index),
            "point_index": point_index,
            "point_ego": points[point_index].astype(float).tolist(),
            "along_m": float(clipped_along[min_index]),
            "lateral_m": float(lateral[point_index]),
            "segment_start_time_s": float(times[index]),
            "segment_end_time_s": float(times[index + 1]),
        }

    if min_ttc is None:
      return {"ttc_s": None, "score": horizon_s + 1.0, "collision": False, "reason": "no corridor overlap"}
    return {
        "ttc_s": min_ttc,
        "score": min_ttc,
        "collision": True,
        "reason": "corridor overlap",
        "collision_detail": min_detail or {},
    }

  @staticmethod
  def _trajectory_times(trajectory_times, path_len: int, horizon_s: float):
    fallback = np.linspace(0.0, horizon_s, path_len, dtype=np.float32)
    if trajectory_times is None:
      return fallback
    times = np.asarray(trajectory_times, dtype=np.float32).reshape(-1)
    if len(times) == path_len - 1:
      times = np.concatenate(([0.0], times))
    elif len(times) != path_len:
      return fallback
    if not np.isfinite(times).all():
      return fallback
    times = np.maximum.accumulate(times)
    if times[0] > 1e-4:
      times = times - times[0]
    times[0] = 0.0
    return np.clip(times, 0.0, horizon_s).astype(np.float32)

  def _rollout_planner_candidates(self, planner_candidates: dict, current_speed: float, horizon_s: float) -> dict:
    candidates = planner_candidates.get("candidates", planner_candidates)
    target_speeds = np.asarray(
        planner_candidates.get("target_speeds", [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0]),
        dtype=np.float32)
    uncertainty_weight = int(planner_candidates.get("uncertainty_weight", 1))
    brake_uncertainty_threshold = float(planner_candidates.get("brake_uncertainty_threshold", 0.5))

    trajectories = {}
    trajectory_times = {}
    controls = {}
    for name in self.ACTION_ORDER:
      candidate = candidates.get(name)
      if not candidate:
        continue
      checkpoints = np.asarray(candidate.get("checkpoints", []), dtype=np.float32)
      if checkpoints.ndim != 2 or checkpoints.shape[1] < 2 or len(checkpoints) == 0:
        continue
      probs = np.asarray(candidate.get("target_speed_probs", []), dtype=np.float32)
      target_speed = self._target_speed_from_probs(
          probs=probs,
          target_speeds=target_speeds,
          uncertainty_weight=uncertainty_weight,
          brake_uncertainty_threshold=brake_uncertainty_threshold)
      trajectory, times, control = self._rollout_single_candidate(
          checkpoints=checkpoints[:, :2],
          target_speed=target_speed,
          current_speed=current_speed,
          horizon_s=horizon_s)
      trajectories[name] = trajectory
      trajectory_times[name] = times
      controls[name] = control

    return {
        "trajectories": trajectories,
        "trajectory_times": trajectory_times,
        "controls": controls,
    }

  @staticmethod
  def _target_speed_from_probs(probs, target_speeds, uncertainty_weight, brake_uncertainty_threshold):
    if len(probs) == 0:
      return 0.0
    if len(target_speeds) != len(probs):
      target_speeds = np.linspace(0.0, 20.0, len(probs), dtype=np.float32)
    probs = probs.astype(np.float32)
    probs_sum = float(np.sum(probs))
    if probs_sum > 1e-6:
      probs = probs / probs_sum
    if uncertainty_weight:
      if float(probs[0]) > brake_uncertainty_threshold:
        return float(target_speeds[0])
      return float(np.sum(probs * target_speeds))
    return float(target_speeds[int(np.argmax(probs))])

  def _rollout_single_candidate(self, checkpoints, target_speed: float, current_speed: float, horizon_s: float):
    location = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    heading = 0.0
    speed = max(float(current_speed), 0.0)
    route_path = self._extended_route_path(checkpoints, max(float(current_speed), float(target_speed)) * horizon_s + 15.0)
    times = np.arange(self.rollout_dt, horizon_s + 1e-6, self.rollout_dt, dtype=np.float32)
    if len(times) == 0:
      times = np.asarray([horizon_s], dtype=np.float32)

    points = []
    steering_window = []
    controls = []
    for _ in times:
      steer = self._lateral_pid_step(route_path, speed, location[:2], heading, steering_window)
      brake = bool(target_speed < 0.01 or (speed / max(target_speed, 1e-5)) > 1.1)
      throttle, control_brake = self._get_throttle(brake=brake, target_speed=target_speed, speed=speed)
      location, heading, speed = self._bicycle_step(
          location=location,
          heading=heading,
          speed=speed,
          steer=steer,
          throttle=throttle,
          brake=control_brake)
      points.append([float(location[0]), float(location[1])])
      controls.append([float(steer), float(throttle), float(control_brake), float(speed)])

    return (
        np.asarray(points, dtype=np.float32).tolist(),
        times.astype(np.float32).tolist(),
        {
            "target_speed": float(target_speed),
            "first_control": controls[0] if controls else [0.0, 0.0, 0.0, float(current_speed)],
            "last_speed": controls[-1][3] if controls else float(current_speed),
        })

  @staticmethod
  def _extended_route_path(route_np, required_distance: float):
    route = np.asarray(route_np, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) == 0:
      return np.zeros((2, 2), dtype=np.float32)
    route = route[:, :2]
    path = np.vstack((np.zeros((1, 2), dtype=np.float32), route))
    if len(path) == 1:
      path = np.vstack((path, path[-1] + np.asarray([1.0, 0.0], dtype=np.float32)))

    segment_vectors = path[1:] - path[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length >= required_distance:
      return path

    last_vector = segment_vectors[-1]
    last_length = float(np.linalg.norm(last_vector))
    if last_length < 1e-3:
      valid = segment_vectors[segment_lengths > 1e-3]
      last_vector = valid[-1] if len(valid) > 0 else np.asarray([1.0, 0.0], dtype=np.float32)
      last_length = max(float(np.linalg.norm(last_vector)), 1e-3)
    direction = last_vector / last_length
    spacing = max(last_length, 1.0)
    needed = max(required_distance - total_length, 0.0)
    num_extra = int(np.ceil(needed / spacing)) + 2
    extras = [path[-1] + direction * spacing * (idx + 1) for idx in range(num_extra)]
    return np.vstack((path, np.asarray(extras, dtype=np.float32)))

  @classmethod
  def _lateral_pid_step(cls, route_np, current_speed, ego_vehicle_location, ego_vehicle_rotation, window):
    lateral_k_p = 3.118357247806046
    lateral_k_d = 1.3782508892109167
    lateral_k_i = 0.6406067986034124
    lateral_speed_scale = 0.9755321901954155
    lateral_speed_offset = 1.9152884533402488
    lateral_n = 6

    current_speed_kmh = float(current_speed) * 3.6
    lookahead_m = float(np.clip(lateral_speed_scale * current_speed_kmh + lateral_speed_offset, 24, 105) / 10 - 2)
    lookahead_m = max(lookahead_m, 0.4)
    target_point = cls._path_target_ahead(route_np, np.asarray(ego_vehicle_location, dtype=np.float32), lookahead_m)
    desired_heading_vec = target_point - np.asarray(ego_vehicle_location, dtype=np.float32)
    yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
    heading_error = (yaw_path - float(ego_vehicle_rotation)) % (2 * np.pi)
    heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi
    heading_error = heading_error * 180.0 / np.pi / 90.0
    window.append(float(heading_error))
    del window[:-lateral_n]
    derivative = 0.0 if len(window) == 1 else window[-1] - window[-2]
    integral = float(np.mean(window))
    return float(np.clip(lateral_k_p * heading_error + lateral_k_d * derivative + lateral_k_i * integral, -1.0, 1.0))

  @staticmethod
  def _path_target_ahead(path, location, lookahead_m: float):
    path = np.asarray(path, dtype=np.float32)
    if len(path) == 0:
      return location + np.asarray([lookahead_m, 0.0], dtype=np.float32)
    if len(path) == 1:
      return path[0]

    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = 0.0
    best_sq_dist = float("inf")
    for index, (p0, segment, length) in enumerate(zip(path[:-1], segments, lengths)):
      if float(length) < 1e-4:
        continue
      ratio = float(np.clip(np.dot(location - p0, segment) / (length * length), 0.0, 1.0))
      projection = p0 + ratio * segment
      sq_dist = float(np.sum((projection - location) ** 2))
      if sq_dist < best_sq_dist:
        best_sq_dist = sq_dist
        best_distance = float(cumulative[index] + ratio * length)

    target_distance = min(best_distance + lookahead_m, float(cumulative[-1]))
    segment_index = int(np.searchsorted(cumulative, target_distance, side="right") - 1)
    segment_index = int(np.clip(segment_index, 0, len(lengths) - 1))
    denom = max(float(lengths[segment_index]), 1e-4)
    ratio = (target_distance - float(cumulative[segment_index])) / denom
    return path[segment_index] + ratio * segments[segment_index]

  @staticmethod
  def _get_throttle(brake: bool, target_speed: float, speed: float):
    if target_speed < 1e-5 or brake:
      return 0.0, True
    if target_speed < 1.0 / 3.6:
      target_speed = 1.0 / 3.6

    speed_kmh = speed * 3.6
    target_speed_kmh = target_speed * 3.6
    params = np.asarray([
        1.1990342347353184,
        -0.8057602384167799,
        1.710818710950062,
        0.921890257450335,
        1.556497522998393,
        -0.7013479734904027,
        1.031266635497984,
    ], dtype=np.float32)
    speed_error = target_speed_kmh - speed_kmh
    if speed_error > 1.89:
      return 1.0, False
    if speed_kmh / target_speed_kmh > params[-1] or brake:
      return 0.0, True

    speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
    speed_norm = speed_kmh / 100.0
    features = np.asarray([
        speed_norm,
        speed_norm**2,
        100 * speed_error_cl,
        speed_error_cl**2,
        speed_norm * speed_error_cl,
        speed_norm**2 * speed_error_cl,
    ], dtype=np.float32)
    return float(np.clip(features @ params[:-1], 0.0, 1.0)), False

  def _bicycle_step(self, location, heading, speed, steer, throttle, brake):
    front_wheel_base = -0.090769015
    rear_wheel_base = 1.4178275
    steering_gain = 0.36848336
    throttle_threshold = 0.3
    throttle_values = np.asarray([
        9.63873001e-01,
        4.37535692e-04,
        -3.80192912e-01,
        1.74950069e+00,
        9.16787414e-02,
        -7.05461530e-02,
        -1.05996152e-03,
        6.71079346e-04,
    ], dtype=np.float32)
    wheel_angle = steering_gain * float(steer)
    slip_angle = np.arctan(rear_wheel_base / (front_wheel_base + rear_wheel_base) * np.tan(wheel_angle))

    if brake:
      next_speed = max(float(speed) - self.brake_deceleration * self.rollout_dt, 0.0)
      displacement_speed = 0.5 * (float(speed) + next_speed)
    else:
      throttle = float(np.clip(throttle, 0.0, 1.0))
      if throttle < throttle_threshold:
        next_speed = float(speed)
      else:
        speed_kph = float(speed) * 3.6
        features = np.asarray([
            speed_kph,
            speed_kph**2,
            throttle,
            throttle**2,
            speed_kph * throttle,
            speed_kph * throttle**2,
            speed_kph**2 * throttle,
            speed_kph**2 * throttle**2,
        ], dtype=np.float32)
        next_speed = float(features @ throttle_values) / 3.6
      next_speed = max(next_speed, 0.0)
      displacement_speed = float(speed)

    next_x = float(location[0]) + displacement_speed * np.cos(float(heading) + slip_angle) * self.rollout_dt
    next_y = float(location[1]) + displacement_speed * np.sin(float(heading) + slip_angle) * self.rollout_dt
    next_heading = float(heading) + displacement_speed / rear_wheel_base * np.sin(slip_angle) * self.rollout_dt
    next_location = np.asarray([next_x, next_y, float(location[2])], dtype=np.float32)
    return next_location, next_heading, next_speed

  def _select(self, scores: dict) -> str:
    selected = "original"
    selected_score = float(scores.get(selected, {}).get("score", -1.0))
    for name in self.ACTION_ORDER[1:]:
      score = float(scores.get(name, {}).get("score", -1.0))
      if score > selected_score + self.tie_epsilon:
        selected = name
        selected_score = score
    return selected

  @staticmethod
  def _format_reason(selected: str, scores: dict) -> str:
    parts = []
    for name in DepthTTCEvaluator.ACTION_ORDER:
      item = scores.get(name, {})
      ttc = item.get("ttc_s")
      parts.append(f"{name}={'safe' if ttc is None else f'{float(ttc):.2f}s'}")
    return f"selected {selected}; " + ", ".join(parts)

  def _save_debug_artifacts(
      self,
      payload: dict,
      image_rgb,
      depth,
      points,
      trajectories: dict,
      trajectory_times: dict,
      rollout_controls: dict,
      scores: dict,
      result: dict,
      camera: dict,
      ego_extent: dict,
      horizon_s: float,
      half_width: float,
      half_length: float):
    if self.debug_dir is None:
      return None
    with self._debug_lock:
      self._debug_counter += 1
      debug_index = self._debug_counter
    if (debug_index - 1) % self.debug_save_every != 0:
      return None

    frame_id = payload.get("frame_id", "unknown")
    request_dir = self.debug_dir / f"frame_{frame_id}_{debug_index:06d}"
    request_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_rgb).save(request_dir / "front.png")
    np.save(request_dir / "metric_depth.npy", depth.astype(np.float32))
    Image.fromarray(self._depth_visualization(depth)).save(request_dir / "depth_vis.png")
    np.savez_compressed(request_dir / "points_ego.npz", points=points.astype(np.float32))
    self._write_point_cloud_ply(request_dir / "points_ego.ply", points)

    traj_json = {
        name: np.asarray(trajectory, dtype=np.float32).tolist()
        for name, trajectory in trajectories.items()
    }
    (request_dir / "trajectories.json").write_text(
        json.dumps(traj_json, indent=2),
        encoding="utf-8")
    (request_dir / "trajectory_times.json").write_text(
        json.dumps(trajectory_times, indent=2),
        encoding="utf-8")
    metadata = {
        "request": {
            "frame_id": payload.get("frame_id"),
            "speed": payload.get("speed"),
            "horizon_s": horizon_s,
            "camera": camera,
            "ego_extent": ego_extent,
            "half_width": half_width,
            "half_length": half_length,
            "ego_hood_mask": {
                "enabled": self.ego_mask_enabled,
                "bottom_frac": self.ego_mask_bottom_frac,
                "center_width_frac": self.ego_mask_center_width_frac,
            },
            "rollout": {
                "dt": self.rollout_dt,
                "brake_deceleration_m_s2": self.brake_deceleration,
            },
            "trajectory_times": trajectory_times,
            "rollout_controls": rollout_controls,
        },
        "response": result,
    }
    (request_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8")

    bev = self._draw_bev_debug(
        points=points,
        trajectories=trajectories,
        scores=scores,
        selected=result.get("selected_trajectory", "original"),
        half_width=half_width,
        half_length=half_length,
        horizon_s=horizon_s)
    Image.fromarray(bev).save(request_dir / "bev_debug.png")

    image_overlay = self._draw_image_trajectory_debug(
        image_rgb=image_rgb,
        trajectories=trajectories,
        scores=scores,
        selected=result.get("selected_trajectory", "original"),
        camera=camera,
        half_width=half_width)
    Image.fromarray(image_overlay).save(request_dir / "front_traj_debug.png")
    mask_overlay = self._draw_ego_mask_debug(image_rgb)
    Image.fromarray(mask_overlay).save(request_dir / "front_ego_mask_debug.png")
    return request_dir

  def _draw_ego_mask_debug(self, image_rgb):
    image = np.ascontiguousarray(image_rgb.copy())
    if not self.ego_mask_enabled or self.ego_mask_bottom_frac <= 0.0 or self.ego_mask_center_width_frac <= 0.0:
      return image
    height, width = image.shape[:2]
    y0 = int(round(float(height) * (1.0 - self.ego_mask_bottom_frac)))
    center = (float(width) - 1.0) * 0.5
    half_width = float(width) * self.ego_mask_center_width_frac * 0.5
    x0 = int(max(0, round(center - half_width)))
    x1 = int(min(width - 1, round(center + half_width)))
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, height - 1), (255, 80, 40), -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    cv2.rectangle(image, (x0, y0), (x1, height - 1), (255, 80, 40), 2)
    cv2.putText(image, "ignored ego hood mask", (x0 + 8, max(22, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 40), 2)
    return image

  @staticmethod
  def _depth_visualization(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if not np.any(valid):
      return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if hi <= lo:
      hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

  def _write_point_cloud_ply(self, path: Path, points):
    sampled = self._sample_points(points, self.debug_max_points)
    if len(sampled) == 0:
      path.write_text(
          "ply\nformat ascii 1.0\nelement vertex 0\n"
          "property float x\nproperty float y\nproperty float z\nend_header\n",
          encoding="utf-8")
      return
    z_values = sampled[:, 2]
    z_norm = (z_values - z_values.min()) / max(float(z_values.max() - z_values.min()), 1e-3)
    colors = cv2.applyColorMap((z_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    colors = colors[:, ::-1]
    with path.open("w", encoding="utf-8") as handle:
      handle.write("ply\nformat ascii 1.0\n")
      handle.write(f"element vertex {len(sampled)}\n")
      handle.write("property float x\nproperty float y\nproperty float z\n")
      handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
      handle.write("end_header\n")
      for point, color in zip(sampled, colors):
        handle.write(
            f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

  def _draw_bev_debug(self, points, trajectories, scores, selected, half_width, half_length, horizon_s):
    height = 760
    width = 920
    margin = 50
    max_forward = max(self.max_forward, 10.0)
    max_lateral = max(self.max_lateral, 4.0)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_left = margin
    plot_right = width - margin
    plot_top = margin
    plot_bottom = height - margin

    def to_pixel(xy):
      arr = np.asarray(xy, dtype=np.float32)
      u = plot_left + ((arr[..., 1] + max_lateral) / (2.0 * max_lateral)) * (plot_right - plot_left)
      v = plot_bottom - (arr[..., 0] / max_forward) * (plot_bottom - plot_top)
      return np.stack((u, v), axis=-1).astype(np.int32)

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (210, 210, 210), 1)
    for x_m in range(0, int(max_forward) + 1, 10):
      y = int(plot_bottom - (x_m / max_forward) * (plot_bottom - plot_top))
      cv2.line(canvas, (plot_left, y), (plot_right, y), (235, 235, 235), 1)
      cv2.putText(canvas, f"{x_m}m", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    for y_m in range(-int(max_lateral), int(max_lateral) + 1, 4):
      x = int(plot_left + ((y_m + max_lateral) / (2.0 * max_lateral)) * (plot_right - plot_left))
      cv2.line(canvas, (x, plot_top), (x, plot_bottom), (240, 240, 240), 1)

    sampled_points = self._sample_points(points, self.debug_max_points)
    if len(sampled_points) > 0:
      pixels = to_pixel(sampled_points[:, :2])
      valid = (
          (pixels[:, 0] >= plot_left) & (pixels[:, 0] < plot_right) &
          (pixels[:, 1] >= plot_top) & (pixels[:, 1] < plot_bottom))
      pixels = pixels[valid]
      canvas[pixels[:, 1], pixels[:, 0]] = (120, 120, 120)

    ego_box = np.asarray([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width],
    ], dtype=np.float32)
    cv2.polylines(canvas, [to_pixel(ego_box).reshape(-1, 1, 2)], True, (0, 0, 0), 2)

    colors = {
        "original": (30, 150, 30),
        "brake": (230, 120, 20),
        "left": (30, 110, 230),
        "right": (190, 40, 190),
    }
    overlay = canvas.copy()
    for name in self.ACTION_ORDER:
      trajectory = trajectories.get(name)
      if trajectory is None:
        continue
      color = colors[name]
      path = np.vstack((np.zeros((1, 2), dtype=np.float32), np.asarray(trajectory, dtype=np.float32)[:, :2]))
      for polygon in self._trajectory_corridor_polygons(path, half_width, half_length):
        cv2.fillPoly(overlay, [to_pixel(polygon).reshape(-1, 1, 2)], color)
      cv2.polylines(canvas, [to_pixel(path).reshape(-1, 1, 2)], False, color, 3 if name == selected else 1)
    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)

    legend_y = 22
    cv2.putText(canvas, "ego-frame BEV: x forward, y lateral", (margin, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    legend_y += 24
    for name in self.ACTION_ORDER:
      score = scores.get(name, {})
      ttc = score.get("ttc_s")
      label = f"{name}: {'safe' if ttc is None else f'{float(ttc):.2f}s'}"
      prefix = "* " if name == selected else "  "
      cv2.putText(canvas, prefix + label, (margin, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)
      legend_y += 22
    return canvas

  def _draw_image_trajectory_debug(self, image_rgb, trajectories, scores, selected, camera, half_width):
    image = np.ascontiguousarray(image_rgb.copy())
    overlay = image.copy()
    colors = {
        "original": (30, 150, 30),
        "brake": (230, 120, 20),
        "left": (30, 110, 230),
        "right": (190, 40, 190),
    }
    for name in self.ACTION_ORDER:
      trajectory = trajectories.get(name)
      if trajectory is None:
        continue
      centers = np.asarray(trajectory, dtype=np.float32)[:, :2]
      corridor_polygons = self._trajectory_corridor_polygons(
          np.vstack((np.zeros((1, 2), dtype=np.float32), centers)),
          half_width=half_width,
          half_length=0.0)
      for polygon in corridor_polygons:
        projected = self._project_ego_points_to_image(polygon, camera, image.shape[1], image.shape[0])
        if projected is not None and len(projected) >= 3:
          cv2.fillPoly(overlay, [projected.reshape(-1, 1, 2)], colors[name])
      center_projected = self._project_ego_points_to_image(centers, camera, image.shape[1], image.shape[0])
      if center_projected is not None and len(center_projected) >= 2:
        cv2.polylines(image, [center_projected.reshape(-1, 1, 2)], False, colors[name], 3 if name == selected else 1)
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0)
    y = 24
    for name in self.ACTION_ORDER:
      score = scores.get(name, {})
      ttc = score.get("ttc_s")
      label = f"{'*' if name == selected else ' '} {name}: {'safe' if ttc is None else f'{float(ttc):.2f}s'}"
      cv2.putText(image, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
      y += 24
    return image

  @staticmethod
  def _trajectory_corridor_polygons(path, half_width, half_length):
    polygons = []
    path = np.asarray(path, dtype=np.float32)
    for index in range(len(path) - 1):
      p0 = path[index]
      p1 = path[index + 1]
      tangent = p1 - p0
      length = float(np.linalg.norm(tangent))
      if length < 1e-3:
        continue
      tangent = tangent / length
      normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
      start = p0 - tangent * half_length
      end = p1 + tangent * half_length
      polygons.append(np.asarray([
          start - normal * half_width,
          end - normal * half_width,
          end + normal * half_width,
          start + normal * half_width,
      ], dtype=np.float32))
    return polygons

  @staticmethod
  def _project_ego_points_to_image(points, camera, width, height):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
      return None
    z_m = float(camera.get("trajectory_z_m", 0.0))
    points_3d = np.concatenate(
        (points[:, :2], np.full((len(points), 1), z_m, dtype=np.float32)),
        axis=1)
    camera_pos = np.asarray(camera.get("position", [-1.5, 0.0, 2.0]), dtype=np.float32).reshape(3)
    camera_points = points_3d - camera_pos.reshape(1, 3)
    pinhole_z = camera_points[:, 0]
    valid = pinhole_z > 0.1
    if not np.any(valid):
      return None
    fx = float(camera.get("fx", width))
    fy = float(camera.get("fy", width))
    cx = float(camera.get("cx", (width - 1) / 2.0))
    cy = float(camera.get("cy", (height - 1) / 2.0))
    u = fx * (camera_points[:, 1] / pinhole_z) + cx
    v = fy * (-camera_points[:, 2] / pinhole_z) + cy
    valid &= (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    if not np.any(valid):
      return None
    return np.stack((u[valid], v[valid]), axis=1).astype(np.int32)

  @staticmethod
  def _sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= max_points:
      return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


class Handler(BaseHTTPRequestHandler):
  evaluator: DepthTTCEvaluator | None = None

  def do_GET(self):  # pylint: disable=invalid-name
    if self.path != "/health":
      self.send_error(404)
      return
    self._write_json({"ok": True})

  def do_POST(self):  # pylint: disable=invalid-name
    if self.path != "/score":
      self.send_error(404)
      return
    try:
      length = int(self.headers.get("Content-Length", "0"))
      payload = json.loads(self.rfile.read(length).decode("utf-8"))
      started_at = time.perf_counter()
      result = self.evaluator.score(payload)
      elapsed_ms = (time.perf_counter() - started_at) * 1000.0
      print(
          f"[DepthTTCServer] frame={payload.get('frame_id')} "
          f"selected={result.get('selected_trajectory')} action={result.get('action')} "
          f"points={result.get('point_count')} elapsed_ms={elapsed_ms:.1f} "
          f"{result.get('reason', '')}",
          flush=True)
      self._write_json(result)
    except Exception as exc:  # pylint: disable=broad-except
      self._write_json(
          {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
          status=500)

  def log_message(self, fmt, *args):
    if _strtobool(os.environ.get("DEPTH_TTC_SERVER_VERBOSE", "0")):
      super().log_message(fmt, *args)

  def _write_json(self, payload: dict, status: int = 200):
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def parse_args():
  repo_root = Path(__file__).resolve().parents[2]
  depth_root = repo_root / "Depth-Anything-V2"
  default_checkpoint = depth_root / "checkpoints" / "depth_anything_v2_metric_vkitti_vitl.pth"
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default=os.environ.get("DEPTH_TTC_HOST", "127.0.0.1"))
  parser.add_argument("--port", type=int, default=int(os.environ.get("DEPTH_TTC_PORT", 8766)))
  parser.add_argument("--depth-anything-root", default=os.environ.get("DEPTH_ANYTHING_ROOT", str(depth_root)))
  parser.add_argument("--checkpoint", default=os.environ.get("DEPTH_TTC_CHECKPOINT", str(default_checkpoint)))
  parser.add_argument("--encoder", default=os.environ.get("DEPTH_TTC_ENCODER", "vitl"), choices=["vits", "vitb", "vitl", "vitg"])
  parser.add_argument("--device", default=os.environ.get("DEPTH_TTC_DEVICE", "cuda:0"))
  parser.add_argument("--input-size", type=int, default=int(os.environ.get("DEPTH_TTC_INPUT_SIZE", 518)))
  parser.add_argument("--max-depth", type=float, default=float(os.environ.get("DEPTH_TTC_MAX_DEPTH", 80.0)))
  parser.add_argument("--horizon-s", type=float, default=float(os.environ.get("DEPTH_TTC_HORIZON_S", 2.0)))
  parser.add_argument("--rollout-dt", type=float, default=float(os.environ.get("DEPTH_TTC_ROLLOUT_DT", 0.1)))
  parser.add_argument("--brake-deceleration", type=float, default=float(os.environ.get("DEPTH_TTC_BRAKE_DECEL_M_S2", 4.952399)))
  parser.add_argument("--point-stride", type=int, default=int(os.environ.get("DEPTH_TTC_POINT_STRIDE", 2)))
  parser.add_argument("--min-depth", type=float, default=float(os.environ.get("DEPTH_TTC_MIN_DEPTH_M", 0.5)))
  parser.add_argument("--max-forward", type=float, default=float(os.environ.get("DEPTH_TTC_MAX_FORWARD_M", 60.0)))
  parser.add_argument("--max-lateral", type=float, default=float(os.environ.get("DEPTH_TTC_MAX_LATERAL_M", 12.0)))
  parser.add_argument("--min-height", type=float, default=float(os.environ.get("DEPTH_TTC_MIN_HEIGHT_M", 0.2)))
  parser.add_argument("--max-height", type=float, default=float(os.environ.get("DEPTH_TTC_MAX_HEIGHT_M", 3.0)))
  parser.add_argument("--corridor-margin", type=float, default=float(os.environ.get("DEPTH_TTC_CORRIDOR_MARGIN_M", 0.35)))
  parser.add_argument("--min-collision-points", type=int, default=int(os.environ.get("DEPTH_TTC_MIN_COLLISION_POINTS", 3)))
  parser.add_argument("--tie-epsilon", type=float, default=float(os.environ.get("DEPTH_TTC_TIE_EPSILON_S", 0.05)))
  parser.add_argument("--ego-mask", default=os.environ.get("DEPTH_TTC_EGO_MASK", "1"))
  parser.add_argument("--ego-mask-bottom-frac", type=float, default=float(os.environ.get("DEPTH_TTC_EGO_MASK_BOTTOM_FRAC", 0.08)))
  parser.add_argument("--ego-mask-center-width-frac", type=float, default=float(os.environ.get("DEPTH_TTC_EGO_MASK_CENTER_WIDTH_FRAC", 0.26)))
  parser.add_argument("--debug-dir", default=os.environ.get("DEPTH_TTC_DEBUG_DIR", ""))
  parser.add_argument("--debug-save-every", type=int, default=int(os.environ.get("DEPTH_TTC_DEBUG_SAVE_EVERY", 1)))
  parser.add_argument("--debug-max-points", type=int, default=int(os.environ.get("DEPTH_TTC_DEBUG_MAX_POINTS", 80000)))
  return parser.parse_args()


def main():
  args = parse_args()
  Handler.evaluator = DepthTTCEvaluator(args)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Depth TTC server listening on http://{args.host}:{args.port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()

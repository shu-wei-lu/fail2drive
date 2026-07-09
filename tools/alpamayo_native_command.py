#!/usr/bin/env python3
"""Derive activation-steering commands from Alpamayo native trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryCommandConfig:
  horizon_s: float = 6.4
  brake_max_avg_speed: float = 1.0
  brake_max_forward: float = 3.0
  lateral_threshold: float = 1.0
  lateral_deadband_forward: float = 2.0
  left_is_negative_y: bool = True


def derive_command_from_pred_xyz(pred_xyz: Any, config: TrajectoryCommandConfig) -> dict:
  """Convert Alpamayo ``pred_xyz`` into a discrete steering command.

  Expected input shape is compatible with Alpamayo 1.5:
  [B, num_traj_sets, num_traj_samples, T, 3]. The server currently uses B=1
  and usually num_traj_samples=1. If more samples are requested, this averages
  the predicted trajectories before deriving the command.
  """
  traj = _to_numpy(pred_xyz)
  if traj.ndim < 3 or traj.shape[-1] < 2:
    raise ValueError(f"pred_xyz must have shape [..., T, >=2], got {traj.shape}")

  samples = traj.reshape(-1, traj.shape[-2], traj.shape[-1])
  xy_samples = samples[:, :, :2].astype(np.float32)
  xy = np.nanmean(xy_samples, axis=0)
  if not np.isfinite(xy).all():
    raise ValueError("pred_xyz contains non-finite values after sample averaging")

  deltas = np.diff(xy, axis=0)
  path_length = float(np.linalg.norm(deltas, axis=1).sum())
  horizon_s = max(float(config.horizon_s), 1e-3)
  avg_speed = path_length / horizon_s
  final_forward = float(xy[-1, 0])
  final_lateral = float(xy[-1, 1])
  max_abs_lateral = float(np.max(np.abs(xy[:, 1])))

  if avg_speed <= config.brake_max_avg_speed or final_forward <= config.brake_max_forward:
    action = "brake"
    reason = (
        f"native trajectory is slow/short: avg_speed={avg_speed:.2f}m/s "
        f"final_forward={final_forward:.2f}m"
    )
  elif final_forward < config.lateral_deadband_forward:
    action = "none"
    reason = (
        "native trajectory has insufficient forward progress for lateral decision: "
        f"final_forward={final_forward:.2f}m"
    )
  elif abs(final_lateral) >= config.lateral_threshold:
    action = _lateral_to_action(final_lateral, config.left_is_negative_y)
    reason = (
        f"native trajectory lateral offset indicates {action}: "
        f"final_lateral={final_lateral:.2f}m"
    )
  else:
    action = "none"
    reason = (
        "native trajectory stays near current lane: "
        f"final_lateral={final_lateral:.2f}m avg_speed={avg_speed:.2f}m/s"
    )

  return {
      "action": action,
      "confidence": 1.0 if action != "none" else 0.0,
      "reason": reason,
      "trajectory_metrics": {
          "avg_speed": avg_speed,
          "path_length": path_length,
          "final_forward": final_forward,
          "final_lateral": final_lateral,
          "max_abs_lateral": max_abs_lateral,
          "num_samples": int(xy_samples.shape[0]),
          "num_waypoints": int(xy.shape[0]),
          "horizon_s": horizon_s,
      },
      "trajectory_preview_xy": _preview_xy(xy),
  }


def derive_command_from_cot(
    cot: str,
    speed_mode: str = "loose",
    lateral_mode: str = "loose",
    weak_brake_mode: str = "none") -> dict:
  """Convert Alpamayo chain-of-thought text into a discrete command.

  This parser intentionally keys on the maneuver phrase, not every occurrence
  of left/right. For example, "nudge left due to a blocker on the right side"
  should be left, while "vehicle on the right side" alone should not steer.
  """
  text = " ".join(str(cot or "").strip().split())
  lower = text.lower()

  direction = _direction_from_cot(lower, lateral_mode=lateral_mode)
  if direction is not None:
    return {
        "action": direction,
        "confidence": 1.0,
        "reason": f"cot maneuver phrase indicates {direction}: {text[:180]}",
        "cot_command_source": "maneuver_phrase",
        "cot_lateral_mode": lateral_mode,
    }

  if _brake_from_cot(lower, speed_mode=speed_mode):
    return {
        "action": "brake",
        "confidence": 1.0,
        "reason": f"cot speed-control phrase indicates brake/slow: {text[:180]}",
        "cot_command_source": "speed_phrase",
        "cot_speed_mode": speed_mode,
        "cot_lateral_mode": lateral_mode,
        "cot_weak_brake_mode": weak_brake_mode,
    }

  if _weak_brake_from_cot(lower, weak_brake_mode=weak_brake_mode):
    return {
        "action": "brake_weak",
        "confidence": 1.0,
        "reason": f"cot weak speed-control phrase indicates gentle brake: {text[:180]}",
        "cot_command_source": "weak_speed_phrase",
        "cot_speed_mode": speed_mode,
        "cot_lateral_mode": lateral_mode,
        "cot_weak_brake_mode": weak_brake_mode,
    }

  return {
      "action": "none",
      "confidence": 0.0,
      "reason": f"cot does not contain an explicit maneuver command: {text[:180]}",
      "cot_command_source": "none",
      "cot_speed_mode": speed_mode,
      "cot_lateral_mode": lateral_mode,
      "cot_weak_brake_mode": weak_brake_mode,
  }


def _to_numpy(value: Any) -> np.ndarray:
  if hasattr(value, "detach"):
    value = value.detach().cpu().float().numpy()
  return np.asarray(value)


def _lateral_to_action(lateral: float, left_is_negative_y: bool) -> str:
  if math.copysign(1.0, lateral) < 0:
    return "left" if left_is_negative_y else "right"
  return "right" if left_is_negative_y else "left"


def _direction_from_cot(lower: str, lateral_mode: str = "loose") -> str | None:
  patterns = (
      (r"\b(?:nudge|steer|turn|veer|move|merge|shift)\s+(?:to\s+the\s+)?left\b", "left"),
      (r"\b(?:nudge|steer|turn|veer|move|merge|shift)\s+(?:to\s+the\s+)?right\b", "right"),
      (r"\b(?:change|switch)\s+lanes?\s+(?:to\s+the\s+)?left\b", "left"),
      (r"\b(?:change|switch)\s+lanes?\s+(?:to\s+the\s+)?right\b", "right"),
      (r"\b(?:keep|stay)\s+(?:to\s+the\s+)?left\b", "left"),
      (r"\b(?:keep|stay)\s+(?:to\s+the\s+)?right\b", "right"),
  )
  matches: list[tuple[int, str]] = []
  for pattern, action in patterns:
    match = re.search(pattern, lower)
    if match:
      matches.append((match.start(), action))
  if not matches:
    return None
  matches.sort(key=lambda item: item[0])
  if lateral_mode == "strict" and not _strict_lateral_allowed(lower):
    return None
  return matches[0][1]


def _strict_lateral_allowed(lower: str) -> bool:
  """Allow lateral CoT commands only when the ego lane is explicitly compromised."""
  allow_patterns = (
      r"\b(?:lane|path|road)\s+(?:is\s+)?(?:blocked|obstructed)\b",
      r"\b(?:blocked|blocking|obstructing)\s+(?:our|my|the|ego)?\s*(?:lane|path)\b",
      r"\b(?:stopped|parked)\s+(?:vehicle|car|truck|bus).*?\b(?:blocking|obstructing)\b.*?\b(?:lane|path)\b",
      r"\b(?:vehicle|car|truck|bus).*?\b(?:encroaching|drifting|merging|cutting)\b.*?\b(?:into|in)\s+(?:our|my|the|ego)?\s*lane\b",
      r"\b(?:vehicle|car|truck|bus).*?\b(?:encroaching|drifting|merging|cutting)\b.*?\bfrom\s+(?:the\s+)?(?:left|right)\s+side\s+of\s+(?:our|my|the|ego)?\s*lane\b",
      r"\b(?:encroaching|drifting|merging|cutting)\b.*?\b(?:into|in)\s+(?:our|my|the|ego)?\s*lane\b",
      r"\boncoming\s+(?:vehicle|car|truck|bus).*?\b(?:in|into)\s+(?:our|my|the|ego)?\s*lane\b",
      r"\b(?:vehicle|car|truck|bus)\s+(?:in|inside)\s+(?:our|my|the|ego)?\s*lane\b.*?\b(?:oncoming|wrong[- ]?way)\b",
  )
  return any(re.search(pattern, lower) for pattern in allow_patterns)


def _brake_from_cot(lower: str, speed_mode: str = "loose") -> bool:
  if re.search(r"\b(?:accelerate|proceed|go through)\b", lower):
    return False
  if speed_mode == "strict":
    return bool(re.search(
        r"\b(?:stop|brake|slow down|decelerate|yield|reduce speed)\b",
        lower))
  return bool(re.search(
      r"\b(?:stop|brake|slow down|decelerate|yield|keep distance|maintain distance|adjust speed)\b",
      lower))


def _weak_brake_from_cot(lower: str, weak_brake_mode: str = "none") -> bool:
  if weak_brake_mode == "none":
    return False
  patterns = []
  if weak_brake_mode in ("distance", "distance_animal"):
    patterns.extend((
        r"\bkeep distance\b",
        r"\bmaintain(?:ing)?\s+(?:a\s+)?(?:safe\s+)?(?:following\s+)?distance\b",
        r"\blead vehicle\b",
        r"\bvehicle\s+directly\s+ahead\b",
        r"\bdirectly\s+ahead\s+in\s+(?:our|my|the|ego)?\s*lane\b",
    ))
  if weak_brake_mode in ("hazard", "distance_animal"):
    patterns.extend((
        r"\banimals?\b",
        r"\bpedestrians?\b",
        r"\bcross(?:ing|es)?\b",
        r"\bobstacles?\b",
        r"\bblocking\s+(?:our|my|the|ego)?\s*(?:lane|path)\b",
        r"\b(?:entering|encroaching|intruding)\s+(?:our|my|the|ego)?\s*(?:lane|path)\b",
    ))
  return any(re.search(pattern, lower) for pattern in patterns)


def _preview_xy(xy: np.ndarray, max_points: int = 8) -> list[list[float]]:
  if xy.shape[0] <= max_points:
    selected = xy
  else:
    indices = np.linspace(0, xy.shape[0] - 1, max_points).round().astype(int)
    selected = xy[indices]
  return [[round(float(x), 3), round(float(y), 3)] for x, y in selected[:, :2]]

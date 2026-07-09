#!/usr/bin/env python3
"""Build a cleaner HiP-AD left-lane-change activation vector from meta files.

The default policy builds a moving left-initiation vector:

- positive: moving, no-brake, strong immediate left steering intent
- negative: no-brake, lateral-neutral normal planning, matched to positives
  by desired speed, ego speed, and temporal path length
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--collection-root", default="steering/hipad")
  parser.add_argument(
      "--output-dir",
      default=None,
      help="Defaults to <collection-root>/post_process/left_change_lane_codex.",
  )
  parser.add_argument("--model-index", type=int, default=0)
  parser.add_argument("--negative-per-positive", type=int, default=4)
  parser.add_argument("--max-frames-per-run", type=int, default=0)

  parser.add_argument("--positive-min-speed", type=float, default=1.0)
  parser.add_argument("--positive-max-speed", type=float, default=6.0)
  parser.add_argument("--positive-min-desired-speed", type=float, default=4.0)
  parser.add_argument("--positive-min-temp-path-len", type=float, default=4.0)
  parser.add_argument("--positive-max-temp-path-len", type=float, default=8.0)
  parser.add_argument("--positive-min-temp-last-y", type=float, default=3.8)
  parser.add_argument("--positive-min-left-steer", type=float, default=0.08)
  parser.add_argument("--positive-min-left-spat-first", type=float, default=0.12)
  parser.add_argument("--positive-min-left-aim", type=float, default=0.15)
  parser.add_argument("--positive-min-left-angle", type=float, default=0.04)
  parser.add_argument("--positive-max-brake", type=float, default=0.1)

  parser.add_argument("--negative-min-speed", type=float, default=1.0)
  parser.add_argument("--negative-max-speed", type=float, default=6.5)
  parser.add_argument("--negative-max-desired-speed", type=float, default=7.0)
  parser.add_argument("--negative-min-temp-path-len", type=float, default=4.0)
  parser.add_argument("--negative-max-temp-path-len", type=float, default=8.5)
  parser.add_argument("--negative-max-brake", type=float, default=0.1)
  parser.add_argument("--negative-max-spat-mean-abs", type=float, default=0.25)
  parser.add_argument("--negative-max-spat-last-abs", type=float, default=0.4)
  parser.add_argument("--negative-max-aim-abs", type=float, default=0.2)
  parser.add_argument("--negative-max-angle-abs", type=float, default=0.05)

  parser.add_argument("--match-desired-scale", type=float, default=2.0)
  parser.add_argument("--match-speed-scale", type=float, default=2.0)
  parser.add_argument("--match-temp-scale", type=float, default=3.0)
  return parser.parse_args()


def path_len(points: list[list[float]]) -> float:
  prev_x = 0.0
  prev_y = 0.0
  total = 0.0
  for point in points:
    x = float(point[0])
    y = float(point[1])
    total += math.hypot(x - prev_x, y - prev_y)
    prev_x = x
    prev_y = y
  return total


def plan_metrics(meta: dict[str, Any]) -> dict[str, float] | None:
  try:
    plan_spat = meta["plan_spat"]
    plan_temp = meta["plan_temp"]
    spat_xs = [float(point[0]) for point in plan_spat]
    temp_xs = [float(point[0]) for point in plan_temp]
    temp_ys = [float(point[1]) for point in plan_temp]
    return {
        "speed": float(meta.get("speed", 0.0)),
        "steer": float(meta.get("steer", 0.0)),
        "throttle": float(meta.get("throttle", 0.0)),
        "brake": float(meta.get("brake", 0.0)),
        "desired_speed": float(meta.get("desired_speed", 0.0)),
        "spat_mean_x": sum(spat_xs) / len(spat_xs),
        "spat_first_x": spat_xs[0],
        "spat_last_x": spat_xs[-1],
        "spat_max_abs_x": max(abs(x) for x in spat_xs),
        "aim_x": float(meta["aim"][0]),
        "angle_final": float(meta.get("angle_final", meta.get("angle", 0.0))),
        "temp_path_len": path_len(plan_temp),
        "temp_last_x": temp_xs[-1],
        "temp_last_y": temp_ys[-1],
        "temp_mean_x": sum(temp_xs) / len(temp_xs),
        "temp_mean_y": sum(temp_ys) / len(temp_ys),
    }
  except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
    return None


def feature_path_for(collection_root: Path, split: str, run_name: str, frame: int, model_index: int) -> Path:
  feature_dir = collection_root / split / "features" / run_name
  model_dir = feature_dir / f"model_{model_index:02d}"
  if model_dir.exists():
    feature_dir = model_dir
  return feature_dir / f"{frame:06d}.pt"


def collect_candidates(collection_root: Path, split: str, model_index: int) -> list[dict[str, Any]]:
  candidates: list[dict[str, Any]] = []
  for meta_path in sorted((collection_root / split / "images").glob("*/metas/*.json")):
    run_name = meta_path.parent.parent.name
    frame = int(meta_path.stem)
    feature_path = feature_path_for(collection_root, split, run_name, frame, model_index)
    if not feature_path.exists():
      continue
    try:
      meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      continue
    metrics = plan_metrics(meta)
    if metrics is None:
      continue
    candidates.append({
        "split": split,
        "run_name": run_name,
        "frame": frame,
        "feature_path": feature_path,
        "hipad_meta_path": meta_path,
        **metrics,
    })
  return candidates


def is_positive(row: dict[str, Any], args: argparse.Namespace) -> bool:
  return (
      row["brake"] <= args.positive_max_brake and
      row["speed"] >= args.positive_min_speed and
      row["speed"] <= args.positive_max_speed and
      row["desired_speed"] >= args.positive_min_desired_speed and
      row["temp_path_len"] >= args.positive_min_temp_path_len and
      row["temp_path_len"] <= args.positive_max_temp_path_len and
      row["temp_last_y"] >= args.positive_min_temp_last_y and
      -row["steer"] >= args.positive_min_left_steer and
      -row["spat_first_x"] >= args.positive_min_left_spat_first and
      -row["aim_x"] >= args.positive_min_left_aim and
      -row["angle_final"] >= args.positive_min_left_angle
  )


def is_negative(row: dict[str, Any], args: argparse.Namespace) -> bool:
  return (
      row["brake"] <= args.negative_max_brake and
      row["speed"] >= args.negative_min_speed and
      row["speed"] <= args.negative_max_speed and
      row["desired_speed"] <= args.negative_max_desired_speed and
      row["temp_path_len"] >= args.negative_min_temp_path_len and
      row["temp_path_len"] <= args.negative_max_temp_path_len and
      abs(row["spat_mean_x"]) <= args.negative_max_spat_mean_abs and
      abs(row["spat_last_x"]) <= args.negative_max_spat_last_abs and
      abs(row["aim_x"]) <= args.negative_max_aim_abs and
      abs(row["angle_final"]) <= args.negative_max_angle_abs
  )


def match_score(positive: dict[str, Any], negative: dict[str, Any], args: argparse.Namespace) -> float:
  return (
      abs(positive["desired_speed"] - negative["desired_speed"]) / args.match_desired_scale +
      abs(positive["speed"] - negative["speed"]) / args.match_speed_scale +
      abs(positive["temp_path_len"] - negative["temp_path_len"]) / args.match_temp_scale
  )


def cap_per_run(rows: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
  if max_frames <= 0:
    return rows
  kept: list[dict[str, Any]] = []
  counts: dict[str, int] = {}
  for row in sorted(rows, key=lambda item: (item["run_name"], item["frame"])):
    count = counts.get(row["run_name"], 0)
    if count >= max_frames:
      continue
    kept.append(row)
    counts[row["run_name"]] = count + 1
  return kept


def select_matched_negatives(
    positives: list[dict[str, Any]],
    negative_pool: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
  selected: list[dict[str, Any]] = []
  selected_keys: set[tuple[str, int]] = set()
  for positive in positives:
    ranked = sorted(negative_pool, key=lambda negative: match_score(positive, negative, args))
    used_for_positive = 0
    for negative in ranked:
      key = (negative["run_name"], int(negative["frame"]))
      if key in selected_keys:
        continue
      negative = dict(negative)
      negative["match_to_run_name"] = positive["run_name"]
      negative["match_to_frame"] = positive["frame"]
      negative["match_score"] = match_score(positive, negative, args)
      selected.append(negative)
      selected_keys.add(key)
      used_for_positive += 1
      if used_for_positive >= args.negative_per_positive:
        break
  return selected


def add_feature(accumulator: dict[str, Any], label: str, path: Path) -> None:
  feature = torch.load(path, map_location="cpu").detach().float()
  key = f"{label}_sum"
  if accumulator[key] is None:
    accumulator[key] = torch.zeros_like(feature)
  accumulator[key] += feature
  accumulator[f"{label}_count"] += 1


def manifest_record(label: str, row: dict[str, Any]) -> dict[str, Any]:
  keys = [
      "speed", "steer", "throttle", "brake", "desired_speed",
      "spat_mean_x", "spat_first_x", "spat_last_x", "spat_max_abs_x",
      "aim_x", "angle_final", "temp_path_len", "temp_last_x",
      "temp_last_y", "temp_mean_x", "temp_mean_y",
      "match_to_run_name", "match_to_frame", "match_score",
  ]
  record = {
      "label": label,
      "action": "left_change_lane",
      "run_name": row["run_name"],
      "frame": int(row["frame"]),
      "feature_path": str(row["feature_path"]),
      "hipad_meta_path": str(row["hipad_meta_path"]),
  }
  for key in keys:
    if key in row:
      record[key] = int(row[key]) if key == "match_to_frame" else row[key]
  return record


def value_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
  values = sorted(float(row[key]) for row in rows if key in row and row[key] is not None)
  if not values:
    return None

  def quantile(position: float) -> float:
    index = int((len(values) - 1) * position)
    return values[index]

  return {
      "min": values[0],
      "p25": quantile(0.25),
      "median": quantile(0.5),
      "p75": quantile(0.75),
      "max": values[-1],
  }


def main() -> int:
  args = parse_args()
  collection_root = Path(args.collection_root)
  output_dir = (
      Path(args.output_dir)
      if args.output_dir
      else collection_root / "post_process" / "left_change_lane_codex"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  left_candidates = collect_candidates(collection_root, "LaneChangeLeft", args.model_index)
  normal_candidates = collect_candidates(collection_root, "Normal", args.model_index)
  positives = cap_per_run([row for row in left_candidates if is_positive(row, args)], args.max_frames_per_run)
  negative_pool = [row for row in normal_candidates if is_negative(row, args)]
  negatives = select_matched_negatives(positives, negative_pool, args)

  if not positives or not negatives:
    raise RuntimeError(
        f"Need both classes, got positive={len(positives)} negative={len(negatives)} "
        f"from pools left={len(left_candidates)} normal={len(normal_candidates)}"
    )

  accumulator: dict[str, Any] = {
      "positive_sum": None,
      "negative_sum": None,
      "positive_count": 0,
      "negative_count": 0,
  }
  manifest_path = output_dir / "selected_frames.jsonl"
  with manifest_path.open("w", encoding="utf-8") as manifest:
    for label, rows in (("positive", positives), ("negative", negatives)):
      for row in rows:
        add_feature(accumulator, label, row["feature_path"])
        manifest.write(json.dumps(manifest_record(label, row)) + "\n")

  positive_mean = accumulator["positive_sum"] / accumulator["positive_count"]
  negative_mean = accumulator["negative_sum"] / accumulator["negative_count"]
  steering_vector = positive_mean - negative_mean
  torch.save(positive_mean, output_dir / "positive_mean.pt")
  torch.save(negative_mean, output_dir / "negative_mean.pt")
  torch.save(steering_vector, output_dir / "steering_vector.pt")

  summary_keys = [
      "speed", "steer", "desired_speed", "brake", "spat_mean_x",
      "spat_first_x", "spat_last_x", "aim_x", "angle_final",
      "temp_path_len", "temp_last_y",
  ]
  summary = {
      "script": str(Path(__file__)),
      "action": "left_change_lane",
      "positive_label": "moving_left_lane_change",
      "negative_label": "matched_lateral_neutral_normal",
      "positive_count": accumulator["positive_count"],
      "negative_count": accumulator["negative_count"],
      "left_candidate_count": len(left_candidates),
      "normal_candidate_count": len(normal_candidates),
      "negative_pool_count": len(negative_pool),
      "vector_formula": "positive_mean - negative_mean",
      "output_files": {
          "positive_mean": str(output_dir / "positive_mean.pt"),
          "negative_mean": str(output_dir / "negative_mean.pt"),
          "steering_vector": str(output_dir / "steering_vector.pt"),
          "selected_frames": str(manifest_path),
      },
      "positive_runs": sorted({row["run_name"] for row in positives}),
      "negative_runs": sorted({row["run_name"] for row in negatives}),
      "metrics": {
          "positive": {key: value_summary(positives, key) for key in summary_keys},
          "negative": {key: value_summary(negatives, key) for key in summary_keys},
      },
      "args": vars(args),
  }
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

#!/usr/bin/env python3
"""Compare feature-only concept scores around picked TransFuser frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


ACTION_CONFIG = {
    "brake": {"split": "Brake", "post_process": "brake"},
    "left": {"split": "left", "post_process": "left"},
    "right": {"split": "right", "post_process": "right"},
}

SCORE_NAMES = (
    "raw_cosine",
    "centered_cosine",
    "scalar_projection",
    "prototype_l2_margin",
    "prototype_cosine_margin",
    "whitened_projection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("steering/transfuser"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--lateral-steer-threshold", type=float, default=0.10)
    parser.add_argument("--moving-speed-threshold", type=float, default=0.50)
    parser.add_argument(
        "--variance-regularization",
        type=float,
        default=0.10,
        help="Fraction of median negative variance added to diagonal whitening.",
    )
    return parser.parse_args()


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(flatten(left), flatten(right), dim=0, eps=1e-8))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def negative_feature_paths(post_root: Path) -> list[Path]:
    records = load_jsonl(post_root / "selected_frames.jsonl")
    return [Path(record["feature_path"]) for record in records if record["label"] == "negative"]


def negative_moments(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    total = None
    squared_total = None
    for path in paths:
        feature = torch.load(path, map_location="cpu").detach().float()
        total = feature.clone() if total is None else total + feature
        squared = feature.square()
        squared_total = squared.clone() if squared_total is None else squared_total + squared
    mean = total / len(paths)
    variance = (squared_total / len(paths) - mean.square()).clamp_min(0.0)
    return mean, variance


def make_scorer(post_root: Path, variance_regularization: float) -> tuple[dict, list[Path]]:
    positive_mean = torch.load(post_root / "positive_mean.pt", map_location="cpu").detach().float()
    saved_negative_mean = torch.load(post_root / "negative_mean.pt", map_location="cpu").detach().float()
    vector = torch.load(post_root / "steering_vector.pt", map_location="cpu").detach().float()
    negative_paths = negative_feature_paths(post_root)
    computed_negative_mean, negative_variance = negative_moments(negative_paths)
    if not torch.allclose(saved_negative_mean, computed_negative_mean, atol=2e-5, rtol=2e-5):
        raise RuntimeError(f"Saved and recomputed negative means disagree under {post_root}")

    median_variance = float(torch.median(flatten(negative_variance)))
    regularizer = max(median_variance * variance_regularization, 1e-8)
    whitened_direction = vector / (negative_variance + regularizer)
    vector_norm_sq = float(torch.sum(vector.square()))
    whitened_span = float(torch.sum(vector * whitened_direction))

    state = {
        "positive_mean": positive_mean,
        "negative_mean": saved_negative_mean,
        "vector": vector,
        "whitened_direction": whitened_direction,
        "vector_norm_sq": vector_norm_sq,
        "whitened_span": whitened_span,
        "median_negative_variance": median_variance,
        "variance_regularizer": regularizer,
    }
    return state, negative_paths


def score_feature(feature: torch.Tensor, state: dict) -> dict[str, float]:
    feature = feature.detach().float()
    positive = state["positive_mean"]
    negative = state["negative_mean"]
    vector = state["vector"]
    centered = feature - negative
    negative_distance = float(torch.sum(centered.square()))
    positive_distance = float(torch.sum((feature - positive).square()))
    return {
        "raw_cosine": cosine(feature, vector),
        "centered_cosine": cosine(centered, vector),
        "scalar_projection": float(torch.sum(centered * vector)) / state["vector_norm_sq"],
        "prototype_l2_margin": negative_distance - positive_distance,
        "prototype_cosine_margin": cosine(feature, positive) - cosine(feature, negative),
        "whitened_projection": (
            float(torch.sum(centered * state["whitened_direction"])) / state["whitened_span"]
        ),
    }


def add_negative_calibration(state: dict, negative_paths: list[Path]) -> None:
    values = defaultdict(list)
    for path in negative_paths:
        scores = score_feature(torch.load(path, map_location="cpu"), state)
        for name, value in scores.items():
            values[name].append(value)
    state["negative_calibration"] = {
        name: {
            "mean": float(np.mean(method_values)),
            "std": float(np.std(method_values, ddof=1)),
            "p05": float(np.quantile(method_values, 0.05)),
            "p50": float(np.quantile(method_values, 0.50)),
            "p95": float(np.quantile(method_values, 0.95)),
        }
        for name, method_values in values.items()
    }


def behavior_phase(action: str, row: dict, steer_threshold: float, speed_threshold: float) -> str:
    speed = float(row.get("speed", 0.0))
    steer = float(row.get("steer", 0.0))
    brake = float(row.get("brake", 0.0))
    if action == "brake":
        if brake >= 0.5 and speed >= speed_threshold:
            return "moving_brake"
        if brake >= 0.5:
            return "stopped_brake_hold"
        return "neutral"
    if brake >= 0.5:
        return "braking"
    signed_steer = -steer if action == "left" else steer
    if signed_steer >= steer_threshold:
        return "same_action"
    if signed_steer <= -steer_threshold:
        return "opposite_action"
    return "neutral"


def action_strength(action: str, row: dict) -> float:
    if action == "brake":
        speed = float(row.get("speed", 0.0))
        target_speed = float(row.get("pred_target_speed", speed))
        return max(speed - target_speed, 0.0)
    steer = float(row.get("steer", 0.0))
    return -steer if action == "left" else steer


def analyze_action(root: Path, action: str, config: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
    split_root = root / config["split"]
    post_root = root / "post_process" / config["post_process"]
    state, negative_paths = make_scorer(post_root, args.variance_regularization)
    add_negative_calibration(state, negative_paths)
    picked = json.loads((split_root / "picked_frames.json").read_text(encoding="utf-8"))

    records = []
    for run_name, picked_values in picked.items():
        picked_frames = sorted(int(value) for value in picked_values)
        log_rows = {
            int(row["frame"]): row
            for row in load_jsonl(split_root / "logs" / run_name / "activation_actions.jsonl")
        }
        requested_frames = set()
        for picked_frame in picked_frames:
            requested_frames.update(range(picked_frame - args.window, picked_frame + args.window + 1))
        for frame in sorted(requested_frames):
            feature_path = split_root / "features" / run_name / f"{frame:06d}.pt"
            row = log_rows.get(frame)
            if row is None or not feature_path.is_file():
                continue
            scores = score_feature(torch.load(feature_path, map_location="cpu"), state)
            record = {
                "action": action,
                "run_name": run_name,
                "frame": frame,
                "is_picked": frame in picked_frames,
                "distance_to_nearest_picked": min(abs(frame - value) for value in picked_frames),
                "phase": behavior_phase(
                    action, row, args.lateral_steer_threshold, args.moving_speed_threshold),
                "action_strength": action_strength(action, row),
                "speed": float(row.get("speed", 0.0)),
                "steer": float(row.get("steer", 0.0)),
                "brake": float(row.get("brake", 0.0)),
                "pred_target_speed": float(row.get("pred_target_speed", 0.0)),
                "feature_path": str(feature_path),
            }
            for name, value in scores.items():
                record[name] = value
                calibration = state["negative_calibration"][name]
                record[f"{name}_negative_z"] = (
                    (value - calibration["mean"]) / calibration["std"]
                    if calibration["std"] > 1e-12 else 0.0
                )
            records.append(record)

    metadata = {
        "negative_count": len(negative_paths),
        "median_negative_variance": state["median_negative_variance"],
        "variance_regularizer": state["variance_regularizer"],
        "negative_calibration": state["negative_calibration"],
    }
    return records, metadata


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def safe_correlation(x: list[float], y: list[float], function) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(function(x, y).statistic)


def separation_auc(positive: list[float], negative: list[float]) -> float | None:
    """Probability that a random positive score exceeds a random negative score."""
    if not positive or not negative:
        return None
    positive_values = np.asarray(positive)[:, None]
    negative_values = np.asarray(negative)[None, :]
    return float(np.mean(positive_values > negative_values) + 0.5 * np.mean(
        positive_values == negative_values))


def summarize(records: list[dict], metadata: dict, args: argparse.Namespace) -> dict:
    result = {
        "window": args.window,
        "lateral_steer_threshold": args.lateral_steer_threshold,
        "moving_speed_threshold": args.moving_speed_threshold,
        "actions": {},
    }
    for action in ACTION_CONFIG:
        rows = [row for row in records if row["action"] == action]
        action_summary = {"metadata": metadata[action], "picked_frames": []}
        for row in rows:
            if row["is_picked"]:
                action_summary["picked_frames"].append({
                    "frame": row["frame"],
                    "phase": row["phase"],
                    "speed": row["speed"],
                    "steer": row["steer"],
                    "brake": row["brake"],
                    **{name: row[name] for name in SCORE_NAMES},
                })
        phases = sorted({row["phase"] for row in rows})
        action_summary["phase_distributions"] = {
            phase: {
                name: distribution([row[name] for row in rows if row["phase"] == phase])
                for name in SCORE_NAMES
            }
            for phase in phases
        }
        strengths = [row["action_strength"] for row in rows]
        action_summary["correlation_with_action_strength"] = {
            name: {
                "pearson": safe_correlation(
                    [row[name] for row in rows], strengths, pearsonr),
                "spearman": safe_correlation(
                    [row[name] for row in rows], strengths, spearmanr),
            }
            for name in SCORE_NAMES
        }
        held_out_rows = [row for row in rows if not row["is_picked"]]
        phase_pairs = (
            [("moving_brake", "neutral"), ("stopped_brake_hold", "neutral")]
            if action == "brake"
            else [("same_action", "opposite_action"), ("same_action", "neutral")]
        )
        action_summary["held_out_phase_auc"] = {
            f"{positive_phase}_vs_{negative_phase}": {
                name: separation_auc(
                    [row[name] for row in held_out_rows if row["phase"] == positive_phase],
                    [row[name] for row in held_out_rows if row["phase"] == negative_phase],
                )
                for name in SCORE_NAMES
            }
            for positive_phase, negative_phase in phase_pairs
        }
        result["actions"][action] = action_summary
    return result


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_action(path: Path, action: str, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: row["frame"])
    frames = [row["frame"] for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

    axes[0].plot(frames, [row["raw_cosine"] for row in rows], label="raw cosine")
    axes[0].plot(frames, [row["centered_cosine"] for row in rows], label="centered cosine")
    axes[0].plot(
        frames, [row["prototype_cosine_margin"] for row in rows], label="cosine prototype margin")
    axes[0].axhline(0.0, color="gray", linewidth=0.7)
    axes[0].set_ylabel("cosine score")
    axes[0].legend(fontsize=8)

    axes[1].plot(frames, [row["scalar_projection"] for row in rows], label="scalar projection")
    axes[1].plot(frames, [row["whitened_projection"] for row in rows], label="whitened projection")
    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].set_ylabel("prototype coordinate")
    axes[1].legend(fontsize=8)

    axes[2].plot(frames, [row["steer"] for row in rows], label="steer")
    axes[2].plot(frames, [row["brake"] for row in rows], label="brake")
    speed_scale = max(max(abs(row["speed"]) for row in rows), 1.0)
    axes[2].plot(frames, [row["speed"] / speed_scale for row in rows], label="speed / max speed")
    axes[2].axhline(0.0, color="gray", linewidth=0.7)
    axes[2].set_ylabel("control / normalized speed")
    axes[2].set_xlabel("frame")
    axes[2].legend(fontsize=8)

    for axis in axes:
        for row in rows:
            if row["is_picked"]:
                axis.axvline(row["frame"], color="black", alpha=0.35, linewidth=0.8)
        axis.grid(alpha=0.25)
    figure.suptitle(action)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.window < 0:
        raise ValueError("--window must be non-negative")
    output_dir = args.output_dir or args.root / "concept_score_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    metadata = {}
    for action, config in ACTION_CONFIG.items():
        action_records, action_metadata = analyze_action(args.root, action, config, args)
        records.extend(action_records)
        metadata[action] = action_metadata
        plot_action(output_dir / f"{action}_concept_scores.png", action, action_records)

    summary = summarize(records, metadata, args)
    write_csv(output_dir / "frame_concept_scores.csv", records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

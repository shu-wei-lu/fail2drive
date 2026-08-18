#!/usr/bin/env python3
"""Analyze TransFuser features near manually picked steering frames.

For each action, compare nearby frame features with the saved steering vector
(`positive_mean - negative_mean`) using both raw and negative-mean-centered
cosine similarity. Results are written as CSV, JSON, and a PNG plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


ACTION_CONFIG = {
    "brake": {"split": "Brake", "post_process": "brake"},
    "left": {"split": "left", "post_process": "left"},
    "right": {"split": "right", "post_process": "right"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("steering/transfuser"),
        help="TransFuser steering collection root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/cosine_analysis).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Number of frames before/after each picked frame to include.",
    )
    parser.add_argument("--gate-low", type=float, default=0.30)
    parser.add_argument("--gate-high", type=float, default=0.60)
    return parser.parse_args()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(1, -1)
    right = right.detach().float().reshape(1, -1)
    if left.shape != right.shape:
        raise ValueError(f"Cosine shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float(F.cosine_similarity(left, right, dim=1, eps=1e-8).item())


def load_log_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows[int(row["frame"])] = row
    return rows


def implied_gate(similarity: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (high - similarity) / (high - low)))


def analyze_action(
    root: Path,
    action: str,
    config: dict[str, str],
    window: int,
    gate_low: float,
    gate_high: float,
) -> list[dict]:
    split_root = root / config["split"]
    post_root = root / "post_process" / config["post_process"]
    picked = json.loads((split_root / "picked_frames.json").read_text(encoding="utf-8"))
    vector = torch.load(post_root / "steering_vector.pt", map_location="cpu")
    positive_mean = torch.load(post_root / "positive_mean.pt", map_location="cpu")
    negative_mean = torch.load(post_root / "negative_mean.pt", map_location="cpu")

    records = []
    for run_name, picked_frames_raw in picked.items():
        picked_frames = sorted(int(frame) for frame in picked_frames_raw)
        feature_dir = split_root / "features" / run_name
        log_path = split_root / "logs" / run_name / "activation_actions.jsonl"
        log_rows = load_log_rows(log_path)
        requested_frames = set()
        for picked_frame in picked_frames:
            requested_frames.update(range(picked_frame - window, picked_frame + window + 1))

        for frame in sorted(requested_frames):
            feature_path = feature_dir / f"{frame:06d}.pt"
            if not feature_path.is_file():
                continue
            feature = torch.load(feature_path, map_location="cpu")
            raw_similarity = cosine(feature, vector)
            centered_similarity = cosine(feature - negative_mean, vector)
            row = log_rows.get(frame, {})
            records.append(
                {
                    "action": action,
                    "run_name": run_name,
                    "frame": frame,
                    "is_picked": frame in picked_frames,
                    "distance_to_nearest_picked": min(abs(frame - value) for value in picked_frames),
                    "cosine_feature_vs_vector": raw_similarity,
                    "cosine_centered_vs_vector": centered_similarity,
                    "cosine_feature_vs_positive_mean": cosine(feature, positive_mean),
                    "cosine_feature_vs_negative_mean": cosine(feature, negative_mean),
                    "implied_gate_raw": implied_gate(raw_similarity, gate_low, gate_high),
                    "implied_gate_centered": implied_gate(centered_similarity, gate_low, gate_high),
                    "speed": float(row.get("speed", "nan")),
                    "pred_target_speed": float(row.get("pred_target_speed", "nan")),
                    "steer": float(row.get("steer", "nan")),
                    "throttle": float(row.get("throttle", "nan")),
                    "brake": float(row.get("brake", "nan")),
                    "feature_path": str(feature_path),
                }
            )
    return records


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def summarize(records: list[dict], window: int, gate_low: float, gate_high: float) -> dict:
    summary = {
        "window": window,
        "gate_low": gate_low,
        "gate_high": gate_high,
        "actions": {},
    }
    for action in ACTION_CONFIG:
        action_rows = [row for row in records if row["action"] == action]
        picked_rows = [row for row in action_rows if row["is_picked"]]
        nearby_rows = [row for row in action_rows if not row["is_picked"]]
        action_summary = {}
        for label, rows in (("picked", picked_rows), ("nearby_excluding_picked", nearby_rows)):
            action_summary[label] = {
                "raw_cosine": stats([row["cosine_feature_vs_vector"] for row in rows]),
                "centered_cosine": stats([row["cosine_centered_vs_vector"] for row in rows]),
                "implied_gate_raw": stats([row["implied_gate_raw"] for row in rows]),
            }
        action_summary["picked_frames"] = [row["frame"] for row in picked_rows]
        action_summary["picked_frame_values"] = [
            {
                "frame": row["frame"],
                "raw_cosine": row["cosine_feature_vs_vector"],
                "centered_cosine": row["cosine_centered_vs_vector"],
                "implied_gate_raw": row["implied_gate_raw"],
                "steer": row["steer"],
                "brake": row["brake"],
            }
            for row in picked_rows
        ]
        summary["actions"][action] = action_summary
    return summary


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot(path: Path, records: list[dict], gate_low: float, gate_high: float) -> None:
    figure, axes = plt.subplots(len(ACTION_CONFIG), 1, figsize=(13, 10), constrained_layout=True)
    for axis, action in zip(axes, ACTION_CONFIG):
        rows = [row for row in records if row["action"] == action]
        frames = [row["frame"] for row in rows]
        raw = [row["cosine_feature_vs_vector"] for row in rows]
        centered = [row["cosine_centered_vs_vector"] for row in rows]
        picked = [row for row in rows if row["is_picked"]]
        axis.plot(frames, raw, label="cos(feature, pos-neg)", linewidth=1.8)
        axis.plot(frames, centered, label="cos(feature-neg_mean, pos-neg)", linewidth=1.5)
        axis.scatter(
            [row["frame"] for row in picked],
            [row["cosine_feature_vs_vector"] for row in picked],
            color="black",
            marker="x",
            s=55,
            zorder=5,
            label="picked frame",
        )
        axis.axhline(gate_low, color="tab:orange", linestyle="--", alpha=0.7, label=f"gate low={gate_low:g}")
        axis.axhline(gate_high, color="tab:red", linestyle="--", alpha=0.7, label=f"gate high={gate_high:g}")
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(action)
        axis.set_xlabel("frame")
        axis.set_ylabel("cosine similarity")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.window < 0:
        raise ValueError("--window must be non-negative")
    if args.gate_high <= args.gate_low:
        raise ValueError("--gate-high must be greater than --gate-low")
    output_dir = args.output_dir or args.root / "cosine_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for action, config in ACTION_CONFIG.items():
        records.extend(
            analyze_action(
                args.root,
                action,
                config,
                args.window,
                args.gate_low,
                args.gate_high,
            )
        )
    if not records:
        raise RuntimeError("No nearby TransFuser feature files were found")

    summary = summarize(records, args.window, args.gate_low, args.gate_high)
    write_csv(output_dir / "frame_cosine.csv", records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(output_dir / "cosine_near_picked_frames.png", records, args.gate_low, args.gate_high)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

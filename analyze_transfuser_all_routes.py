#!/usr/bin/env python3
"""Evaluate TransFuser steering concepts on every saved feature/log route."""

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

from analyze_transfuser_concept_scores import (
    ACTION_CONFIG,
    SCORE_NAMES,
    add_negative_calibration,
    behavior_phase,
    load_jsonl,
    make_scorer,
    score_feature,
    separation_auc,
)


SPLITS = ("Brake", "Normal", "left", "right")
PHASE_PAIRS = {
    "brake": (("moving_brake", "neutral"), ("stopped_brake_hold", "neutral")),
    "left": (("same_action", "neutral"), ("same_action", "opposite_action")),
    "right": (("same_action", "neutral"), ("same_action", "opposite_action")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("steering/transfuser"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lateral-steer-threshold", type=float, default=0.10)
    parser.add_argument("--moving-speed-threshold", type=float, default=0.50)
    parser.add_argument("--variance-regularization", type=float, default=0.10)
    return parser.parse_args()


def load_states(root: Path, variance_regularization: float) -> dict[str, dict]:
    states = {}
    for concept, config in ACTION_CONFIG.items():
        state, negative_paths = make_scorer(
            root / "post_process" / config["post_process"], variance_regularization)
        add_negative_calibration(state, negative_paths)
        states[concept] = state
    return states


def source_routes(root: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, set[int]]]]:
    routes = {}
    frames = {}
    for concept, config in ACTION_CONFIG.items():
        picked = json.loads(
            (root / config["split"] / "picked_frames.json").read_text(encoding="utf-8"))
        routes[concept] = set(picked)
        frames[concept] = {
            run_name: {int(frame) for frame in run_frames}
            for run_name, run_frames in picked.items()
        }
    return routes, frames


def collect_records(args: argparse.Namespace, states: dict[str, dict]) -> list[dict]:
    source, picked = source_routes(args.root)
    records = []
    for split in SPLITS:
        feature_root = args.root / split / "features"
        log_root = args.root / split / "logs"
        if not feature_root.is_dir() or not log_root.is_dir():
            continue
        for run_dir in sorted(path for path in feature_root.iterdir() if path.is_dir()):
            log_path = log_root / run_dir.name / "activation_actions.jsonl"
            if not log_path.is_file():
                continue
            logs = {int(row["frame"]): row for row in load_jsonl(log_path)}
            for feature_path in sorted(run_dir.glob("*.pt")):
                frame = int(feature_path.stem)
                log = logs.get(frame)
                if log is None:
                    continue
                feature = torch.load(feature_path, map_location="cpu")
                for concept, state in states.items():
                    scores = score_feature(feature, state)
                    record = {
                        "concept": concept,
                        "split": split,
                        "run_name": run_dir.name,
                        "frame": frame,
                        "is_source_route": run_dir.name in source[concept],
                        "is_picked": frame in picked[concept].get(run_dir.name, set()),
                        "phase": behavior_phase(
                            concept,
                            log,
                            args.lateral_steer_threshold,
                            args.moving_speed_threshold,
                        ),
                        "speed": float(log.get("speed", 0.0)),
                        "steer": float(log.get("steer", 0.0)),
                        "brake": float(log.get("brake", 0.0)),
                        "pred_target_speed": float(log.get("pred_target_speed", 0.0)),
                    }
                    for name, value in scores.items():
                        record[name] = value
                        calibration = state["negative_calibration"][name]
                        record[f"{name}_negative_z"] = (
                            (value - calibration["mean"]) / calibration["std"]
                            if calibration["std"] > 1e-12
                            else 0.0
                        )
                    records.append(record)
    return records


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def threshold_summary(positive: list[float], negative: list[float]) -> dict | None:
    if not positive or not negative:
        return None
    positive_values = np.asarray(positive)
    negative_values = np.asarray(negative)
    combined = np.concatenate((positive_values, negative_values))
    thresholds = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, 1001)))
    best = None
    for threshold in thresholds:
        true_positive_rate = float(np.mean(positive_values > threshold))
        true_negative_rate = float(np.mean(negative_values <= threshold))
        balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
        candidate = (balanced_accuracy, true_positive_rate, true_negative_rate, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return {
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "true_positive_rate": best[1],
        "true_negative_rate": best[2],
        "false_positive_rate": 1.0 - best[2],
    }


def summarize(records: list[dict]) -> dict:
    summary = {
        "route_count": len({(row["split"], row["run_name"]) for row in records}),
        "feature_frame_count": len({(row["split"], row["run_name"], row["frame"]) for row in records}),
        "concepts": {},
    }
    for concept in ACTION_CONFIG:
        concept_rows = [row for row in records if row["concept"] == concept]
        held_out = [row for row in concept_rows if not row["is_source_route"]]
        phases = sorted({row["phase"] for row in held_out})
        concept_summary = {
            "source_routes_excluded": sorted({
                row["run_name"] for row in concept_rows if row["is_source_route"]}),
            "held_out_route_count": len({(row["split"], row["run_name"]) for row in held_out}),
            "held_out_frame_count": len(held_out),
            "phase_distributions": {
                phase: {
                    name: distribution([row[name] for row in held_out if row["phase"] == phase])
                    for name in SCORE_NAMES
                }
                for phase in phases
            },
            "split_phase_projection": {},
            "phase_auc": {},
        }
        for split in SPLITS:
            split_rows = [row for row in held_out if row["split"] == split]
            if not split_rows:
                continue
            concept_summary["split_phase_projection"][split] = {
                phase: distribution([
                    row["scalar_projection"] for row in split_rows if row["phase"] == phase])
                for phase in sorted({row["phase"] for row in split_rows})
            }

        for positive_phase, negative_phase in PHASE_PAIRS[concept]:
            pair_name = f"{positive_phase}_vs_{negative_phase}"
            positive_rows = [row for row in held_out if row["phase"] == positive_phase]
            negative_rows = [row for row in held_out if row["phase"] == negative_phase]
            pooled = {
                name: separation_auc(
                    [row[name] for row in positive_rows],
                    [row[name] for row in negative_rows],
                )
                for name in SCORE_NAMES
            }
            per_route = []
            route_groups = defaultdict(list)
            for row in held_out:
                route_groups[(row["split"], row["run_name"])].append(row)
            for (split, run_name), route_rows in route_groups.items():
                positive = [
                    row["scalar_projection"] for row in route_rows
                    if row["phase"] == positive_phase]
                negative = [
                    row["scalar_projection"] for row in route_rows
                    if row["phase"] == negative_phase]
                auc = separation_auc(positive, negative)
                if auc is not None:
                    per_route.append({
                        "split": split,
                        "run_name": run_name,
                        "positive_count": len(positive),
                        "negative_count": len(negative),
                        "scalar_projection_auc": auc,
                    })
            route_aucs = [row["scalar_projection_auc"] for row in per_route]
            concept_summary["phase_auc"][pair_name] = {
                "positive_count": len(positive_rows),
                "negative_count": len(negative_rows),
                "pooled": pooled,
                "macro_route_scalar_projection_auc": (
                    float(np.mean(route_aucs)) if route_aucs else None),
                "median_route_scalar_projection_auc": (
                    float(np.median(route_aucs)) if route_aucs else None),
                "route_count_with_both_phases": len(per_route),
                "per_route": per_route,
                "scalar_projection_threshold": threshold_summary(
                    [row["scalar_projection"] for row in positive_rows],
                    [row["scalar_projection"] for row in negative_rows],
                ),
            }
        summary["concepts"][concept] = concept_summary
    return summary


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_distributions(path: Path, records: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, concept in zip(axes, ACTION_CONFIG):
        rows = [
            row for row in records
            if row["concept"] == concept and not row["is_source_route"]]
        desired_phases = (
            ["moving_brake", "stopped_brake_hold", "neutral"]
            if concept == "brake"
            else ["same_action", "opposite_action", "neutral"]
        )
        values = [[
            row["scalar_projection"] for row in rows if row["phase"] == phase]
            for phase in desired_phases]
        axis.boxplot(values, labels=desired_phases, showfliers=False)
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
        axis.set_title(concept)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("negative-centered scalar projection")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.root / "all_routes_concept_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    states = load_states(args.root, args.variance_regularization)
    records = collect_records(args, states)
    if not records:
        raise RuntimeError("No feature/log pairs found")
    summary = summarize(records)
    write_csv(output_dir / "all_route_frame_scores.csv", records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_distributions(output_dir / "held_out_projection_distributions.png", records)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

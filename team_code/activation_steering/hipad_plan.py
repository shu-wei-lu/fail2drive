from __future__ import annotations

import json
from pathlib import Path

from activation_steering.transfuser_target_speed import TransFuserTargetSpeedAdapter


class HiPADPlanAdapter(TransFuserTargetSpeedAdapter):
  """Adapter for HiP-AD align-query activation steering."""

  name = "hipad_plan"

  def add_post_process_args(self, parser) -> None:
    super().add_post_process_args(parser)
    parser.add_argument(
        "--hipad-disable-plan-meta-labels",
        action="store_true",
        help="Use action-log steer thresholds instead of HiP-AD plan_spat metadata labels.",
    )
    parser.add_argument("--hipad-plan-spat-mean-threshold", type=float, default=1.0)
    parser.add_argument("--hipad-plan-spat-last-threshold", type=float, default=0.8)
    parser.add_argument("--hipad-plan-spat-first-threshold", type=float, default=0.0)
    parser.add_argument("--hipad-plan-aim-threshold", type=float, default=0.5)
    parser.add_argument("--hipad-plan-angle-threshold", type=float, default=0.12)
    parser.add_argument("--hipad-plan-min-desired-speed", type=float, default=0.0)
    parser.add_argument("--hipad-normal-plan-spat-mean-max", type=float, default=0.25)
    parser.add_argument("--hipad-normal-plan-spat-last-max", type=float, default=0.4)
    parser.add_argument("--hipad-normal-aim-max", type=float, default=0.2)
    parser.add_argument("--hipad-normal-angle-max", type=float, default=0.05)
    parser.add_argument(
        "--hipad-brake-require-neutral-plan",
        action="store_true",
        help=(
            "For brake positives and normal negatives, require HiP-AD plan metadata "
            "to be lateral-neutral using the hipad-normal-* thresholds."
        ),
    )

  def augment_rows(self, rows: list[dict], log_source: Path, collection_root: Path) -> list[dict]:
    meta_dir = self._meta_dir_for_log(log_source, collection_root)
    if meta_dir is None:
      return rows

    for row in rows:
      meta_path = meta_dir / f"{int(row['frame']):04d}.json"
      if not meta_path.exists():
        continue
      try:
        with meta_path.open("r", encoding="utf-8") as f:
          row["_hipad_meta"] = json.load(f)
        row["_hipad_meta_path"] = str(meta_path)
      except (OSError, json.JSONDecodeError):
        continue
    return rows

  def classify_post_process_label(self, row: dict, action: str, run_name: str, args) -> str | None:
    if args.hipad_disable_plan_meta_labels:
      return None
    if action not in ("left_change_lane", "right_change_lane"):
      return None

    meta = row.get("_hipad_meta")
    if meta is None:
      return "skip"

    metrics = self._plan_metrics(meta)
    if metrics is None:
      return "skip"

    direction = -1.0 if action == "left_change_lane" else 1.0
    signed_mean = direction * metrics["spat_mean_x"]
    signed_first = direction * metrics["spat_first_x"]
    signed_last = direction * metrics["spat_last_x"]
    signed_aim = direction * metrics["aim_x"]
    signed_angle = direction * metrics["angle_final"]

    if (
        signed_mean >= args.hipad_plan_spat_mean_threshold and
        signed_first >= args.hipad_plan_spat_first_threshold and
        signed_last >= args.hipad_plan_spat_last_threshold and
        signed_aim >= args.hipad_plan_aim_threshold and
        signed_angle >= args.hipad_plan_angle_threshold and
        metrics["desired_speed"] >= args.hipad_plan_min_desired_speed
    ):
      return "positive"

    if (
        abs(metrics["spat_mean_x"]) <= args.hipad_normal_plan_spat_mean_max and
        abs(metrics["spat_last_x"]) <= args.hipad_normal_plan_spat_last_max and
        abs(metrics["aim_x"]) <= args.hipad_normal_aim_max and
        abs(metrics["angle_final"]) <= args.hipad_normal_angle_max
    ):
      return "negative"

    return "skip"

  def classify_frame(self, row: dict, run_name: str, args) -> str | None:
    label = super().classify_frame(row, run_name, args)
    if label != "brake" or not args.hipad_brake_require_neutral_plan:
      return label

    meta = row.get("_hipad_meta")
    if meta is None:
      return None

    metrics = self._plan_metrics(meta)
    if metrics is None:
      return None

    if self._plan_is_lateral_neutral(metrics, args):
      return label
    return None

  def filter_post_process_label(self, row: dict, label: str, action: str, run_name: str, args) -> str | None:
    if action != "brake" or not args.hipad_brake_require_neutral_plan:
      return label
    if label not in ("positive", "negative"):
      return label

    meta = row.get("_hipad_meta")
    if meta is None:
      return None

    metrics = self._plan_metrics(meta)
    if metrics is None:
      return None

    if self._plan_is_lateral_neutral(metrics, args):
      return label
    return None

  def manifest_extra(self, row: dict) -> dict:
    extra = super().manifest_extra(row)
    meta = row.get("_hipad_meta")
    metrics = self._plan_metrics(meta) if meta is not None else None
    if metrics is not None:
      extra.update(metrics)
    if row.get("_hipad_meta_path"):
      extra["hipad_meta_path"] = row["_hipad_meta_path"]
    return extra

  def vector_filename(self) -> str:
    return "hipad_plan_steering_vector.pt"

  @staticmethod
  def _meta_dir_for_log(log_source: Path, collection_root: Path) -> Path | None:
    run_name = log_source.parent.name if log_source.name == "activation_actions.jsonl" else log_source.name

    if log_source.name == "activation_actions.jsonl":
      log_dir = log_source.parent
      if log_dir.parent.name == "logs":
        candidate = log_dir.parent.parent / "images" / run_name / "metas"
        if candidate.exists():
          return candidate

    candidates = sorted(collection_root.rglob(f"images/{run_name}/metas"))
    return candidates[0] if candidates else None

  @staticmethod
  def _plan_metrics(meta: dict) -> dict | None:
    try:
      plan_spat = meta["plan_spat"]
      xs = [float(point[0]) for point in plan_spat]
      return {
          "spat_mean_x": sum(xs) / len(xs),
          "spat_first_x": xs[0],
          "spat_last_x": xs[-1],
          "aim_x": float(meta["aim"][0]),
          "angle_final": float(meta.get("angle_final", meta.get("angle", 0.0))),
          "desired_speed": float(meta.get("desired_speed", 0.0)),
      }
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
      return None

  @staticmethod
  def _plan_is_lateral_neutral(metrics: dict, args) -> bool:
    return (
        abs(metrics["spat_mean_x"]) <= args.hipad_normal_plan_spat_mean_max and
        abs(metrics["spat_last_x"]) <= args.hipad_normal_plan_spat_last_max and
        abs(metrics["aim_x"]) <= args.hipad_normal_aim_max and
        abs(metrics["angle_final"]) <= args.hipad_normal_angle_max
    )

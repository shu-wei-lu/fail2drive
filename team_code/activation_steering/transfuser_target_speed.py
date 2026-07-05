from __future__ import annotations

from activation_steering.base import PlannerAdapter


def matches_patterns(name: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
  if include_patterns and not any(pattern in name for pattern in include_patterns):
    return False
  if any(pattern in name for pattern in exclude_patterns):
    return False
  return True


class TransFuserTargetSpeedAdapter(PlannerAdapter):
  """Adapter for TF++/TransFuser target-speed-token steering."""

  name = "transfuser_target_speed"

  def add_post_process_args(self, parser) -> None:
    parser.add_argument("--brake-threshold", type=float, default=0.5)
    parser.add_argument(
        "--brake-min-target-speed-drop",
        type=float,
        default=1.0,
        help="Require current speed - target speed to be at least this many m/s for brake labels.",
    )
    parser.add_argument(
        "--brake-min-speed-ratio",
        type=float,
        default=1.1,
        help="Require current speed / target speed to be at least this value for brake labels.",
    )
    parser.add_argument(
        "--brake-min-run-length",
        type=int,
        default=2,
        help="Require at least this many consecutive brake frames in the same route log.",
    )
    parser.add_argument(
        "--brake-target-speed-state-threshold",
        type=float,
        default=None,
        help=(
            "Also label frames as brake when pred_target_speed/target_speed is at "
            "or below this threshold, even if the controller is not currently braking."
        ),
    )
    parser.add_argument("--normal-throttle-threshold", type=float, default=0.1)
    parser.add_argument("--min-speed", type=float, default=0.5)
    parser.add_argument("--exclude-stop-sign", action="store_true")
    parser.add_argument("--brake-include-pattern", action="append", default=[])
    parser.add_argument("--brake-exclude-pattern", action="append", default=[])
    parser.add_argument("--normal-include-pattern", action="append", default=[])
    parser.add_argument("--normal-exclude-pattern", action="append", default=[])

  def add_calibration_args(self, parser) -> None:
    parser.add_argument("--success-ratio", type=float, default=0.95)
    parser.add_argument(
        "--success-metric",
        choices=("target_speed", "target_speed_drop", "target_speed_ratio", "brake_or_target_speed"),
        default="target_speed",
        help="Metric used to decide whether a frame was forced enough.",
    )
    parser.add_argument("--target-speed-threshold", type=float, default=1.0)
    parser.add_argument("--target-speed-drop-threshold", type=float, default=8.0)
    parser.add_argument("--target-speed-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--brake-threshold", type=float, default=0.8)

  def brake_action(self, row: dict, args) -> bool:
    brake = float(row.get("brake", row.get("control_brake", 0.0)))
    control_brake = bool(row.get("control_brake", brake >= args.brake_threshold))
    return brake >= args.brake_threshold or control_brake

  def target_speed_for(self, row: dict) -> float | None:
    target_speed = row.get("pred_target_speed", row.get("target_speed"))
    if target_speed is None:
      return None
    return float(target_speed)

  def annotate_rows(self, rows: list[dict], args) -> list[dict]:
    run_start = None
    for index, row in enumerate(rows):
      row["_raw_brake_action"] = self.brake_action(row, args)
      if row["_raw_brake_action"]:
        if run_start is None:
          run_start = index
      elif run_start is not None:
        run_length = index - run_start
        for run_index in range(run_start, index):
          rows[run_index]["_brake_run_length"] = run_length
        run_start = None

    if run_start is not None:
      run_length = len(rows) - run_start
      for run_index in range(run_start, len(rows)):
        rows[run_index]["_brake_run_length"] = run_length

    for row in rows:
      row.setdefault("_brake_run_length", 0)
    return rows

  def classify_frame(self, row: dict, run_name: str, args) -> str | None:
    speed = float(row.get("speed", 0.0))
    throttle = float(row.get("throttle", 0.0))
    brake = float(row.get("brake", row.get("control_brake", 0.0)))
    stop_for_stop_sign = bool(row.get("stop_for_stop_sign", row.get("stop_sign_hazard", False)))

    if args.exclude_stop_sign and stop_for_stop_sign:
      return None
    if speed < args.min_speed:
      return None

    target_speed = self.target_speed_for(row)
    if args.brake_target_speed_state_threshold is not None and target_speed is not None:
      if target_speed <= args.brake_target_speed_state_threshold:
        if not matches_patterns(run_name, args.brake_include_pattern, args.brake_exclude_pattern):
          return None
        return "brake"

    if row.get("_raw_brake_action", brake >= args.brake_threshold):
      if not matches_patterns(run_name, args.brake_include_pattern, args.brake_exclude_pattern):
        return None
      if int(row.get("_brake_run_length", 0)) < args.brake_min_run_length:
        return None

      if target_speed is not None:
        speed_drop = speed - target_speed
        speed_ratio = speed / max(target_speed, 1e-6)
        if speed_drop < args.brake_min_target_speed_drop and speed_ratio < args.brake_min_speed_ratio:
          return None

      return "brake"

    if brake < args.brake_threshold and throttle >= args.normal_throttle_threshold:
      if not matches_patterns(run_name, args.normal_include_pattern, args.normal_exclude_pattern):
        return None
      return "normal"
    return None

  def calibration_metric(self, row: dict) -> float:
    target_speed = self.target_speed_for(row)
    return 1e9 if target_speed is None else target_speed

  def is_forced_frame(self, row: dict, args, baseline_row: dict | None = None) -> bool:
    pred_speed = self.calibration_metric(row)
    brake = float(row.get("brake", 0.0))
    if args.success_metric == "target_speed":
      return pred_speed < args.target_speed_threshold
    if args.success_metric == "brake_or_target_speed":
      return pred_speed < args.target_speed_threshold or brake > args.brake_threshold

    if baseline_row is None:
      return False
    baseline_speed = self.calibration_metric(baseline_row)
    if args.success_metric == "target_speed_drop":
      return baseline_speed - pred_speed >= args.target_speed_drop_threshold
    if args.success_metric == "target_speed_ratio":
      return baseline_speed > 1e-6 and pred_speed / baseline_speed <= args.target_speed_ratio_threshold
    raise ValueError(f"Unsupported success metric: {args.success_metric}")

  def manifest_extra(self, row: dict) -> dict:
    target_speed = self.target_speed_for(row)
    speed = float(row.get("speed", 0.0))
    return {
        "target_speed": target_speed,
        "target_speed_drop": None if target_speed is None else speed - target_speed,
        "throttle": float(row.get("throttle", 0.0)),
        "brake": float(row.get("brake", row.get("control_brake", 0.0))),
        "brake_run_length": int(row.get("_brake_run_length", 0)),
    }

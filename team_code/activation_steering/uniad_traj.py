from __future__ import annotations

from activation_steering.base import PlannerAdapter


class UniADTrajectoryAdapter(PlannerAdapter):
  name = "uniad_traj"

  def classify_frame(self, row: dict, run_name: str, args) -> str | None:
    raise NotImplementedError("UniAD activation steering adapter is a placeholder.")

  def calibration_metric(self, row: dict) -> float:
    raise NotImplementedError("UniAD activation steering adapter is a placeholder.")

  def is_forced_frame(self, row: dict, args, baseline_row: dict | None = None) -> bool:
    raise NotImplementedError("UniAD activation steering adapter is a placeholder.")

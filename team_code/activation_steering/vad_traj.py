from __future__ import annotations

from activation_steering.base import PlannerAdapter


class VADTrajectoryAdapter(PlannerAdapter):
  name = "vad_traj"

  def classify_frame(self, row: dict, run_name: str, args) -> str | None:
    raise NotImplementedError("VAD activation steering adapter is a placeholder.")

  def calibration_metric(self, row: dict) -> float:
    raise NotImplementedError("VAD activation steering adapter is a placeholder.")

  def is_forced_frame(self, row: dict, args, baseline_row: dict | None = None) -> bool:
    raise NotImplementedError("VAD activation steering adapter is a placeholder.")

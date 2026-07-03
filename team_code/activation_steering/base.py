from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import torch


class PlannerAdapter(ABC):
  """Planner-specific hooks for activation collection, vectors, and calibration."""

  name = "base"

  def add_post_process_args(self, parser) -> None:
    return None

  def add_calibration_args(self, parser) -> None:
    return None

  def feature_path_for(self, row: dict, log_source: Path, features_root: Path, model_index: int) -> Path:
    explicit = row.get("feature_path")
    if explicit:
      path = Path(explicit)
      if path.exists():
        return path

    run_name = log_source.parent.name if log_source.name == "activation_actions.jsonl" else log_source.name
    feature_dir = features_root / run_name
    model_dir = feature_dir / f"model_{model_index:02d}"
    if model_dir.exists():
      feature_dir = model_dir
    return feature_dir / f"{int(row['frame']):06d}.pt"

  def load_feature(self, path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")

  def annotate_rows(self, rows: list[dict], args) -> list[dict]:
    return rows

  def augment_rows(self, rows: list[dict], log_source: Path, collection_root: Path) -> list[dict]:
    return rows

  def classify_post_process_label(self, row: dict, action: str, run_name: str, args) -> str | None:
    return None

  def filter_post_process_label(self, row: dict, label: str, action: str, run_name: str, args) -> str | None:
    return label

  @abstractmethod
  def classify_frame(self, row: dict, run_name: str, args) -> str | None:
    """Return 'brake', 'normal', or None for vector construction."""

  @abstractmethod
  def calibration_metric(self, row: dict) -> float:
    """Scalar planner output used for alpha calibration."""

  @abstractmethod
  def is_forced_frame(self, row: dict, args, baseline_row: dict | None = None) -> bool:
    """Whether one frame satisfies the steering success criterion."""

  def manifest_extra(self, row: dict) -> dict:
    return {}

  def vector_filename(self) -> str:
    return "brake_minus_normal.pt"

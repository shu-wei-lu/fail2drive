from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F


class ActivationInjector:
  """Applies activation steering vectors for one planner hook.

  Scalar alpha preserves the old single-vector behavior. A length-3 alpha
  applies action vectors in [brake, left, right] order.
  """

  ACTIONS = ("brake", "left", "right")

  def __init__(
      self,
      vector_path: Path | None = None,
      vector_paths: list[Path | None] | None = None,
      normalize: bool = True,
  ):
    self.vector_path = Path(vector_path) if vector_path else None
    self.vector_paths = [Path(path) if path else None for path in vector_paths] if vector_paths else None
    self.normalize = bool(normalize)
    self._vector = None
    self._vectors = None
    self._warned_disabled = False
    self.verbose = os.environ.get("ACTIVATION_INJECTOR_VERBOSE", "0").lower() in ("1", "true", "t", "yes", "y")

  @classmethod
  def from_env(cls, default_vector_path: Path) -> "ActivationInjector":
    vector_path = os.environ.get("ACTIVATION_VECTOR_PATH")
    vector_paths = cls._vector_paths_from_env()
    enable_default = os.environ.get("ENABLE_ACTIVATION_STEERING", os.environ.get("ENABLE_ACTIVATION_INJECTOR", "0"))
    if vector_path is None and vector_paths is None and bool(int(enable_default)):
      vector_path = str(default_vector_path)

    return cls(
        vector_path=Path(vector_path) if vector_path else None,
        vector_paths=vector_paths,
        normalize=bool(int(os.environ.get("NORMALIZE_STEERING_VECTOR", os.environ.get("NORMALIZE_ACTIVATION_VECTOR", 1)))),
    )

  def enabled(self, alpha: float) -> bool:
    return self.alpha_value(alpha) > 0.0 and (self.vector_path is not None or self._has_action_vectors())

  def alpha_value(self, alpha: float | None) -> float:
    if alpha is None:
      return 0.0
    values = torch.as_tensor(alpha).detach().cpu().flatten().float()
    if values.numel() == 0:
      return 0.0
    return float(torch.max(torch.abs(values)))

  def alpha_vector(self, alpha: float | None) -> torch.Tensor:
    if alpha is None:
      return torch.zeros(len(self.ACTIONS), dtype=torch.float32)
    values = torch.as_tensor(alpha).detach().cpu().flatten().float()
    if values.numel() == 1:
      vector = torch.zeros(len(self.ACTIONS), dtype=torch.float32)
      vector[0] = values[0]
      return vector
    if values.numel() != len(self.ACTIONS):
      raise ValueError(f"Activation alpha must be scalar or length {len(self.ACTIONS)} [brake, left, right].")
    return values

  def vector(self, reference: torch.Tensor) -> torch.Tensor:
    if self.vector_path is None:
      raise RuntimeError("ActivationInjector has no vector path. Set ACTIVATION_VECTOR_PATH to enable it.")
    if self._vector is None:
      self._vector = self._load_vector()
    return self._vector.to(device=reference.device, dtype=reference.dtype)

  def _load_vector(self) -> torch.Tensor:
    if self.vector_path is None:
      raise RuntimeError("ActivationInjector has no vector path. Set ACTIVATION_VECTOR_PATH to enable it.")
    vector = torch.load(self.vector_path, map_location="cpu").detach().float()
    if self.normalize:
      vector = F.normalize(vector.reshape(1, -1), dim=1).reshape_as(vector)
    return vector

  def apply(self, features: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
    alpha_value = self.alpha_value(alpha)
    if alpha_value <= 0.0:
      return features
    alpha_vector = self.alpha_vector(alpha)
    if alpha_vector.numel() == len(self.ACTIONS) and torch.count_nonzero(alpha_vector[1:]).item() > 0:
      return self._apply_action_vectors(features, alpha_vector)
    if self._has_action_vectors():
      return self._apply_action_vectors(features, alpha_vector)
    if self.vector_path is None:
      if not self._warned_disabled:
        print("[ActivationInjector] disabled: set ACTIVATION_VECTOR_PATH to enable activation steering", flush=True)
        self._warned_disabled = True
      return features
    if self.verbose:
      print("[ActivationInjector] apply alpha =", alpha_value)
    return features + float(alpha_vector[0]) * self.vector(features)

  def _apply_action_vectors(self, features: torch.Tensor, alpha_vector: torch.Tensor) -> torch.Tensor:
    if not self._has_action_vectors():
      if not self._warned_disabled:
        print(
            "[ActivationInjector] disabled: set ACTIVATION_VECTOR_PATHS or "
            "BRAKE/LEFT/RIGHT_ACTIVATION_VECTOR_PATH to enable action steering",
            flush=True)
        self._warned_disabled = True
      return features

    vectors = self.action_vectors(features)
    result = features
    alpha_vector = alpha_vector.to(device=features.device, dtype=features.dtype)
    active = []
    for index, alpha in enumerate(alpha_vector):
      if float(alpha.detach().cpu()) == 0.0:
        continue
      vector = vectors[index]
      if vector is None:
        continue
      active.append(f"{self.ACTIONS[index]}={float(alpha.detach().cpu()):.3f}")
      result = result + alpha * vector
    if active and self.verbose:
      print("[ActivationInjector] apply actions:", ", ".join(active))
    return result

  def action_vectors(self, reference: torch.Tensor) -> list[torch.Tensor | None]:
    if self.vector_paths is None:
      raise RuntimeError("ActivationInjector has no action vector paths.")
    if self._vectors is None:
      self._vectors = [self._load_action_vector(path) for path in self.vector_paths]
    return [None if vector is None else vector.to(device=reference.device, dtype=reference.dtype) for vector in self._vectors]

  def _load_action_vector(self, path: Path | None) -> torch.Tensor | None:
    if path is None:
      return None
    vector = torch.load(path, map_location="cpu").detach().float()
    if self.normalize:
      vector = F.normalize(vector.reshape(1, -1), dim=1).reshape_as(vector)
    return vector

  def _has_action_vectors(self) -> bool:
    return self.vector_paths is not None and any(path is not None for path in self.vector_paths)

  @classmethod
  def _vector_paths_from_env(cls) -> list[Path | None] | None:
    paths_spec = os.environ.get("ACTIVATION_VECTOR_PATHS")
    if paths_spec:
      raw_paths = [item.strip() for item in paths_spec.split(",")]
      if len(raw_paths) != len(cls.ACTIONS):
        raise ValueError("ACTIVATION_VECTOR_PATHS must contain exactly 3 comma-separated paths: brake,left,right.")
      return [Path(path) if path else None for path in raw_paths]

    action_paths = [
        os.environ.get("BRAKE_ACTIVATION_VECTOR_PATH", os.environ.get("ACTIVATION_VECTOR_PATH_BRAKE")),
        os.environ.get("LEFT_ACTIVATION_VECTOR_PATH", os.environ.get("ACTIVATION_VECTOR_PATH_LEFT")),
        os.environ.get("RIGHT_ACTIVATION_VECTOR_PATH", os.environ.get("ACTIVATION_VECTOR_PATH_RIGHT")),
    ]
    if any(action_paths):
      return [Path(path) if path else None for path in action_paths]
    return None

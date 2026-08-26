from __future__ import annotations

import os
from pathlib import Path

import torch


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
      action_alpha_scales: list[float] | None = None,
  ):
    self.vector_path = Path(vector_path) if vector_path else None
    self.vector_paths = [Path(path) if path else None for path in vector_paths] if vector_paths else None
    self.action_alpha_scales = action_alpha_scales or [1.0 for _ in self.ACTIONS]
    # Cache loaded payloads by their resolved file path. A configured path may
    # either be a legacy .pt file or a directory containing one vector per
    # decoder layer.
    self._payload_cache: dict[Path, object] = {}
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
        action_alpha_scales=cls._action_alpha_scales_from_env(),
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

  def vector(
      self,
      reference: torch.Tensor,
      layer_index: int | None = None,
      num_layers: int | None = None,
  ) -> torch.Tensor:
    if self.vector_path is None:
      raise RuntimeError("ActivationInjector has no vector path. Set ACTIVATION_VECTOR_PATH to enable it.")
    return self._vector_for_path(
        self.vector_path, reference, layer_index=layer_index, num_layers=num_layers)

  def apply(
      self,
      features: torch.Tensor,
      alpha: float | None = None,
      layer_index: int | None = None,
      num_layers: int | None = None,
  ) -> torch.Tensor:
    alpha_value = self.alpha_value(alpha)
    if alpha_value <= 0.0:
      return features
    alpha_vector = self.alpha_vector(alpha)
    if alpha_vector.numel() == len(self.ACTIONS) and torch.count_nonzero(alpha_vector[1:]).item() > 0:
      return self._apply_action_vectors(features, alpha_vector, layer_index, num_layers)
    if self._has_action_vectors():
      return self._apply_action_vectors(features, alpha_vector, layer_index, num_layers)
    if self.vector_path is None:
      if not self._warned_disabled:
        print("[ActivationInjector] disabled: set ACTIVATION_VECTOR_PATH to enable activation steering", flush=True)
        self._warned_disabled = True
      return features
    if self.verbose:
      print(f"[ActivationInjector] apply layer={layer_index} alpha={alpha_value}")
    return features + float(alpha_vector[0]) * self.vector(features, layer_index, num_layers)

  def apply_projection_gated(
      self,
      features: torch.Tensor,
      alpha: float | None = None,
      low: float = 0.05,
      high: float = 0.10,
      gated_actions: tuple[str, ...] = ("left", "right"),
      layer_index: int | None = None,
      num_layers: int | None = None,
      verbose: bool = False,
      log_prefix: str = "ActivationInjector",
  ) -> torch.Tensor:
    """Apply steering vectors with an inverse scalar-projection gate.

    The projection coordinate is ``dot(features - negative_mean, vector) /
    dot(vector, vector)``. Therefore the negative prototype has coordinate 0
    and the positive prototype has coordinate 1. Coordinates at or below
    ``low`` use the full alpha; coordinates at or above ``high`` use zero
    alpha. A legacy single vector is always gated because its action identity
    is not available. For multi-action vectors, ``gated_actions`` controls
    which of [brake, left, right] are gated.
    """
    if high <= low:
      raise ValueError("Projection gate high threshold must be greater than low threshold.")
    if self.alpha_value(alpha) <= 0.0:
      return features

    alpha_vector = self.alpha_vector(alpha).to(device=features.device, dtype=features.dtype)
    if self._has_action_vectors():
      vectors = self.action_vectors(features, layer_index, num_layers)
      alpha_scales = torch.as_tensor(
          self.action_alpha_scales, device=features.device, dtype=features.dtype)
      result = features
      for action_index, (action_alpha, vector, vector_path) in enumerate(
          zip(alpha_vector * alpha_scales, vectors, self.vector_paths)):
        if vector is None or float(action_alpha.detach().cpu()) == 0.0:
          continue
        action = self.ACTIONS[action_index]
        gate = 1.0
        projection = None
        if action in gated_actions:
          negative_mean = self._negative_mean_for_path(
              vector_path, features, layer_index, num_layers)
          projection, gate = self._projection_gate(
              features, negative_mean, vector, low, high)
        result = result + action_alpha * gate * vector
        if verbose:
          self._print_projection_gate(log_prefix, action, projection, gate, action_alpha)
      return result

    if self.vector_path is None:
      if not self._warned_disabled:
        print("[ActivationInjector] disabled: set ACTIVATION_VECTOR_PATH to enable activation steering", flush=True)
        self._warned_disabled = True
      return features

    vector = self.vector(features, layer_index, num_layers)
    negative_mean = self._negative_mean_for_path(
        self.vector_path, features, layer_index, num_layers)
    projection, gate = self._projection_gate(features, negative_mean, vector, low, high)
    if verbose:
      self._print_projection_gate(log_prefix, "single", projection, gate, alpha_vector[0])
    return features + alpha_vector[0] * gate * vector

  @staticmethod
  def _projection_gate(
      features: torch.Tensor,
      negative_mean: torch.Tensor,
      vector: torch.Tensor,
      low: float,
      high: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if vector.ndim != features.ndim:
      raise ValueError(
          f"Projection gate requires matching ranks, got feature rank {features.ndim} "
          f"and vector rank {vector.ndim}.")
    if not all(vector_size in (1, feature_size)
               for vector_size, feature_size in zip(vector.shape, features.shape)):
      raise ValueError(
          f"Projection gate vector shape {tuple(vector.shape)} cannot be broadcast to "
          f"feature shape {tuple(features.shape)}.")
    if negative_mean.ndim != features.ndim:
      raise ValueError(
          f"Projection gate requires matching ranks, got feature rank {features.ndim} "
          f"and negative-mean rank {negative_mean.ndim}.")
    if not all(mean_size in (1, feature_size)
               for mean_size, feature_size in zip(negative_mean.shape, features.shape)):
      raise ValueError(
          f"Projection gate negative-mean shape {tuple(negative_mean.shape)} cannot be "
          f"broadcast to feature shape {tuple(features.shape)}.")

    reference_vector = vector.expand(*features.shape)
    reference_mean = negative_mean.expand(*features.shape)
    centered_flat = (
        features.detach().float() - reference_mean.detach().float()
    ).reshape(features.shape[0], -1)
    vector_flat = reference_vector.detach().float().reshape(reference_vector.shape[0], -1)
    vector_norm_sq = torch.sum(vector_flat.square(), dim=1).clamp_min(1e-12)
    projection = torch.sum(centered_flat * vector_flat, dim=1) / vector_norm_sq
    gate = ((high - projection) / (high - low)).clamp(0.0, 1.0)
    gate = gate.to(device=features.device, dtype=features.dtype)
    gate = gate.reshape(features.shape[0], *([1] * (features.ndim - 1)))
    return projection, gate

  @staticmethod
  def _print_projection_gate(log_prefix, action, projection, gate, alpha):
    projection_text = "disabled" if projection is None else projection.detach().cpu().tolist()
    gate_text = gate if isinstance(gate, float) else gate.detach().cpu().flatten().tolist()
    print(
        f"[{log_prefix} projection gate] action={action}, projection={projection_text}, "
        f"gate={gate_text}, alpha={float(alpha.detach().cpu()):.4f}",
        flush=True,
    )

  def apply_cosine_gated(self, *args, **kwargs) -> torch.Tensor:
    """Backward-compatible name; the gate now uses scalar projection."""
    return self.apply_projection_gated(*args, **kwargs)

  def _apply_action_vectors(
      self,
      features: torch.Tensor,
      alpha_vector: torch.Tensor,
      layer_index: int | None,
      num_layers: int | None,
  ) -> torch.Tensor:
    if not self._has_action_vectors():
      if not self._warned_disabled:
        print(
            "[ActivationInjector] disabled: set ACTIVATION_VECTOR_PATHS or "
            "BRAKE/LEFT/RIGHT_ACTIVATION_VECTOR_PATH to enable action steering",
            flush=True)
        self._warned_disabled = True
      return features

    vectors = self.action_vectors(features, layer_index, num_layers)
    result = features
    alpha_vector = alpha_vector.to(device=features.device, dtype=features.dtype)
    alpha_scales = torch.as_tensor(self.action_alpha_scales, device=features.device, dtype=features.dtype)
    scaled_alpha_vector = alpha_vector * alpha_scales
    active = []
    for index, alpha in enumerate(scaled_alpha_vector):
      effective_alpha = float(alpha.detach().cpu())
      if effective_alpha == 0.0:
        continue
      vector = vectors[index]
      if vector is None:
        continue
      raw_alpha = float(alpha_vector[index].detach().cpu())
      scale = float(alpha_scales[index].detach().cpu())
      if scale == 1.0:
        active.append(f"{self.ACTIONS[index]}={effective_alpha:.3f}")
      else:
        active.append(f"{self.ACTIONS[index]}={effective_alpha:.3f}({raw_alpha:.3f}x{scale:.3f})")
      result = result + alpha * vector
    if active and self.verbose:
      print(f"[ActivationInjector] apply layer={layer_index} actions:", ", ".join(active))
    return result

  def action_vectors(
      self,
      reference: torch.Tensor,
      layer_index: int | None = None,
      num_layers: int | None = None,
  ) -> list[torch.Tensor | None]:
    if self.vector_paths is None:
      raise RuntimeError("ActivationInjector has no action vector paths.")
    return [
        None if path is None else self._vector_for_path(
            path, reference, layer_index=layer_index, num_layers=num_layers)
        for path in self.vector_paths
    ]

  def _vector_for_path(
      self,
      configured_path: Path,
      reference: torch.Tensor,
      layer_index: int | None,
      num_layers: int | None,
  ) -> torch.Tensor:
    path = self._resolve_layer_path(configured_path, layer_index)
    if path not in self._payload_cache:
      self._payload_cache[path] = torch.load(path, map_location="cpu")
    vector = self._select_layer_vector(
        self._payload_cache[path], layer_index=layer_index, num_layers=num_layers)
    if not torch.is_tensor(vector):
      raise TypeError(f"Activation vector from {path} is not a tensor (got {type(vector).__name__}).")
    return vector.detach().float().to(device=reference.device, dtype=reference.dtype)

  def _negative_mean_for_path(
      self,
      configured_path: Path | None,
      reference: torch.Tensor,
      layer_index: int | None,
      num_layers: int | None,
  ) -> torch.Tensor:
    if configured_path is None:
      raise RuntimeError("Projection gate requires an activation vector path.")
    vector_path = self._resolve_layer_path(configured_path, layer_index)
    candidates = [vector_path.with_name("negative_mean.pt")]
    if configured_path.is_dir() and layer_index is not None:
      layer_name = f"layer_{layer_index:02d}"
      candidates.extend((
          configured_path / f"negative_mean_{layer_name}.pt",
          configured_path / f"{layer_name}_negative_mean.pt",
      ))
    mean_path = next((path for path in candidates if path.is_file()), None)
    if mean_path is None:
      expected = ", ".join(str(path) for path in candidates)
      raise FileNotFoundError(
          f"Projection gate requires the negative prototype for {vector_path}. "
          f"Expected one of: {expected}")
    if mean_path not in self._payload_cache:
      self._payload_cache[mean_path] = torch.load(mean_path, map_location="cpu")
    negative_mean = self._select_layer_vector(
        self._payload_cache[mean_path], layer_index=layer_index, num_layers=num_layers)
    if not torch.is_tensor(negative_mean):
      raise TypeError(
          f"Negative mean from {mean_path} is not a tensor "
          f"(got {type(negative_mean).__name__}).")
    return negative_mean.detach().float().to(
        device=reference.device, dtype=reference.dtype)

  @staticmethod
  def _resolve_layer_path(configured_path: Path, layer_index: int | None) -> Path:
    if not configured_path.is_dir():
      return configured_path
    if layer_index is None:
      raise ValueError(
          f"Layer-specific activation vector directory {configured_path} requires a layer index.")

    layer_name = f"layer_{layer_index:02d}"
    candidates = (
        configured_path / layer_name / "steering_vector.pt",
        configured_path / f"{layer_name}.pt",
        configured_path / f"steering_vector_{layer_name}.pt",
    )
    for candidate in candidates:
      if candidate.is_file():
        return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No activation vector found for decoder layer {layer_index} under {configured_path}. "
        f"Expected one of: {expected}")

  @staticmethod
  def _select_layer_vector(payload, layer_index: int | None, num_layers: int | None):
    """Select one layer from a packed payload, or return a legacy tensor.

    Supported packed formats are {"vectors": tensor/list/dict}, a dict keyed
    by layer_00/layer_01/..., or a tensor shaped [num_layers, *feature_shape].
    The last tensor format intentionally keeps the feature batch dimension, so
    an align-query bank normally has shape [6, 1, 48, 256].
    """
    if isinstance(payload, dict):
      vectors = payload.get("vectors", payload)
      if layer_index is None:
        raise ValueError("A packed layer-specific activation vector requires a layer index.")
      if isinstance(vectors, dict):
        for key in (f"layer_{layer_index:02d}", str(layer_index), layer_index):
          if key in vectors:
            return vectors[key]
        raise KeyError(f"Packed activation vector has no entry for decoder layer {layer_index}.")
      if isinstance(vectors, (list, tuple)):
        return vectors[layer_index]
      payload = vectors

    if torch.is_tensor(payload):
      if (
          layer_index is not None and
          num_layers is not None and
          payload.ndim >= 1 and
          payload.shape[0] == num_layers
      ):
        return payload[layer_index]
      return payload
    return payload

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

  @classmethod
  def _action_alpha_scales_from_env(cls) -> list[float]:
    scales_spec = os.environ.get("ACTIVATION_ACTION_ALPHA_SCALES")
    if scales_spec:
      raw_scales = [item.strip() for item in scales_spec.split(",")]
      if len(raw_scales) != len(cls.ACTIONS):
        raise ValueError("ACTIVATION_ACTION_ALPHA_SCALES must contain exactly 3 comma-separated values: brake,left,right.")
      return [float(value) for value in raw_scales]

    return [
        float(os.environ.get("BRAKE_ACTIVATION_ALPHA_SCALE", "1.0")),
        float(os.environ.get("LEFT_ACTIVATION_ALPHA_SCALE", "1.0")),
        float(os.environ.get("RIGHT_ACTIVATION_ALPHA_SCALE", "1.0")),
    ]

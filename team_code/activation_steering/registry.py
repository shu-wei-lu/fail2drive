from __future__ import annotations

from activation_steering.transfuser_target_speed import TransFuserTargetSpeedAdapter
from activation_steering.uniad_traj import UniADTrajectoryAdapter
from activation_steering.vad_traj import VADTrajectoryAdapter


_ADAPTERS = {
    TransFuserTargetSpeedAdapter.name: TransFuserTargetSpeedAdapter,
    VADTrajectoryAdapter.name: VADTrajectoryAdapter,
    UniADTrajectoryAdapter.name: UniADTrajectoryAdapter,
}


def adapter_names() -> tuple[str, ...]:
  return tuple(_ADAPTERS)


def get_adapter(name: str):
  try:
    return _ADAPTERS[name]()
  except KeyError as exc:
    raise ValueError(f"Unknown activation steering adapter '{name}'. Options: {', '.join(adapter_names())}") from exc

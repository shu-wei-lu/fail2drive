from __future__ import annotations

import math
import os


def _clip01(value: float) -> float:
  return max(0.0, min(1.0, float(value)))


def _strtobool(value: str | None) -> bool:
  return str(value).lower() in ("1", "true", "t", "yes", "y", "on")


def _parse_points(spec: str) -> list[tuple[float, float]]:
  points = []
  for item in spec.split(","):
    if not item.strip():
      continue
    x_raw, y_raw = item.split(":", maxsplit=1)
    points.append((_clip01(float(x_raw)), float(y_raw)))
  if not points:
    raise ValueError("VLM_ALPHA_POINTS must contain at least one x:y point")
  return sorted(points)


def _interp(points: list[tuple[float, float]], x: float) -> float:
  x = _clip01(x)
  if x <= points[0][0]:
    return points[0][1]
  if x >= points[-1][0]:
    return points[-1][1]
  for (x0, y0), (x1, y1) in zip(points, points[1:]):
    if x0 <= x <= x1:
      if x1 == x0:
        return y1
      t = (x - x0) / (x1 - x0)
      return y0 + t * (y1 - y0)
  return points[-1][1]


class ActivationPolicy:
  """Maps each agent frame to an activation steering alpha."""

  def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> float:
    return 0.0


class FixedAfterFramePolicy(ActivationPolicy):
  def __init__(self, alpha: float, start_frame: int, end_frame: int | None = None):
    self.fixed_alpha = float(alpha)
    self.start_frame = int(start_frame)
    self.end_frame = None if end_frame is None or int(end_frame) < 0 else int(end_frame)

  def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> float:
    if frame < self.start_frame:
      return 0.0
    if self.end_frame is not None and frame > self.end_frame:
      return 0.0
    return self.fixed_alpha


class VLMPolicy(ActivationPolicy):
  def __init__(
      self,
      alpha_max: float = 1.0,
      mapping: str = "linear",
      deadzone: float = 0.0,
      gamma: float = 1.0,
      points: str | None = None,
      decay_frames: int = 10,
  ):
    self.alpha_max = float(alpha_max)
    self.mapping = mapping
    self.deadzone = _clip01(deadzone)
    self.gamma = float(gamma)
    self.points = _parse_points(points) if points else None
    self.decay_frames = int(decay_frames)

  def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> float:
    if vlm_decision is None:
      return 0.0
    score = _clip01(vlm_decision.steering_alpha)
    if score <= self.deadzone:
      return 0.0

    if self.mapping == "linear":
      mapped = score * self.alpha_max
    elif self.mapping == "power":
      scaled = (score - self.deadzone) / max(1.0 - self.deadzone, 1e-6)
      mapped = self.alpha_max * (scaled ** self.gamma)
    elif self.mapping == "piecewise":
      points = self.points or [
          (0.0, 0.0),
          (0.3, 0.0),
          (0.6, 4.0),
          (0.8, 4.5),
          (1.0, self.alpha_max),
      ]
      mapped = _interp(points, score)
    else:
      raise ValueError("VLM_ALPHA_MAPPING must be one of: linear, power, piecewise")

    if decision_age_frames is not None and self.decay_frames > 0:
      decay = max(0.0, 1.0 - (float(decision_age_frames) / float(self.decay_frames)))
      mapped = mapped * decay

    return max(0.0, min(self.alpha_max, float(mapped)))


class OraclePolicy(ActivationPolicy):
  """Uses scenario-runner oracle state to gate activation steering.

  This follows the same privileged signal PDM-Lite uses for route-obstacle
  handling: CarlaDataProvider.active_scenarios stores the currently relevant
  scenario actors. It returns [brake, left, right] action alphas.
  """

  ACTIONS = ("brake", "left", "right")
  ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

  def __init__(
      self,
      alpha: float,
      action: str = "auto",
      trigger_distance: float = 35.0,
      min_distance: float = 0.0,
      brake_hazard_distance: float = 20.0,
      brake_hazard_lateral_margin: float = 2.5,
      brake_reaction_time: float = 0.4,
      brake_deceleration: float = 6.0,
      brake_distance_margin: float = 2.0,
      brake_ttc_threshold: float = 2.0,
      brake_min_closing_speed: float = 0.5,
      hold_frames: int = 12,
      cooldown_frames: int = 30,
      allow_multi_action: bool = False,
      verbose: bool = False,
  ):
    self.fixed_alpha = float(alpha)
    self.action = action.lower().strip()
    self.trigger_distance = float(trigger_distance)
    self.min_distance = float(min_distance)
    self.brake_hazard_distance = float(brake_hazard_distance)
    self.brake_hazard_lateral_margin = float(brake_hazard_lateral_margin)
    self.brake_reaction_time = float(brake_reaction_time)
    self.brake_deceleration = max(float(brake_deceleration), 1e-3)
    self.brake_distance_margin = float(brake_distance_margin)
    self.brake_ttc_threshold = float(brake_ttc_threshold)
    self.brake_min_closing_speed = float(brake_min_closing_speed)
    self.hold_frames = max(1, int(hold_frames))
    self.cooldown_frames = max(0, int(cooldown_frames))
    self.allow_multi_action = bool(allow_multi_action)
    self.verbose = bool(verbose)
    self._active_until = [-1 for _ in self.ACTIONS]
    self._cooldown_until = [-1 for _ in self.ACTIONS]
    self._last_trigger_key = None

  @classmethod
  def from_env(cls) -> "OraclePolicy":
    action = os.environ.get("ORACLE_ACTION", os.environ.get("ACTIVATION_ACTION", "auto"))
    alpha = float(os.environ.get("ORACLE_ALPHA", os.environ.get("STEERING_ALPHA", os.environ.get("ACTIVATION_ALPHA", 1.0))))
    return cls(
        alpha=alpha,
        action=action,
        trigger_distance=float(os.environ.get("ORACLE_TRIGGER_DISTANCE", 35.0)),
        min_distance=float(os.environ.get("ORACLE_MIN_DISTANCE", 0.0)),
        brake_hazard_distance=float(os.environ.get("ORACLE_BRAKE_HAZARD_DISTANCE", 20.0)),
        brake_hazard_lateral_margin=float(os.environ.get("ORACLE_BRAKE_HAZARD_LATERAL_MARGIN", 2.5)),
        brake_reaction_time=float(os.environ.get("ORACLE_BRAKE_REACTION_TIME", 0.4)),
        brake_deceleration=float(os.environ.get("ORACLE_BRAKE_DECELERATION", 6.0)),
        brake_distance_margin=float(os.environ.get("ORACLE_BRAKE_DISTANCE_MARGIN", 2.0)),
        brake_ttc_threshold=float(os.environ.get("ORACLE_BRAKE_TTC_THRESHOLD", 2.0)),
        brake_min_closing_speed=float(os.environ.get("ORACLE_BRAKE_MIN_CLOSING_SPEED", 0.5)),
        hold_frames=int(os.environ.get("ORACLE_HOLD_FRAMES", 12)),
        cooldown_frames=int(os.environ.get("ORACLE_COOLDOWN_FRAMES", 30)),
        allow_multi_action=_strtobool(os.environ.get("ORACLE_ALLOW_MULTI_ACTION")),
        verbose=_strtobool(os.environ.get("ORACLE_VERBOSE")),
    )

  def alpha(self, frame: int) -> list[float]:
    alpha_vector = [0.0 for _ in self.ACTIONS]
    if self.fixed_alpha <= 0.0:
      return alpha_vector

    for action_index in range(len(self.ACTIONS)):
      if frame <= self._active_until[action_index]:
        alpha_vector[action_index] = self.fixed_alpha

    if any(value > 0.0 for value in alpha_vector):
      return alpha_vector

    triggers = self._oracle_triggers()
    if not triggers:
      return alpha_vector

    selected = self._select_triggers(triggers)
    for action, scenario_type, actor_id, distance in selected:
      action_index = self.ACTION_INDEX[action]
      if frame < self._cooldown_until[action_index]:
        continue
      self._active_until[action_index] = frame + self.hold_frames - 1
      self._cooldown_until[action_index] = self._active_until[action_index] + self.cooldown_frames
      alpha_vector[action_index] = self.fixed_alpha
      self._log_trigger(frame, action, scenario_type, actor_id, distance)
      if not self.allow_multi_action:
        break
    return alpha_vector

  def _oracle_triggers(self) -> list[tuple[str, str, int | None, float]]:
    try:
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=import-outside-toplevel
    except Exception:
      return []

    try:
      ego = CarlaDataProvider.get_hero_actor()
    except Exception:
      ego = None
    if ego is None or not getattr(ego, "is_alive", True):
      return []

    ego_location = ego.get_location()
    triggers = []
    for scenario_type, scenario_data in list(getattr(CarlaDataProvider, "active_scenarios", [])):
      actor = self._first_alive_actor(scenario_data)
      if actor is None:
        continue
      action = self._action_from_autopilot_state(scenario_data, ego, actor)
      if action is None or not self._action_allowed(action):
        continue
      distance = self._horizontal_distance(ego_location, actor.get_location())
      if self.min_distance <= distance <= self.trigger_distance:
        actor_id = getattr(actor, "id", None)
        triggers.append((action, scenario_type, actor_id, distance))
    return triggers

  def _select_triggers(self, triggers: list[tuple[str, str, int | None, float]]) -> list[tuple[str, str, int | None, float]]:
    if self.allow_multi_action:
      best_by_action = {}
      for trigger in triggers:
        action = trigger[0]
        if action not in best_by_action or trigger[3] < best_by_action[action][3]:
          best_by_action[action] = trigger
      return [best_by_action[action] for action in self.ACTIONS if action in best_by_action]

    # Keep the intervention one-hot. Braking has priority because an incorrect
    # lateral intervention is usually worse than staying conservative.
    action_priority = {"brake": 0, "left": 1, "right": 1}
    return [min(triggers, key=lambda item: (action_priority[item[0]], item[3]))]

  def _log_trigger(self, frame: int, action: str, scenario_type: str, actor_id: int | None, distance: float) -> None:
    if not self.verbose:
      return
    trigger_key = (action, scenario_type, actor_id, int(distance))
    if trigger_key == self._last_trigger_key:
      return
    self._last_trigger_key = trigger_key
    print(
        f"[OraclePolicy] frame={frame} action={action} scenario={scenario_type} actor={actor_id} "
        f"distance={distance:.1f} alpha={self.fixed_alpha:.3f}",
        flush=True)

  def _action_from_autopilot_state(self, scenario_data, ego=None, actor=None) -> str | None:
    direction = self._scenario_direction(scenario_data)
    if direction == "right":
      return "left"
    if direction == "left":
      return "right"

    invading_offset = self._invading_turn_offset(scenario_data)
    if invading_offset is not None:
      return "left" if invading_offset > 0.0 else "right"

    if ego is not None and actor is not None and self._scenario_actor_brake_hazard(ego, actor):
      return "brake"
    return None

  def _action_allowed(self, action: str) -> bool:
    if self.action in ("auto", "any", "all"):
      return True
    if self.action == action:
      return True
    if self.action in ("lane", "lane_change", "change_lane") and action in ("left", "right"):
      return True
    return False

  def _scenario_actor_brake_hazard(self, ego, actor) -> bool:
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()

    actor_location = actor.get_location()
    diff_x = actor_location.x - ego_location.x
    diff_y = actor_location.y - ego_location.y
    longitudinal = diff_x * forward.x + diff_y * forward.y
    if longitudinal < self.min_distance or longitudinal > self.brake_hazard_distance:
      return False

    lateral = abs(diff_x * right.x + diff_y * right.y)
    if lateral > self.brake_hazard_lateral_margin:
      return False

    ego_velocity = ego.get_velocity()
    actor_velocity = actor.get_velocity()
    ego_forward_speed = ego_velocity.x * forward.x + ego_velocity.y * forward.y
    actor_forward_speed = actor_velocity.x * forward.x + actor_velocity.y * forward.y
    closing_speed = ego_forward_speed - actor_forward_speed
    if closing_speed < self.brake_min_closing_speed:
      return False

    ttc = longitudinal / max(closing_speed, 1e-3)
    stopping_distance = (
        max(ego_forward_speed, 0.0) * self.brake_reaction_time +
        max(ego_forward_speed, 0.0)**2 / (2.0 * self.brake_deceleration) +
        self.brake_distance_margin)
    return ttc <= self.brake_ttc_threshold or longitudinal <= stopping_distance

  @staticmethod
  def _first_alive_actor(scenario_data):
    if scenario_data is None:
      return None
    for item in scenario_data:
      if hasattr(item, "get_location") and getattr(item, "is_alive", True):
        return item
    return None

  @staticmethod
  def _scenario_direction(scenario_data) -> str | None:
    if scenario_data is None:
      return None
    for item in scenario_data:
      if isinstance(item, str) and item in ("left", "right"):
        return item
    return None

  @staticmethod
  def _invading_turn_offset(scenario_data) -> float | None:
    if scenario_data is None or len(scenario_data) < 3:
      return None
    item = scenario_data[2]
    if isinstance(item, bool):
      return None
    if isinstance(item, (int, float)) and abs(float(item)) > 1e-6:
      return float(item)
    return None

  @staticmethod
  def _horizontal_distance(location_a, location_b) -> float:
    dx = float(location_a.x - location_b.x)
    dy = float(location_a.y - location_b.y)
    return math.hypot(dx, dy)


def policy_from_env(vlm_enabled: bool = False) -> ActivationPolicy:
  if vlm_enabled:
    return VLMPolicy(
        alpha_max=float(os.environ.get("VLM_ALPHA_MAX", os.environ.get("ACTIVATION_ALPHA_MAX", 1.0))),
        mapping=os.environ.get("VLM_ALPHA_MAPPING", "linear"),
        deadzone=float(os.environ.get("VLM_ALPHA_DEADZONE", 0.0)),
        gamma=float(os.environ.get("VLM_ALPHA_GAMMA", 1.0)),
        points=os.environ.get("VLM_ALPHA_POINTS", None),
        decay_frames=int(os.environ.get("VLM_DECAY_FRAMES", 10)),
    )

  policy_name = os.environ.get("ACTIVATION_POLICY", os.environ.get("STEERING_POLICY", "")).lower()
  if policy_name == "oracle" or _strtobool(os.environ.get("ORACLE_STEERING")) or _strtobool(os.environ.get("ORACLE_POLICY")):
    return OraclePolicy.from_env()

  alpha = float(os.environ.get("STEERING_ALPHA", os.environ.get("ACTIVATION_ALPHA", 0.0)))
  start_frame = int(os.environ.get("START_STEERING_FRAME", os.environ.get("ACTIVATION_START_FRAME", 0)))
  end_frame_raw = os.environ.get("END_STEERING_FRAME", os.environ.get("ACTIVATION_END_FRAME"))
  end_frame = None if end_frame_raw is None or end_frame_raw == "" else int(end_frame_raw)
  if alpha > 0.0:
    return FixedAfterFramePolicy(alpha=alpha, start_frame=start_frame, end_frame=end_frame)
  return ActivationPolicy()

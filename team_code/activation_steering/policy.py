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


class PDMOraclePolicy(OraclePolicy):
  """Conservative PDM-like oracle using only online CARLA actor/scenario state.

  This policy does not consume PDM-Lite labels or expert trajectories. It maps
  scenario-runner active_scenarios plus live actor geometry into the same action
  vector as OraclePolicy: [brake, left, right].

  The main difference from OraclePolicy is that two-way obstacle scenarios do not
  trigger lateral steering unless a simple opposite-lane clearance check passes.
  If the path is not clear, the policy stays inactive unless the obstacle is an
  imminent brake hazard.
  """

  ONE_WAY_LATERAL_SCENARIOS = frozenset({
      "Accident",
      "ConstructionObstacle",
      "ParkedObstacle",
  })
  TWO_WAY_LATERAL_SCENARIOS = frozenset({
      "AccidentTwoWays",
      "ConstructionObstacleTwoWays",
      "ParkedObstacleTwoWays",
      "VehicleOpensDoorTwoWays",
  })
  PRIORITY_BRAKE_SCENARIOS = frozenset({
      "OppositeVehicleTakingPriority",
      "OppositeVehicleRunningRedLight",
      "NormalVehicleTakingPriority",
      "NormalVehicleRunningRedLight",
  })

  def __init__(
      self,
      alpha: float,
      action: str = "auto",
      trigger_distance: float = 50.0,
      min_distance: float = 0.0,
      brake_hazard_distance: float = 20.0,
      brake_hazard_lateral_margin: float = 2.5,
      brake_reaction_time: float = 0.4,
      brake_deceleration: float = 6.0,
      brake_distance_margin: float = 2.0,
      brake_ttc_threshold: float = 2.0,
      brake_min_closing_speed: float = 0.5,
      hold_frames: int = 8,
      cooldown_frames: int = 20,
      allow_multi_action: bool = False,
      verbose: bool = False,
      two_way_clear_distance: float = 70.0,
      lane_key_search_distance: float = 90.0,
      side_hazard_distance: float = 25.0,
      side_hazard_two_way_distance: float = 10.0,
      roadblocked_distance: float = 40.0,
      priority_distance: float = 50.0,
      yield_emergency_distance: float = 50.0,
      general_brake: bool = False,
  ):
    super().__init__(
        alpha=alpha,
        action=action,
        trigger_distance=trigger_distance,
        min_distance=min_distance,
        brake_hazard_distance=brake_hazard_distance,
        brake_hazard_lateral_margin=brake_hazard_lateral_margin,
        brake_reaction_time=brake_reaction_time,
        brake_deceleration=brake_deceleration,
        brake_distance_margin=brake_distance_margin,
        brake_ttc_threshold=brake_ttc_threshold,
        brake_min_closing_speed=brake_min_closing_speed,
        hold_frames=hold_frames,
        cooldown_frames=cooldown_frames,
        allow_multi_action=allow_multi_action,
        verbose=verbose,
    )
    self.two_way_clear_distance = float(two_way_clear_distance)
    self.lane_key_search_distance = float(lane_key_search_distance)
    self.side_hazard_distance = float(side_hazard_distance)
    self.side_hazard_two_way_distance = float(side_hazard_two_way_distance)
    self.roadblocked_distance = float(roadblocked_distance)
    self.priority_distance = float(priority_distance)
    self.yield_emergency_distance = float(yield_emergency_distance)
    self.general_brake = bool(general_brake)

  @classmethod
  def from_env(cls) -> "PDMOraclePolicy":
    action = os.environ.get("PDM_ORACLE_ACTION", os.environ.get("ORACLE_ACTION", os.environ.get("ACTIVATION_ACTION", "auto")))
    alpha = float(
        os.environ.get("PDM_ORACLE_ALPHA",
                       os.environ.get("ORACLE_ALPHA", os.environ.get("STEERING_ALPHA", os.environ.get("ACTIVATION_ALPHA", 1.0)))))
    return cls(
        alpha=alpha,
        action=action,
        trigger_distance=float(os.environ.get("PDM_ORACLE_TRIGGER_DISTANCE", os.environ.get("ORACLE_TRIGGER_DISTANCE", 50.0))),
        min_distance=float(os.environ.get("PDM_ORACLE_MIN_DISTANCE", os.environ.get("ORACLE_MIN_DISTANCE", 0.0))),
        brake_hazard_distance=float(
            os.environ.get("PDM_ORACLE_BRAKE_HAZARD_DISTANCE", os.environ.get("ORACLE_BRAKE_HAZARD_DISTANCE", 20.0))),
        brake_hazard_lateral_margin=float(
            os.environ.get("PDM_ORACLE_BRAKE_HAZARD_LATERAL_MARGIN",
                           os.environ.get("ORACLE_BRAKE_HAZARD_LATERAL_MARGIN", 2.5))),
        brake_reaction_time=float(
            os.environ.get("PDM_ORACLE_BRAKE_REACTION_TIME", os.environ.get("ORACLE_BRAKE_REACTION_TIME", 0.4))),
        brake_deceleration=float(
            os.environ.get("PDM_ORACLE_BRAKE_DECELERATION", os.environ.get("ORACLE_BRAKE_DECELERATION", 6.0))),
        brake_distance_margin=float(
            os.environ.get("PDM_ORACLE_BRAKE_DISTANCE_MARGIN", os.environ.get("ORACLE_BRAKE_DISTANCE_MARGIN", 2.0))),
        brake_ttc_threshold=float(
            os.environ.get("PDM_ORACLE_BRAKE_TTC_THRESHOLD", os.environ.get("ORACLE_BRAKE_TTC_THRESHOLD", 2.0))),
        brake_min_closing_speed=float(
            os.environ.get("PDM_ORACLE_BRAKE_MIN_CLOSING_SPEED",
                           os.environ.get("ORACLE_BRAKE_MIN_CLOSING_SPEED", 0.5))),
        hold_frames=int(os.environ.get("PDM_ORACLE_HOLD_FRAMES", os.environ.get("ORACLE_HOLD_FRAMES", 8))),
        cooldown_frames=int(os.environ.get("PDM_ORACLE_COOLDOWN_FRAMES", os.environ.get("ORACLE_COOLDOWN_FRAMES", 20))),
        allow_multi_action=_strtobool(os.environ.get("PDM_ORACLE_ALLOW_MULTI_ACTION", os.environ.get("ORACLE_ALLOW_MULTI_ACTION"))),
        verbose=_strtobool(os.environ.get("PDM_ORACLE_VERBOSE", os.environ.get("ORACLE_VERBOSE"))),
        two_way_clear_distance=float(os.environ.get("PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE", 70.0)),
        lane_key_search_distance=float(os.environ.get("PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE", 90.0)),
        side_hazard_distance=float(os.environ.get("PDM_ORACLE_SIDE_HAZARD_DISTANCE", 25.0)),
        side_hazard_two_way_distance=float(os.environ.get("PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE", 10.0)),
        roadblocked_distance=float(os.environ.get("PDM_ORACLE_ROADBLOCKED_DISTANCE", 40.0)),
        priority_distance=float(os.environ.get("PDM_ORACLE_PRIORITY_DISTANCE", 50.0)),
        yield_emergency_distance=float(os.environ.get("PDM_ORACLE_YIELD_EMERGENCY_DISTANCE", 50.0)),
        general_brake=_strtobool(os.environ.get("PDM_ORACLE_GENERAL_BRAKE")),
    )

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
    scenarios = []
    for scenario_type, scenario_data in list(getattr(CarlaDataProvider, "active_scenarios", [])):
      actor = self._first_alive_actor(scenario_data)
      if actor is None:
        continue
      distance = self._horizontal_distance(ego_location, actor.get_location())
      scenarios.append((distance, scenario_type, scenario_data, actor))
    scenarios.sort(key=lambda item: item[0])

    triggers = []
    for distance, scenario_type, scenario_data, actor in scenarios:
      action = self._action_from_pdm_state(scenario_type, scenario_data, ego, actor, distance, CarlaDataProvider)
      if action is None or not self._action_allowed(action):
        continue
      if self.min_distance <= distance <= self._distance_limit_for_scenario(scenario_type):
        actor_id = getattr(actor, "id", None)
        triggers.append((action, scenario_type, actor_id, distance))

    if triggers or not self.general_brake:
      return triggers

    brake_actor = self._general_brake_actor(ego, CarlaDataProvider)
    if brake_actor is None:
      return triggers
    distance = self._horizontal_distance(ego_location, brake_actor.get_location())
    if self._action_allowed("brake"):
      triggers.append(("brake", "GeneralActorBrake", getattr(brake_actor, "id", None), distance))
    return triggers

  def _distance_limit_for_scenario(self, scenario_type: str) -> float:
    if scenario_type in ("HazardAtSideLane",):
      return self.side_hazard_distance
    if scenario_type in ("HazardAtSideLaneTwoWays",):
      return self.side_hazard_two_way_distance
    if scenario_type == "RoadBlocked":
      return self.roadblocked_distance
    if scenario_type in self.PRIORITY_BRAKE_SCENARIOS:
      return self.priority_distance
    if scenario_type == "YieldToEmergencyVehicle":
      return self.yield_emergency_distance
    return self.trigger_distance

  def _action_from_pdm_state(self, scenario_type, scenario_data, ego, actor, distance, data_provider) -> str | None:
    if scenario_type == "InvadingTurn":
      offset = self._invading_turn_offset(scenario_data)
      if offset is None:
        return None
      return "left" if offset > 0.0 else "right"

    if scenario_type in self.ONE_WAY_LATERAL_SCENARIOS:
      return self._lateral_action_from_direction(self._scenario_direction(scenario_data))

    if scenario_type in self.TWO_WAY_LATERAL_SCENARIOS:
      direction = self._scenario_direction(scenario_data)
      if self._two_way_path_clear(ego, scenario_data, direction, data_provider):
        return self._lateral_action_from_direction(direction)
      if self._scenario_actor_brake_hazard(ego, actor):
        return "brake"
      return None

    if scenario_type == "HazardAtSideLane":
      # PDM-Lite shifts around the bicycles as if the obstacle is on the right.
      return "left"

    if scenario_type == "HazardAtSideLaneTwoWays":
      if self._two_way_path_clear(ego, scenario_data, "right", data_provider):
        return "left"
      if self._scenario_actor_brake_hazard(ego, actor):
        return "brake"
      return None

    if scenario_type == "YieldToEmergencyVehicle":
      return self._yield_emergency_action(ego, data_provider)

    if scenario_type == "RoadBlocked":
      if distance <= self.roadblocked_distance:
        return "brake"
      return None

    if scenario_type in self.PRIORITY_BRAKE_SCENARIOS:
      if distance <= self.priority_distance and self._priority_actor_conflict(ego, actor, scenario_data):
        return "brake"
      return None

    if self._scenario_actor_brake_hazard(ego, actor):
      return "brake"
    return None

  @staticmethod
  def _lateral_action_from_direction(direction: str | None) -> str | None:
    if direction == "right":
      return "left"
    if direction == "left":
      return "right"
    return None

  def _two_way_path_clear(self, ego, scenario_data, direction: str | None, data_provider) -> bool:
    if direction not in ("left", "right"):
      return False

    world_map = self._get_map(data_provider)
    if world_map is None:
      return False

    try:
      ego_waypoint = world_map.get_waypoint(ego.get_location())
    except Exception:
      return False
    if ego_waypoint is None:
      return False

    target_lane = ego_waypoint.get_left_lane() if direction == "right" else ego_waypoint.get_right_lane()
    if target_lane is None:
      return False

    lane_keys = self._collect_lane_keys(target_lane, self.lane_key_search_distance)
    if not lane_keys:
      return False

    ignored_ids = {getattr(actor, "id", None) for actor in self._alive_actors(scenario_data)}
    ignored_ids.add(getattr(ego, "id", None))
    ego_location = ego.get_location()
    ego_forward = ego.get_transform().get_forward_vector()

    for vehicle in self._vehicle_actors(data_provider):
      if getattr(vehicle, "id", None) in ignored_ids or not getattr(vehicle, "is_alive", True):
        continue
      try:
        vehicle_waypoint = world_map.get_waypoint(vehicle.get_location())
      except Exception:
        continue
      if vehicle_waypoint is None:
        continue
      if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) not in lane_keys:
        continue

      vehicle_location = vehicle.get_location()
      diff = vehicle_location - ego_location
      longitudinal = diff.x * ego_forward.x + diff.y * ego_forward.y
      if -5.0 <= longitudinal <= self.two_way_clear_distance:
        return False
    return True

  def _collect_lane_keys(self, waypoint, distance: float) -> set[tuple[int, int]]:
    lane_keys = set()
    step = 2.0

    def add_chain(start, method_name):
      current = start
      traveled = 0.0
      while current is not None and traveled <= distance:
        lane_keys.add((current.road_id, current.lane_id))
        try:
          next_waypoints = getattr(current, method_name)(step)
        except Exception:
          break
        if not next_waypoints:
          break
        current = next_waypoints[0]
        traveled += step

    add_chain(waypoint, "next")
    add_chain(waypoint, "previous")
    return lane_keys

  def _yield_emergency_action(self, ego, data_provider) -> str | None:
    world_map = self._get_map(data_provider)
    if world_map is None:
      return None
    try:
      ego_waypoint = world_map.get_waypoint(ego.get_location())
    except Exception:
      return None
    if ego_waypoint is None:
      return None

    # Mirrors the PDM-Lite rule: shift left unless the current waypoint only
    # allows a right lane change. Avoid importing carla at module import time.
    try:
      import carla  # pylint: disable=import-outside-toplevel
      if ego_waypoint.lane_change != carla.LaneChange.Right:
        return "left"
      return "right"
    except Exception:
      left_lane = ego_waypoint.get_left_lane()
      return "left" if left_lane is not None else "right"

  def _priority_actor_conflict(self, ego, actor, scenario_data) -> bool:
    if actor is None or not getattr(actor, "is_alive", True):
      return False
    if actor.get_velocity().length() <= 0.1:
      return False

    ego_transform = ego.get_transform()
    actor_location = actor.get_location()
    ego_location = ego_transform.location
    diff_x = actor_location.x - ego_location.x
    diff_y = actor_location.y - ego_location.y
    right = ego_transform.get_right_vector()
    lateral = diff_x * right.x + diff_y * right.y
    actor_side = "right" if lateral > 0.0 else "left"
    direction = self._scenario_direction(scenario_data)
    return direction is None or direction == actor_side

  def _general_brake_actor(self, ego, data_provider):
    for actor in self._vehicle_actors(data_provider) + self._walker_actors(data_provider):
      if getattr(actor, "id", None) == getattr(ego, "id", None) or not getattr(actor, "is_alive", True):
        continue
      if self._scenario_actor_brake_hazard(ego, actor):
        return actor
    return None

  @staticmethod
  def _get_map(data_provider):
    try:
      return data_provider.get_map()
    except Exception:
      return None

  @staticmethod
  def _vehicle_actors(data_provider):
    try:
      world = data_provider.get_world()
      if world is not None:
        return list(world.get_actors().filter("vehicle.*"))
    except Exception:
      pass
    try:
      return [actor for _, actor in data_provider.get_actors() if getattr(actor, "type_id", "").startswith("vehicle.")]
    except Exception:
      return []

  @staticmethod
  def _walker_actors(data_provider):
    try:
      world = data_provider.get_world()
      if world is not None:
        return list(world.get_actors().filter("walker.*"))
    except Exception:
      pass
    try:
      return [actor for _, actor in data_provider.get_actors() if getattr(actor, "type_id", "").startswith("walker.")]
    except Exception:
      return []

  @staticmethod
  def _alive_actors(scenario_data):
    if scenario_data is None:
      return []
    return [item for item in scenario_data if hasattr(item, "get_location") and getattr(item, "is_alive", True)]


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
  if (policy_name in ("pdm_oracle", "pdm-oracle", "pdmoracle") or
      _strtobool(os.environ.get("PDM_ORACLE_STEERING")) or _strtobool(os.environ.get("PDM_ORACLE_POLICY"))):
    return PDMOraclePolicy.from_env()
  if policy_name == "oracle" or _strtobool(os.environ.get("ORACLE_STEERING")) or _strtobool(os.environ.get("ORACLE_POLICY")):
    return OraclePolicy.from_env()

  alpha = float(os.environ.get("STEERING_ALPHA", os.environ.get("ACTIVATION_ALPHA", 0.0)))
  start_frame = int(os.environ.get("START_STEERING_FRAME", os.environ.get("ACTIVATION_START_FRAME", 0)))
  end_frame_raw = os.environ.get("END_STEERING_FRAME", os.environ.get("ACTIVATION_END_FRAME"))
  end_frame = None if end_frame_raw is None or end_frame_raw == "" else int(end_frame_raw)
  if alpha > 0.0:
    return FixedAfterFramePolicy(alpha=alpha, start_frame=start_frame, end_frame=end_frame)
  return ActivationPolicy()

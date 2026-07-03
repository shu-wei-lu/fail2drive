import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      HandBrakeVehicle,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeRoadBehavior


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


class VehicleOnRoad(BasicScenario):
    """
    Spawn a stationary vehicle on the ego route at a configurable distance after the
    trigger point.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._distance = get_value_parameter(config, 'distance', float, 60)
        self._stop_time = get_value_parameter(config, 'stop_time', float, 20)
        self._vehicle_model = get_value_parameter(config, 'vehicle', str, 'vehicle.*')
        self._lateral_offset = get_value_parameter(config, 'lateral_offset', float, 0)
        self._z_offset = get_value_parameter(config, 'z_offset', float, 0.5)
        self._end_distance = get_value_parameter(config, 'end_distance', float, self._distance + 30)

        self._vehicle_transform = None

        super().__init__("VehicleOnRoad",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Spawn the vehicle ahead of the trigger point, then hide it below the map
        until the route trigger activates this scenario.
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        vehicle_wp = starting_wp.next(self._distance)[0]

        spawn_transform = carla.Transform(vehicle_wp.transform.location, vehicle_wp.transform.rotation)
        if self._lateral_offset:
            right_vector = spawn_transform.rotation.get_right_vector()
            spawn_transform.location += self._lateral_offset * right_vector
        spawn_transform.location.z += self._z_offset
        self._vehicle_transform = spawn_transform

        vehicle = CarlaDataProvider.request_new_actor(
            self._vehicle_model,
            spawn_transform,
            rolename='scenario no lights',
            attribute_filter={'base_type': 'car'}
        )
        if not vehicle:
            raise ValueError("Couldn't spawn the VehicleOnRoad actor")

        vehicle.set_simulate_physics(False)
        vehicle.set_location(spawn_transform.location + carla.Location(z=-200))
        vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(vehicle)

    def _create_behavior(self):
        """
        Reveal the vehicle at the configured route location, keep it stopped for a
        while or until the ego has driven past the obstacle, then destroy it.
        """
        root = py_trees.composites.Sequence(name="VehicleOnRoad")

        if self.route_mode:
            total_dist = self._distance + 50
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=total_dist))
            root.add_child(Idle(0.1))

        vehicle_behavior = py_trees.composites.Sequence(name="Stationary vehicle")
        vehicle_behavior.add_child(ActorTransformSetter(self.other_actors[0], self._vehicle_transform, True))
        vehicle_behavior.add_child(HandBrakeVehicle(self.other_actors[0], 1))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(Idle(self._stop_time))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        vehicle_behavior.add_child(end_condition)

        root.add_child(vehicle_behavior)

        if self.route_mode:
            root.add_child(ChangeRoadBehavior(extra_space=0))

        root.add_child(ActorDestroy(self.other_actors[0]))
        return root

    def _create_test_criteria(self):
        criteria = []
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria


class StaticPropOnRoad(BasicScenario):
    """
    Spawn a static prop on the ego route at a configurable distance after the
    trigger point.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._distance = get_value_parameter(config, 'distance', float, 60)
        self._stop_time = get_value_parameter(config, 'stop_time', float, 20)
        self._prop_model = get_value_parameter(config, 'prop', str, 'static.prop.trafficwarning')
        self._lateral_offset = get_value_parameter(config, 'lateral_offset', float, 0)
        self._z_offset = get_value_parameter(config, 'z_offset', float, 0.0)
        self._yaw_offset = get_value_parameter(config, 'yaw_offset', float, 0.0)
        self._end_distance = get_value_parameter(config, 'end_distance', float, self._distance + 30)

        self._prop_transform = None

        super().__init__("StaticPropOnRoad",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Spawn the prop ahead of the trigger point, then hide it below the map
        until the route trigger activates this scenario.
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        prop_wp = starting_wp.next(self._distance)[0]

        spawn_transform = carla.Transform(prop_wp.transform.location, prop_wp.transform.rotation)
        if self._lateral_offset:
            right_vector = spawn_transform.rotation.get_right_vector()
            spawn_transform.location += self._lateral_offset * right_vector
        spawn_transform.location.z += self._z_offset
        spawn_transform.rotation.yaw += self._yaw_offset
        self._prop_transform = spawn_transform

        prop = CarlaDataProvider.request_new_actor(
            self._prop_model,
            spawn_transform,
            rolename='scenario prop'
        )
        if not prop:
            raise ValueError("Couldn't spawn the StaticPropOnRoad actor")

        prop.set_simulate_physics(False)
        prop.set_location(spawn_transform.location + carla.Location(z=-200))
        self.other_actors.append(prop)

    def _create_behavior(self):
        """
        Reveal the prop at the configured route location, keep it there for a
        while or until the ego has driven past it, then destroy it.
        """
        root = py_trees.composites.Sequence(name="StaticPropOnRoad")

        if self.route_mode:
            total_dist = self._distance + 50
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=total_dist))
            root.add_child(Idle(0.1))

        prop_behavior = py_trees.composites.Sequence(name="Static prop")
        prop_behavior.add_child(ActorTransformSetter(self.other_actors[0], self._prop_transform, False))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(Idle(self._stop_time))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        prop_behavior.add_child(end_condition)

        root.add_child(prop_behavior)

        if self.route_mode:
            root.add_child(ChangeRoadBehavior(extra_space=0))

        root.add_child(ActorDestroy(self.other_actors[0]))
        return root

    def _create_test_criteria(self):
        criteria = []
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

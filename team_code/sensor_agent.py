"""
Agent file that runs the evaluations for all models supported by this repo.
Run it by giving it as the agent option to the
leaderboard/leaderboard/leaderboard_evaluator.py file
"""

import os
from copy import deepcopy

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
from nav_planner import extrapolate_waypoint_route

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve

from scenario_logger import ScenarioLogger
import transfuser_utils as t_u
from vlm_gate import AsyncVLMGate
from depth_ttc_gate import DepthTTCClient
from activation_steering.policy import policy_from_env

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'SensorAgent'


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


class SensorAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    """Sets up the agent. route_index is for logging purposes"""
    torch.cuda.empty_cache()
    self.IS_BENCH2DRIVE = strtobool(os.environ.get('IS_BENCH2DRIVE', 'False'))
    print('IS_BENCH2DRIVE: ', self.IS_BENCH2DRIVE)
    self.track = autonomous_agent.Track.MAP if os.environ.get(
        'CHALLENGE_TRACK_CODENAME') == 'MAP' else autonomous_agent.Track.SENSORS
    if self.IS_BENCH2DRIVE:
      self.config_path = path_to_conf_file.split('+')[0]
    else:
      self.config_path = path_to_conf_file

    self.step = -1
    self.initialized = False
    self.device = torch.device('cuda:0')

    # Load the config saved during training
    with open(os.path.join(self.config_path, 'config.json'), 'rt', encoding='utf-8') as f:
      json_config = f.read()

    loaded_config = jsonpickle.decode(json_config)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)

    # For models supporting different output modalities we select which one to use here.
    # 0: Waypoints
    # 1: Path + Target Speed

    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    self.tuned_aim_distance = int(os.environ.get('TUNED_AIM_DISTANCE', 0))
    print('TUNED_AIM_DISTANCE for wp rep?: ', self.tuned_aim_distance)
    direct = os.environ.get('DIRECT', 1)
    self.config.inference_direct_controller = int(direct)
    print('Direct control prediction?: ', direct)
    self.stop_after_meter = int(os.environ.get('STOP_AFTER_METER', -1))
    print('STOP_AFTER_METER: ', self.stop_after_meter)
    self.vlm_steering = strtobool(os.environ.get('VLM_STEERING', 'False'))
    self.depth_ttc_steering = self._depth_ttc_enabled_from_env()
    self.vlm_gate = None
    self.depth_ttc_client = None
    self.depth_ttc_last_decision = None
    self.depth_ttc_every_n = max(1, int(os.environ.get('DEPTH_TTC_EVERY_N', 1)))
    self.depth_ttc_hold_frames = max(1, int(os.environ.get('DEPTH_TTC_HOLD_FRAMES', 1)))
    self.depth_ttc_last_score_frame = None
    self.depth_ttc_active_until = -1
    self.depth_ttc_active_candidate = 'original'
    self.vlm_last_decision = None
    self.vlm_last_decision_key = None
    self.vlm_decision_start_frame = None
    self.vlm_history_steps = int(os.environ.get('VLM_ALPAMAYO_HISTORY_STEPS', 16))
    self.vlm_history_stride = int(os.environ.get('VLM_ALPAMAYO_HISTORY_STRIDE_FRAMES', 2))
    self.activation_policy = policy_from_env(vlm_enabled=self.vlm_steering and not self.depth_ttc_steering)
    print('Steering Policy: ', self.activation_policy.__class__.__name__)

    # If set to true, will generate visualizations at SAVE_PATH
    self.config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0)) == 1

    self.compile = int(os.environ.get('COMPILE', 0)) == 1

    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
    print('Brake uncertainty threshold: ', self.config.brake_uncertainty_threshold)

    # Classification networks are known to be overconfident which leads to them braking a bit too late in our case.
    # Reducing the driving speed slightly counteracts that.
    if int(os.environ.get('SLOWER', 0)):
      print(f'Reduce target speeds during evaluation by factor {self.config.slower_factor}.')
      self.inference_target_speeds = [self.config.slower_factor * speed for speed in self.config.target_speeds]
    else:
      print('No speed reduction during inference.')
      self.inference_target_speeds = self.config.target_speeds

    if self.config.tp_attention:
      self.tp_attention_buffer = []

    # Stop signs can be occluded with our camera setup. This buffer remembers them until cleared.
    # Very useful on the LAV benchmark
    self.stop_sign_controller = int(os.environ.get('STOP_CONTROL', 1))
    print('Use stop sign controller:', self.stop_sign_controller)
    if self.stop_sign_controller:
      # There can be max 1 stop sign affecting the ego
      self.stop_sign_buffer = deque(maxlen=1)
      self.clear_stop_sign = 0  # Counter if we recently cleared a stop sign

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(self.config_path):
      if file.endswith('.pth') and file.startswith('model'):
        self.model_count += 1
        print(os.path.join(self.config_path, file))
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(self.config_path, file), map_location=self.device)
        net.load_state_dict(state_dict, strict=True)
        net.cuda(device=self.device)
        net.eval()

        if self.config.compile or self.compile:
          net = torch.compile(net, mode=self.config.compile_mode)

        self.nets.append(net)

    if self.depth_ttc_steering:
      self.depth_ttc_client = DepthTTCClient.from_env()
      print(
          f'Depth TTC steering enabled: {self.depth_ttc_client.server_url} '
          f'horizon_s={self.depth_ttc_client.horizon_s} '
          f'every_n={self.depth_ttc_every_n} hold_frames={self.depth_ttc_hold_frames}')

    if self.vlm_steering and not self.depth_ttc_steering:
      self.vlm_gate = AsyncVLMGate.from_env()
      self.vlm_gate.start()

    self.stuck_detector = 0
    self.force_move = 0

    self.bb_buffer = deque(maxlen=1)
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.target_point_prev = [1e5, 1e5, 1e5]

    # Filtering
    self.ego_model = EgoModel(dt=self.config.carla_frame_rate)
    self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
    # Still uses the leaderboard 1.0 bicycle model for the unscented kalman filter
    self.ukf = UKF(dim_x=4,
                   dim_z=4,
                   fx=bicycle_model_forward,
                   hx=measurement_function_hx,
                   dt=self.config.carla_frame_rate,
                   points=self.points,
                   x_mean_fn=state_mean,
                   z_mean_fn=measurement_mean,
                   residual_x=residual_state_x,
                   residual_z=residual_measurement_h)

    # State noise, same as measurement because we
    # initialize with the first measurement later
    self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
    # Measurement noise
    self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
    self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
    # Used to set the filter state equal the first measurement
    self.filter_initialized = False
    # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
    self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))
    self.vlm_state_log = deque(maxlen=max((self.vlm_history_steps * self.vlm_history_stride) + 1, 2))

    #Temporal LiDAR
    self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)

    self.lidar_last = None

    # Forced stopping
    if self.stop_after_meter > 0:
      self.meters_travelled = 0

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    # Path to where visualizations and other debug output gets stored
    self.save_path = os.environ.get('SAVE_PATH', None)
    self.save_fused_features = strtobool(os.environ.get('SAVE_FUSED_FEATURES', 'False'))
    self.fused_features_save_path = None
    self.visual_output_path = None
    self.activation_action_log = None

    # Logger that generates logs used for infraction replay in the results_parser.
    if self.save_path is not None and route_index is not None:
      self.save_path = pathlib.Path(self.save_path) / str(route_index)
      pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
      self.activation_action_log = self.save_path / 'activation_actions.jsonl'

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )
    else:
      self.save_path = None

    visual_output_root = os.environ.get('VISUAL_OUTPUT_PATH', None)
    if visual_output_root is not None and route_index is not None:
      self.visual_output_path = pathlib.Path(visual_output_root) / str(route_index)
      self.visual_output_path.mkdir(parents=True, exist_ok=True)
    elif self.save_path is not None:
      self.visual_output_path = self.save_path

    fused_features_root = os.environ.get('FUSED_FEATURES_PATH', None)
    if self.save_fused_features:
      if fused_features_root is not None:
        self.fused_features_save_path = pathlib.Path(fused_features_root)
        if route_index is not None:
          self.fused_features_save_path = self.fused_features_save_path / str(route_index)
      elif self.save_path is not None:
        self.fused_features_save_path = self.save_path / 'fused_features'
      elif route_index is not None:
        self.fused_features_save_path = pathlib.Path('fused_features') / str(route_index)
      else:
        self.fused_features_save_path = pathlib.Path('fused_features')

      self.fused_features_save_path.mkdir(parents=True, exist_ok=True)
      print('Save fused features to:', self.fused_features_save_path)

    self.metric_info = {}
    self.live_visu = strtobool(os.environ.get('LIVE_VISU', 'False'))
    self._visu_interface = None
    self._quit_requested = False

  def _init(self):
    # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
    # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
    # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
    # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
    # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
    # of the GPS. It seems to work reasonably well, so we use this workaround for now.
    try:
      locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
      lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
      earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

      def equations(variables):
        x, y = variables
        eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa) -
               math.cos(x * math.pi / 180.0) * y)
        eq2 = (math.log(math.tan(
            (lat + 90.0) * math.pi / 360.0)) * earth_radius_equa * math.cos(x * math.pi / 180.0) + locy -
               math.cos(x * math.pi / 180.0) * earth_radius_equa * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
        return [eq1, eq2]

      initial_guess = [0.0, 0.0]
      solution = fsolve(equations, initial_guess)
      self.lat_ref, self.lon_ref = solution[0], solution[1]
    except Exception as e:
      print(e, flush=True)
      self.lat_ref, self.lon_ref = 0.0, 0.0

    # During setup() not everything is available yet, so this _init is a second setup in run_step()
    if self.save_path is not None:
      # Privileged map access for logging and visualizations. Turned off during normal evaluation.
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
      from nav_planner import interpolate_trajectory  # pylint: disable=locally-disabled, import-outside-toplevel
      self.world_map = CarlaDataProvider.get_map()
      trajectory = [item[0].location for item in self._global_plan_world_coord]
      self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)  # privileged

      self._waypoint_planner = RoutePlanner(self.config.log_route_planner_min_distance,
                                            self.config.route_planner_max_distance, self.lat_ref, self.lon_ref)
      self._waypoint_planner.set_route(self.dense_route, True)

      vehicle = CarlaDataProvider.get_hero_actor()
      self.lon_logger.ego_vehicle = vehicle
      self.lon_logger.world = vehicle.get_world()

      self.nets[0].init_visualization()

    self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance,
                                       self.lat_ref, self.lon_ref)
    self._route_planner.set_route(self._global_plan, True)
    self.initialized = True

  def sensors(self):
    sensors = [{
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_front'
    }, {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': self.config.carla_frame_rate,
        'id': 'imu'
    }, {
        'type': 'sensor.other.gnss',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': 0.01,
        'id': 'gps'
    }, {
        'type': 'sensor.speedometer',
        'reading_frequency': self.config.carla_fps,
        'id': 'speed'
    }]
    # Don't set up LiDAR for camera only approaches
    if self.config.backbone not in ('aim'):
      sensors.append({
          'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'id': 'lidar'
      })

    return sensors

  @torch.inference_mode()  # Turns off gradient computation
  def tick(self, input_data):
    """Pre-processes sensor data and runs the Unscented Kalman Filter"""
    rgb = []
    for camera_pos in ['front']:
      rgb_cam = 'rgb_' + camera_pos
      camera = input_data[rgb_cam][1][:, :, :3]

      # Also add jpg artifacts at test time, because the training data was saved as jpg.
      _, compressed_image_i = cv2.imencode('.jpg', camera)
      camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      rgb_pos = t_u.crop_array(self.config, rgb_pos)
      rgb_front_np = rgb_pos.copy()

      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)
    rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

    gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    result = {
        'rgb': rgb,
        'compass': compass,
        'rgb_front_np': rgb_front_np,
    }

    if self.config.backbone not in ('aim'):
      result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    if not self.filter_initialized:
      # apply ukf only to x and y coordinates, append z coordinate afterwards
      self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
      self.filter_initialized = True

    self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
    filtered_state = self.ukf.x
    self.state_log.append(filtered_state)
    self.vlm_state_log.append(filtered_state.copy())
    result['gps'] = filtered_state[0:2]

    waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))

    if len(waypoint_route) > 2:
      target_point, far_command = waypoint_route[1]
      target_point_next, _ = waypoint_route[2]
    elif len(waypoint_route) > 1:
      target_point, far_command = waypoint_route[1]
      target_point_next = target_point
    else:
      target_point, far_command = waypoint_route[0]
      target_point_next = target_point

    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)

    one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

    ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])  # original

    ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    result['target_point'] = ego_target_point

    if self.config.two_tp_input:
      ego_target_point_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
      ego_target_point_next = torch.from_numpy(ego_target_point_next[np.newaxis]).to(self.device, dtype=torch.float32)
      result['target_point_next'] = ego_target_point_next

    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

    if self.save_path is not None:
      pass
      waypoint_route = self._waypoint_planner.run_step(np.append(result['gps'], gps_pos[2]))
      waypoint_route = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
      route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.config.route_points]
      self.lon_logger.log_step(route)

    return result

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1

    if not self.initialized:
      self._init()
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.control = control
      tick_data = self.tick(input_data)
      if self.config.backbone not in ('aim'):
        self.lidar_last = deepcopy(tick_data['lidar'])
      return control

    # Need to run this every step for GPS filtering
    tick_data = self.tick(input_data)

    lidar_indices = []
    for i in range(self.config.lidar_seq_len):
      lidar_indices.append(i * self.config.data_save_freq)

    #Current position of the car
    ego_x = self.state_log[-1][0]
    ego_y = self.state_log[-1][1]
    ego_theta = self.state_log[-1][2]

    ego_x_last = self.state_log[-2][0]
    ego_y_last = self.state_log[-2][1]
    ego_theta_last = self.state_log[-2][2]

    # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
    if self.config.backbone not in ('aim'):
      lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    # Updates stop boxes by vehicle movement converting past predictions into the current frame.
    if self.stop_sign_controller:
      self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    if self.config.backbone not in ('aim'):
      lidar_current = deepcopy(tick_data['lidar'])
      lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

      self.lidar_buffer.append(lidar_full)

    if self.config.backbone not in ('aim'):
      # We wait until we have sufficient LiDARs
      if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
        self.lidar_last = deepcopy(tick_data['lidar'])
        tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
        self.control = tmp_control

        return tmp_control

    if self.config.backbone in ('aim'):  # Image only method
      # Dummy data
      lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height,
                               self.config.lidar_resolution_width)).to(self.device, dtype=torch.float32)
    else:
      # Voxelize LiDAR and stack temporal frames
      lidar_bev = []
      # prepare LiDAR input
      for i in lidar_indices:
        lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])

        # For single frame there is no point in realignment. The state_log index will also differ.
        if self.config.realign_lidar and self.config.lidar_seq_len > 1:
          # Position of the car when the LiDAR was collected
          curr_x = self.state_log[i][0]
          curr_y = self.state_log[i][1]
          curr_theta = self.state_log[i][2]

          # Voxelize to BEV for NN to process
          lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

        lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                                use_ground_plane=self.config.use_ground_plane)

        lidar_histogram = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
        lidar_bev.append(lidar_histogram)

        lidar_bev = torch.cat(lidar_bev, dim=1)

    if self.config.backbone not in ('aim'):
      self.lidar_last = deepcopy(tick_data['lidar'])

    # prepare velocity input
    gt_velocity = tick_data['speed']
    velocity = gt_velocity.reshape(1, 1)  # used by transfuser

    compute_debug_output = self.live_visu or (self.config.debug and (self.save_path is not None))

    # new checkpoint lookahead: calculate which checkpoint to use for control
    speed = gt_velocity.item()

    if self.stop_after_meter > 0:
      dt = self.config.carla_frame_rate
      self.meters_travelled = self.meters_travelled + speed * dt

    steering_alpha = 0.0
    if self.vlm_gate is not None:
      self.vlm_last_decision = self.vlm_gate.latest()
      decision_age_frames = None
      if self.vlm_last_decision is not None:
        decision_key = (self.vlm_last_decision.frame_id, self.vlm_last_decision.timestamp)
        if decision_key != self.vlm_last_decision_key:
          self.vlm_last_decision_key = decision_key
          self.vlm_decision_start_frame = self.step
        decision_age_frames = self.step - self.vlm_decision_start_frame
      steering_alpha = self.activation_policy.alpha(
          self.step,
          self.vlm_last_decision,
          decision_age_frames=decision_age_frames)
      if self.vlm_last_decision is not None:
        if self.vlm_gate.verbose and self._activation_alpha_active(steering_alpha):
          print(
              f"[VLMGate] use step={self.step} decision_frame={self.vlm_last_decision.frame_id} "
              f"age={decision_age_frames} action={getattr(self.vlm_last_decision, 'action', 'none')} "
              f"reason={self.vlm_last_decision.reason}",
              flush=True)
      elif self.vlm_gate.verbose:
        print(f"[VLMGate] use step={self.step} no completed decision yet", flush=True)
    else:
      steering_alpha = self.activation_policy.alpha(self.step)

    # forward pass
    feature_frame_idx = self.step
    forward_candidates = [('original', steering_alpha)]
    depth_ttc_should_score = False
    if self.depth_ttc_client is not None:
      depth_ttc_should_score = self._depth_ttc_should_score(speed)
      if depth_ttc_should_score:
        forward_candidates = self._depth_ttc_candidate_alphas()
      else:
        held_candidate = self._depth_ttc_held_candidate()
        forward_candidates = [(held_candidate, self._depth_ttc_alpha_for_candidate(held_candidate))]

    candidate_results = {}
    for candidate_name, candidate_alpha in forward_candidates:
      pred_wps = []
      pred_target_speeds = []
      pred_checkpoints = []
      bounding_boxes = []
      wp_selected = None
      for i in range(self.model_count):
        if self.config.backbone in ('transFuser', 'aim', 'bev_encoder'):
          model_output = self.nets[i].forward(
            rgb=tick_data['rgb'],
            lidar_bev=lidar_bev,
            target_point=tick_data['target_point'],
            target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
            ego_vel=velocity,
            command=tick_data['command'],
            steering_alpha=candidate_alpha,
            return_fused_features=self.save_fused_features and candidate_name == 'original')
          if self.save_fused_features and candidate_name == 'original':
            pred_wp, \
            pred_target_speed, \
            pred_checkpoint, \
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bb_features,\
            attention_weights,\
            pred_wp_1,\
            selected_path, \
            fused_features = model_output
            self._save_fused_features(fused_features, i, feature_frame_idx)
          else:
            pred_wp, \
            pred_target_speed, \
            pred_checkpoint, \
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bb_features,\
            attention_weights,\
            pred_wp_1,\
            selected_path = model_output
          # Only convert bounding boxes when they are used.
          if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                           self.stop_sign_controller):
            pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
          else:
            pred_bounding_box = None
        else:
          raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

        if self.config.use_wp_gru:
          if self.config.multi_wp_output:
            wp_selected = 0
            if F.sigmoid(selected_path)[0].item() > 0.5:
              wp_selected = 1
              pred_wps.append(pred_wp_1)
            else:
              pred_wps.append(pred_wp)
          else:
            pred_wps.append(pred_wp)
        if self.config.use_controller_input_prediction:
          pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
          pred_checkpoints.append(pred_checkpoint[0])

        bounding_boxes.append(pred_bounding_box)

      pred_wp_ensemble = torch.stack(pred_wps, dim=0).mean(dim=0) if pred_wps else None
      pred_checkpoint_ensemble = torch.stack(pred_checkpoints, dim=0).mean(dim=0) if pred_checkpoints else None
      pred_target_speed_ensemble = torch.stack(pred_target_speeds, dim=0).mean(dim=0) if pred_target_speeds else None
      if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                       self.stop_sign_controller):
        bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)
      else:
        bbs_vehicle_coordinate_system = None

      candidate_results[candidate_name] = {
          'steering_alpha': candidate_alpha,
          'pred_wps': pred_wps,
          'pred_target_speeds': pred_target_speeds,
          'pred_checkpoints': pred_checkpoints,
          'pred_wp_ensemble': pred_wp_ensemble,
          'pred_checkpoint_ensemble': pred_checkpoint_ensemble,
          'pred_target_speed_ensemble': pred_target_speed_ensemble,
          'bbs_vehicle_coordinate_system': bbs_vehicle_coordinate_system,
          'wp_selected': wp_selected,
          'pred_wp': pred_wp,
          'pred_target_speed': pred_target_speed,
          'pred_checkpoint': pred_checkpoint,
          'pred_semantic': pred_semantic,
          'pred_bev_semantic': pred_bev_semantic,
          'pred_depth': pred_depth,
          'attention_weights': attention_weights,
          'pred_wp_1': pred_wp_1,
      }

    selected_candidate = forward_candidates[0][0]
    if self.depth_ttc_client is not None and depth_ttc_should_score and len(candidate_results) > 1:
      planner_candidates = self._depth_ttc_planner_candidates_from_results(candidate_results)
      if planner_candidates:
        self.depth_ttc_last_score_frame = self.step
        self.depth_ttc_last_decision = self.depth_ttc_client.score(
            frame_id=self.step,
            rgb_image=tick_data['rgb_front_np'],
            speed=speed,
            camera=self._depth_ttc_camera_payload(tick_data['rgb_front_np']),
            ego_extent={'x': float(self.config.ego_extent_x), 'y': float(self.config.ego_extent_y)},
            planner_candidates={
                'candidates': planner_candidates,
                'target_speeds': [float(value) for value in self.inference_target_speeds],
                'uncertainty_weight': int(self.uncertainty_weight),
                'brake_uncertainty_threshold': float(self.config.brake_uncertainty_threshold),
            })
        selected_candidate = getattr(self.depth_ttc_last_decision, 'selected_trajectory', 'original')
        if selected_candidate not in candidate_results:
          selected_candidate = 'original'
        self._depth_ttc_activate_candidate(selected_candidate)
        if self.depth_ttc_client.verbose:
          print(
              f"[DepthTTC] use step={self.step} selected={selected_candidate} "
              f"action={getattr(self.depth_ttc_last_decision, 'action', 'none')} "
              f"active_until={self.depth_ttc_active_until}",
              flush=True)

    selected_result = candidate_results[selected_candidate]
    steering_alpha = selected_result['steering_alpha']
    pred_wps = selected_result['pred_wps']
    pred_target_speeds = selected_result['pred_target_speeds']
    pred_checkpoints = selected_result['pred_checkpoints']
    bbs_vehicle_coordinate_system = selected_result['bbs_vehicle_coordinate_system']
    wp_selected = selected_result['wp_selected']
    pred_wp = selected_result['pred_wp']
    pred_target_speed = selected_result['pred_target_speed']
    pred_checkpoint = selected_result['pred_checkpoint']
    pred_semantic = selected_result['pred_semantic']
    pred_bev_semantic = selected_result['pred_bev_semantic']
    pred_depth = selected_result['pred_depth']
    attention_weights = selected_result['attention_weights']
    pred_wp_1 = selected_result['pred_wp_1']

    if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                     self.stop_sign_controller):
      self.bb_buffer.append(bbs_vehicle_coordinate_system)

    stop_for_stop_sign = False
    if self.stop_sign_controller:
      stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.tp_attention:
      self.tp_attention_buffer.append(attention_weights[2])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    if self.vlm_gate is not None:
      self.vlm_gate.observe(self.step, tick_data['rgb_front_np'])
      if speed > 3.5:
        self.vlm_gate.submit(
            frame_id=self.step,
            rgb_image=tick_data['rgb_front_np'],
            speed=speed,
            command=self.commands[-2],
            target_point=tick_data['target_point'],
            ego_history_xyz=self._vlm_ego_history_xyz(),
            ego_history_rot=self._vlm_ego_history_rot())

    # calculate target speed scalar from model predictions
    if self.config.use_controller_input_prediction:
      pred_target_speed_ensemble = torch.stack(pred_target_speeds,
                                               dim=0).mean(dim=0)  # average across ensemble models' prediction

      if self.uncertainty_weight:
        uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed_scalar = self.inference_target_speeds[0]
        else:
          pred_target_speed_scalar = sum(uncertainty * self.inference_target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed_ensemble)
        pred_target_speed_scalar = self.inference_target_speeds[pred_target_speed_index]

    # Visualize the output of the last model
    if compute_debug_output:
      if self.config.use_controller_input_prediction:
        prob_target_speed = F.softmax(pred_target_speed, dim=1)
      else:
        prob_target_speed = pred_target_speed

      debug_image = self.nets[0].visualize_model(
          self.visual_output_path if self.visual_output_path is not None else '.',
          self.step,
          tick_data['rgb'],
          lidar_bev,
          tick_data['target_point'],
          pred_wp,
          target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
          pred_semantic=pred_semantic,
          pred_bev_semantic=pred_bev_semantic,
          pred_depth=pred_depth,
          pred_checkpoint=pred_checkpoint,
          pred_speed=prob_target_speed,
          pred_target_speed_scalar=pred_target_speed_scalar,
          pred_bb=bbs_vehicle_coordinate_system,
          gt_speed=gt_velocity,
          gt_wp=pred_wp_1,
          wp_selected=wp_selected,
          return_image=self.live_visu,
          save_to_disk=(self.visual_output_path is not None))

      if self.live_visu and debug_image is not None:
        self._display_debug_output(debug_image)

    if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
      pred_checkpoints = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)
    elif self.config.use_wp_gru and not self.config.inference_direct_controller:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp,
                                                        gt_velocity,
                                                        tuned_aim_distance=bool(self.tuned_aim_distance))
    else:
      raise ValueError('An output representation was chosen that was not trained.')

    # 0.1 is just an arbitrary low number to threshold when the car is stopped
    if gt_velocity < 0.1:
      self.stuck_detector += 1
    else:
      self.stuck_detector = 0

    # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
    if self.stuck_detector > self.config.stuck_threshold:
      self.force_move = self.config.creep_duration

    if self.force_move > 0:
      emergency_stop = False
      if self.config.backbone not in ('aim'):
        # safety check
        safety_box = deepcopy(self.lidar_buffer[-1])

        # z-axis
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

        # y-axis
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

        # x-axis
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

      if not emergency_stop:
        print('Detected agent being stuck. Step: ', self.step)
        throttle = max(self.config.creep_throttle, throttle)
        brake = False
        self.force_move -= 1
      else:
        print('Creeping stopped by safety box. Step: ', self.step)
        throttle = 0.0
        brake = True
        self.force_move = self.config.creep_duration

    if self.stop_sign_controller:
      if stop_for_stop_sign:
        throttle = 0.0
        brake = True

    if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
      print(f'Stopping after {self.stop_after_meter} meters.')
      throttle = 0.0
      brake = True

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
    self._log_activation_action(
        frame_idx=feature_frame_idx,
        control=control,
        speed=float(gt_velocity.item() if hasattr(gt_velocity, 'item') else gt_velocity),
        pred_target_speed=float(pred_target_speed_scalar),
        stop_for_stop_sign=bool(stop_for_stop_sign),
        activation_alpha=steering_alpha)

    if self.IS_BENCH2DRIVE:
      # TODO doesn't seem to work
      metric_info = self.get_metric_info()
      self.metric_info[self.step] = metric_info
      if self.save_path is not None and self.step % 1 == 0:
        with open(self.save_path / 'metric_info.json', 'w') as outfile:
          ujson.dump(self.metric_info, outfile, indent=4)

    # CARLA will not let the car drive in the initial frames.
    # We set the action to brake so that the filter does not get confused.
    if self.step < self.config.inital_frames_delay:
      self.control = carla.VehicleControl(0.0, 0.0, 1.0)
    else:
      self.control = control

    return control

  def _draw_vlm_trajectory_overlay(self, rgb_image, trajectory):
    image = np.ascontiguousarray(rgb_image.copy())
    corridor_segments = self._build_vlm_trajectory_corridor_segments(trajectory)
    if not corridor_segments:
      return image

    opacity = 0.25
    overlay = image.copy()

    for t_s, polygon_ego in corridor_segments:
      projected = self._project_ego_points_to_front_image(polygon_ego)
      if projected is None:
        continue
      polygon_xy, _ = projected
      if len(polygon_xy) != 4:
        continue
      polygon_xy = polygon_xy.astype(np.int32).reshape(-1, 1, 2)
      color = self._vlm_trajectory_color(t_s)
      cv2.fillPoly(overlay, [polygon_xy], color, lineType=cv2.LINE_AA)
      cv2.polylines(overlay, [polygon_xy], True, color, 1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0)

  def _build_vlm_trajectory_corridor_segments(self, trajectory):
    if trajectory is None:
      return []

    centers = np.asarray(trajectory, dtype=np.float32)
    if centers.ndim == 3:
      centers = centers[0]
    if centers.ndim != 2 or centers.shape[1] < 2 or len(centers) < 2:
      return []

    centers = centers[:, :2]
    half_width = float(self.config.ego_extent_y)
    horizon_s = 4.0
    num_points = len(centers)
    segments = []

    for idx in range(num_points - 1):
      p0 = centers[idx]
      p1 = centers[idx + 1]
      tangent = p1 - p0
      norm = float(np.linalg.norm(tangent))
      if norm < 1e-4:
        continue
      tangent = tangent / norm
      right_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
      polygon = np.asarray([
          p0 - right_normal * half_width,
          p1 - right_normal * half_width,
          p1 + right_normal * half_width,
          p0 + right_normal * half_width,
      ], dtype=np.float32)
      t_s = (float(idx) + 1.5) / float(num_points) * horizon_s
      segments.append((t_s, polygon))

    return segments

  @staticmethod
  def _vlm_trajectory_color(t_s):
    if t_s <= 1.5:
      return (180, 0, 0)
    if t_s <= 3.0:
      return (255, 128, 0)
    return (255, 230, 0)

  @staticmethod
  def _depth_ttc_enabled_from_env():
    policy_name = os.environ.get('ACTIVATION_POLICY', os.environ.get('STEERING_POLICY', '')).lower()
    return (
        strtobool(os.environ.get('DEPTH_TTC_STEERING', 'False')) or
        policy_name in ('depth_ttc', 'depth-ttc', 'depthttc'))

  @staticmethod
  def _depth_ttc_candidate_alphas():
    brake_alpha = float(os.environ.get('DEPTH_TTC_BRAKE_ALPHA', 3.0))
    lateral_alpha = float(os.environ.get('DEPTH_TTC_LATERAL_ALPHA', 1.0))
    return [
        ('original', [0.0, 0.0, 0.0]),
        ('brake', [brake_alpha, 0.0, 0.0]),
        ('left', [0.0, lateral_alpha, 0.0]),
        ('right', [0.0, 0.0, lateral_alpha]),
    ]

  def _depth_ttc_alpha_for_candidate(self, candidate):
    candidate = str(candidate or 'original').lower()
    for name, alpha in self._depth_ttc_candidate_alphas():
      if candidate == name:
        return alpha
    return self._depth_ttc_candidate_alphas()[0][1]

  def _depth_ttc_should_score(self, speed):
    min_speed = float(os.environ.get('DEPTH_TTC_MIN_SPEED_M_S', 0.0))
    if float(speed) < min_speed:
      return False
    if self.depth_ttc_last_score_frame is None:
      return True
    return (self.step - self.depth_ttc_last_score_frame) >= self.depth_ttc_every_n

  def _depth_ttc_held_candidate(self):
    if self.step <= self.depth_ttc_active_until:
      candidate = str(self.depth_ttc_active_candidate or 'original').lower()
      if candidate in ('brake', 'left', 'right'):
        return candidate
    return 'original'

  def _depth_ttc_activate_candidate(self, candidate):
    candidate = str(candidate or 'original').lower()
    if candidate not in ('brake', 'left', 'right'):
      self.depth_ttc_active_candidate = 'original'
      self.depth_ttc_active_until = self.step
      return
    self.depth_ttc_active_candidate = candidate
    self.depth_ttc_active_until = self.step + self.depth_ttc_hold_frames - 1

  @staticmethod
  def _depth_ttc_planner_candidates_from_results(candidate_results):
    planner_candidates = {}
    for name, result in candidate_results.items():
      checkpoints = result.get('pred_checkpoint_ensemble')
      target_speed_probs = result.get('pred_target_speed_ensemble')
      if checkpoints is None or target_speed_probs is None:
        continue
      if hasattr(checkpoints, 'detach'):
        checkpoints = checkpoints.detach().cpu().numpy()
      if hasattr(target_speed_probs, 'detach'):
        target_speed_probs = target_speed_probs.detach().cpu().numpy()
      checkpoints = np.asarray(checkpoints, dtype=np.float32)
      if checkpoints.ndim == 3:
        checkpoints = checkpoints[0]
      target_speed_probs = np.asarray(target_speed_probs, dtype=np.float32).reshape(-1)
      if checkpoints.ndim != 2 or checkpoints.shape[1] < 2 or len(target_speed_probs) == 0:
        continue
      planner_candidates[name] = {
          'checkpoints': checkpoints[:, :2].tolist(),
          'target_speed_probs': target_speed_probs.tolist(),
      }
    return planner_candidates

  def _depth_ttc_camera_payload(self, image):
    image_height, image_width = np.asarray(image).shape[:2]
    intrinsic = t_u.calculate_intrinsic_matrix(
        fov=self.config.camera_fov,
        height=self.config.camera_height,
        width=self.config.camera_width)
    base_width = self.config.camera_width
    base_height = self.config.camera_height
    if self.config.crop_image:
      side_crop = (self.config.camera_width - self.config.cropped_width) // 2
      intrinsic[0, 2] -= side_crop
      base_width = self.config.cropped_width
      base_height = self.config.cropped_height

    scale_x = float(image_width) / float(base_width)
    scale_y = float(image_height) / float(base_height)
    return {
        'fx': float(intrinsic[0, 0] * scale_x),
        'fy': float(intrinsic[1, 1] * scale_y),
        'cx': float(intrinsic[0, 2] * scale_x),
        'cy': float(intrinsic[1, 2] * scale_y),
        'fov': float(self.config.camera_fov),
        'position': [float(value) for value in self.config.camera_pos],
    }

  def _vlm_ego_history_states(self):
    history_log = getattr(self, 'vlm_state_log', self.state_log)
    if len(history_log) == 0:
      return None

    states = list(history_log)
    last_index = len(states) - 1
    indices = [
        max(0, last_index - self.vlm_history_stride * offset)
        for offset in reversed(range(self.vlm_history_steps))
    ]
    return np.asarray([states[index][:3] for index in indices], dtype=np.float32)

  @staticmethod
  def _yaw_rotation_matrix(yaw):
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.asarray([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

  def _vlm_ego_history_xyz(self):
    states = self._vlm_ego_history_states()
    if states is None:
      return None

    t0_xy = states[-1, :2].copy()
    t0_yaw = float(states[-1, 2])
    c = math.cos(t0_yaw)
    s = math.sin(t0_yaw)
    delta = states[:, :2] - t0_xy
    local_xy = np.stack([
        c * delta[:, 0] + s * delta[:, 1],
        -s * delta[:, 0] + c * delta[:, 1],
    ], axis=1)
    xyz = np.zeros((self.vlm_history_steps, 3), dtype=np.float32)
    xyz[:, :2] = local_xy.astype(np.float32)
    return xyz.reshape(1, 1, self.vlm_history_steps, 3).tolist()

  def _vlm_ego_history_rot(self):
    states = self._vlm_ego_history_states()
    if states is None:
      return None

    current_inv = self._yaw_rotation_matrix(float(states[-1, 2])).T
    rotations = []
    for yaw in states[:, 2]:
      rotations.append(current_inv @ self._yaw_rotation_matrix(float(yaw)))
    rot = np.stack(rotations, axis=0).astype(np.float32)
    return rot.reshape(1, 1, self.vlm_history_steps, 3, 3).tolist()

  def _resample_vlm_trajectory_by_time(self, trajectory, speed):
    if trajectory is None:
      return None

    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim == 3:
      points = points[0]
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
      return None

    points = points[:, :2]
    num_points = 16
    horizon_s = 4.0
    speed = max(float(speed), 0.0)
    target_distances = np.linspace(horizon_s / num_points, horizon_s, num_points, dtype=np.float32) * speed

    path = np.vstack((np.zeros((1, 2), dtype=np.float32), points))
    segment_vectors = path[1:] - path[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if cumulative[-1] <= 1e-4:
      return np.zeros((num_points, 2), dtype=np.float32)

    sampled = []
    for distance in target_distances:
      if distance <= cumulative[-1]:
        segment_idx = np.searchsorted(cumulative, distance, side='right') - 1
        segment_idx = int(np.clip(segment_idx, 0, len(segment_lengths) - 1))
        denom = max(segment_lengths[segment_idx], 1e-4)
        ratio = (distance - cumulative[segment_idx]) / denom
        sampled.append(path[segment_idx] + ratio * segment_vectors[segment_idx])
      else:
        direction = segment_vectors[-1] / max(segment_lengths[-1], 1e-4)
        sampled.append(path[-1] + direction * (distance - cumulative[-1]))

    return np.asarray(sampled, dtype=np.float32)

  def _project_ego_points_to_front_image(self, trajectory):
    if trajectory is None:
      return None

    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim == 3:
      points = points[0]
    if points.ndim != 2 or points.shape[1] < 2:
      return None

    points = points[:, :2]
    if len(points) == 0:
      return None

    z = float(os.environ.get('VLM_TRAJECTORY_Z_M', 0.0))
    points_3d = np.concatenate((points, np.full((len(points), 1), z, dtype=np.float32)), axis=1)

    camera_pos = np.asarray(self.config.camera_pos, dtype=np.float32)
    camera_points = points_3d - camera_pos.reshape(1, 3)
    # Ego/CARLA uses z-up; pinhole image coordinates use y-down.
    pinhole_points = np.stack((camera_points[:, 1], -camera_points[:, 2], camera_points[:, 0]), axis=0)
    intrinsic = t_u.calculate_intrinsic_matrix(
        fov=self.config.camera_fov,
        height=self.config.camera_height,
        width=self.config.camera_width)
    projected = intrinsic @ pinhole_points
    depth = projected[2]
    valid = depth > 0.1
    projected_xy = projected[:2].T
    projected_xy[valid] = projected_xy[valid] / depth[valid, None]

    if self.config.crop_image:
      side_crop = (self.config.camera_width - self.config.cropped_width) // 2
      projected_xy[:, 0] -= side_crop
      image_width = self.config.cropped_width
      image_height = self.config.cropped_height
    else:
      image_width = self.config.camera_width
      image_height = self.config.camera_height

    valid = valid & (projected_xy[:, 0] >= 0.0) & (projected_xy[:, 0] < image_width)
    valid = valid & (projected_xy[:, 1] >= 0.0) & (projected_xy[:, 1] < image_height)
    projected_xy = projected_xy[valid]
    if len(projected_xy) == 0:
      return None

    return projected_xy, np.nonzero(valid)[0]

  def _save_fused_features(self, fused_features, model_idx, frame_idx):
    if self.fused_features_save_path is None or fused_features is None:
      return

    save_path = self.fused_features_save_path
    if self.model_count > 1:
      save_path = save_path / f'model_{model_idx:02d}'
      save_path.mkdir(parents=True, exist_ok=True)

    torch.save(fused_features.detach().cpu(), save_path / f'{int(frame_idx):06d}.pt')

  def _log_activation_action(self,
                             frame_idx,
                             control,
                             speed,
                             pred_target_speed,
                             stop_for_stop_sign=False,
                             activation_alpha=None):
    if self.activation_action_log is None:
      return

    feature_path = None
    if self.fused_features_save_path is not None:
      feature_dir = self.fused_features_save_path
      if self.model_count > 1:
        feature_dir = feature_dir / 'model_00'
      feature_path = str(feature_dir / f'{int(frame_idx):06d}.pt')

    record = {
        'frame': int(frame_idx),
        'speed': float(speed),
        'pred_target_speed': float(pred_target_speed),
        'steer': float(control.steer),
        'throttle': float(control.throttle),
        'brake': float(control.brake),
        'stop_for_stop_sign': bool(stop_for_stop_sign),
        'activation_alpha': self._json_activation_alpha(activation_alpha),
        'feature_path': feature_path,
    }
    with open(self.activation_action_log, 'a', encoding='utf-8') as f:
      f.write(ujson.dumps(record) + '\n')

  @staticmethod
  def _json_activation_alpha(activation_alpha):
    if activation_alpha is None:
      return None
    if hasattr(activation_alpha, 'detach'):
      activation_alpha = activation_alpha.detach().cpu().flatten().tolist()
    elif isinstance(activation_alpha, np.ndarray):
      activation_alpha = activation_alpha.reshape(-1).tolist()
    elif isinstance(activation_alpha, (list, tuple)):
      activation_alpha = list(activation_alpha)
    else:
      return float(activation_alpha)
    return [float(value) for value in activation_alpha]

  @staticmethod
  def _activation_alpha_active(activation_alpha):
    if activation_alpha is None:
      return False
    if hasattr(activation_alpha, 'detach'):
      values = activation_alpha.detach().cpu().flatten().tolist()
    elif isinstance(activation_alpha, np.ndarray):
      values = activation_alpha.reshape(-1).tolist()
    elif isinstance(activation_alpha, (list, tuple)):
      values = list(activation_alpha)
    else:
      return float(activation_alpha) > 0.0
    return any(float(value) > 0.0 for value in values)

  @classmethod
  def _format_activation_alpha(cls, activation_alpha):
    value = cls._json_activation_alpha(activation_alpha)
    if isinstance(value, list):
      return '[' + ', '.join(f'{float(item):.3f}' for item in value) + ']'
    if value is None:
      return 'None'
    return f'{float(value):.3f}'

  def _display_debug_output(self, image):
    if not self.live_visu or self._quit_requested:
      return

    if self._visu_interface is None:
      self._visu_interface = _SensorDebugInterface(image.shape[1], image.shape[0])

    self._visu_interface.run_interface(image)
    if self._visu_interface.quit_requested:
      self._quit_requested = True

  def stop_sign_controller_step(self, ego_speed):
    """Checks whether the car is intersecting with one of the detected stop signs"""
    if self.clear_stop_sign > 0:
      self.clear_stop_sign -= 1

    if len(self.bb_buffer) < 1:
      return False
    stop_sign_stop_predicted = False
    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)
    origin = carla.Location(x=0.0, y=0.0, z=0.0)

    car_box = carla.BoundingBox(origin, extent)

    for bb in self.bb_buffer[-1]:
      if bb[7] == 3:  # Stop sign detected
        self.stop_sign_buffer.append(bb)

    if len(self.stop_sign_buffer) > 0:
      # Check if we need to stop
      stop_box = self.stop_sign_buffer[0]
      stop_origin = carla.Location(x=stop_box[0], y=stop_box[1], z=0.0)
      stop_extent = carla.Vector3D(stop_box[2], stop_box[3], 1.0)
      stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
      stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box[4]), 0.0)

      if t_u.check_obb_intersection(stop_carla_box, car_box) and self.clear_stop_sign <= 0:
        if ego_speed > 0.01:
          stop_sign_stop_predicted = True
        else:
          # We have cleared the stop sign
          stop_sign_stop_predicted = False
          self.stop_sign_buffer.pop()
          # Stop signs don't come in herds, so we know we don't need to clear one for a while.
          self.clear_stop_sign = 100

    if len(self.stop_sign_buffer) > 0:
      # Remove boxes that are too far away
      if np.linalg.norm(self.stop_sign_buffer[0][:2]) > abs(self.config.max_x):
        self.stop_sign_buffer.pop()

    return stop_sign_stop_predicted

  def bb_detected_in_front_of_vehicle(self, ego_speed):
    if len(self.bb_buffer) < 1:  # We only start after we have 4 time steps.
      return False

    collision_predicted = False

    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)

    # Safety box
    bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0)**2 / 2.0  # Bremsweg formula for emergency break
    safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

    center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

    safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
    safety_bounding_box.rotation = carla.Rotation(0.0, 0.0, 0.0)

    for bb in self.bb_buffer[-1]:
      # We just give them some arbitrary height. Does not matter
      bb_extent_z = 1.0
      loc_local = carla.Location(bb[0], bb[1], 0.0)
      extent_det = carla.Vector3D(bb[2], bb[3], bb_extent_z)
      bb_local = carla.BoundingBox(loc_local, extent_det)
      bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb[4]).item(), 0.0)

      if t_u.check_obb_intersection(safety_bounding_box, bb_local):
        collision_predicted = True

    return collision_predicted

  def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar(lidar, pos_diff, rot_diff)

  def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target]) - np.array([x, y])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                                [np.sin(orientation_target), np.cos(orientation_target)]])
    pos_diff = rotation_matrix.T @ pos_diff

    # Rotation matrix in local coordinate system
    local_rot_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff)], [np.sin(rot_diff), np.cos(rot_diff)]])

    for _, box_pred in enumerate(boxes):
      box_pred[:2] = (local_rot_matrix.T @ (box_pred[:2] - pos_diff).T).T
      box_pred[4] = t_u.normalize_angle(box_pred[4] - rot_diff)

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    if self.save_path is not None:
      self.lon_logger.dump_to_json()
      if len(self.nets[0].speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.nets[0].speed_histogram, f, indent=4)

      if self.config.tp_attention:
        if len(self.tp_attention_buffer) > 0:
          print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
          with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_attention_buffer, f, indent=4)

        del self.tp_attention_buffer

    if self._visu_interface is not None:
      self._visu_interface.close()

    if self.vlm_gate is not None:
      self.vlm_gate.stop()

    del self.nets
    del self.config
    del self.metric_info


# Filter Functions
class _SensorDebugInterface:
  """Minimal pygame interface that displays one RGB image per step."""

  def __init__(self, width, height):
    self._width = width
    self._height = height
    self.quit_requested = False
    self._pygame = self._import_pygame()
    self._pygame.init()
    self._display = self._pygame.display.set_mode((self._width, self._height),
                                                  self._pygame.HWSURFACE | self._pygame.DOUBLEBUF)
    self._pygame.display.set_caption('Sensor Agent Debug')

  def _import_pygame(self):
    try:
      import pygame  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
      raise RuntimeError('cannot import pygame, make sure pygame package is installed') from exc
    return pygame

  def run_interface(self, image):
    for event in self._pygame.event.get():
      if event.type == self._pygame.QUIT:
        self.quit_requested = True
      if event.type == self._pygame.KEYDOWN and event.key == self._pygame.K_ESCAPE:
        self.quit_requested = True

    frame = np.ascontiguousarray(image)
    surface = self._pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    self._display.blit(surface, (0, 0))
    self._pygame.display.flip()

  def close(self):
    self._pygame.quit()


def bicycle_model_forward(x, dt, steer, throttle, brake):
  # Kinematic bicycle model.
  # Numbers are the tuned parameters from World on Rails
  front_wb = -0.090769015
  rear_wb = 1.4178275

  steer_gain = 0.36848336
  brake_accel = -4.952399
  throt_accel = 0.5633837

  locs_0 = x[0]
  locs_1 = x[1]
  yaw = x[2]
  speed = x[3]

  if brake:
    accel = brake_accel
  else:
    accel = throt_accel * throttle

  wheel = steer_gain * steer

  beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
  next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
  next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
  next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
  next_speed = speed + accel * dt
  next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

  next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

  return next_state_x


def measurement_function_hx(vehicle_state):
  '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
  return vehicle_state


def state_mean(state, wm):
  '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def measurement_mean(state, wm):
  '''
  We use the arctan of the average of sin and cos of the angle to
  calculate the average of orientations.
  :param state: array of states to be averaged. First index is the
  timestep.
  '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def residual_state_x(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


def residual_measurement_h(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


class EgoModel:
  """
      Kinematic bicycle model describing the motion of a car given it's state and
      action. Tuned parameters are taken from World on Rails.
      """

  def __init__(self, dt, ego_vehicle_model=True):
    self.dt = dt  # the following numbers are optimized for dt=1./20. = 20 FPS

    self.ego_vehicle_model = ego_vehicle_model

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275
    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

    # Numbers are tuned parameters for the polynomial equations below using
    # a dataset where the car drives on a straight highway, accelerates to
    # 80 km/h and brakes to 0 km/h
    self.throt_values = np.array([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ])
    self.brake_values = np.array([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ])

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt

    if self.ego_vehicle_model:
      if brake:
        spds = spds * 3.6
        features = np.array([spds, spds**2, spds**3, spds**4, spds**5, spds**6, spds**7]).T

        next_spds = (features @ self.brake_values).item() / 3.6
      else:
        throttle = np.clip(throt, 0., 1.0)
        # for a throttle value < 0.3 the car doesn't accelerate and the polynomial model below breaks
        if throttle < 0.3:
          next_spds = spds
        else:
          spds = spds * 3.6
          features = np.array([
              spds, spds**2, throttle, throttle**2, spds * throttle, spds * throttle**2, spds**2 * throttle,
              spds**2 * throttle**2
          ]).T

          next_spds = (features @ self.throt_values).item() / 3.6
    else:
      if brake:
        next_spds = spds + self.brake_accel * self.dt
      else:
        next_spds = spds + self.throt_accel * self.dt

    next_spds = max(0, next_spds)

    next_locs = np.array([next_locs_0, next_locs_1, locs[2]])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds

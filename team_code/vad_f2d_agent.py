"""Fail2Drive adapter for Bench2DriveZoo's VAD closed-loop agent."""

import os
import sys
import json
from pathlib import Path

import cv2
import math
import numpy as np
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAIL2DRIVE_ROOT = _REPO_ROOT / "fail2drive"
_F2D_LEADERBOARD = _FAIL2DRIVE_ROOT / "leaderboard"
_F2D_CARLA_API = _FAIL2DRIVE_ROOT / "f2d_carla" / "PythonAPI"
_F2D_CARLA_AGENTS = _F2D_CARLA_API / "carla"
_F2D_SCENARIO_RUNNER = _FAIL2DRIVE_ROOT / "scenario_runner"

for _path in (
    _REPO_ROOT,
    _FAIL2DRIVE_ROOT,
    _F2D_LEADERBOARD,
    _F2D_CARLA_API,
    _F2D_CARLA_AGENTS,
    _F2D_SCENARIO_RUNNER,
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# Bench2DriveZoo's VAD module reads this at import time. Keep it truthy so its
# config-name parsing follows the Bench2Drive closed-loop format.
os.environ.setdefault("IS_BENCH2DRIVE", "1")

from Bench2DriveZoo.team_code.vad_b2d_agent import VadAgent as Bench2DriveVadAgent  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: E402


def get_entry_point():
    return "Fail2DriveVadAgent"


def _strtobool(value):
    return str(value).lower() in ("1", "true", "t", "yes", "y")


class Fail2DriveVadAgent(Bench2DriveVadAgent):
    """Adapts Bench2DriveZoo VAD to fail2drive's local leaderboard API."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.route_index = route_index
        self.traffic_manager = traffic_manager
        self.visual_save_path = None
        os.environ.setdefault("ROUTES", str(route_index or "fail2drive_vad"))
        parts = path_to_conf_file.split("+")
        if len(parts) < 2:
            raise ValueError("VAD agent config must be '<config.py>+<checkpoint.pth>[+save_name]'.")
        if len(parts) == 2:
            path_to_conf_file = path_to_conf_file + "+vad_f2d"
        super().setup(path_to_conf_file)
        if self.save_path is not None:
            (self.save_path / "rgb_all").mkdir(exist_ok=True)
            for folder in (
                "rgb_front",
                "rgb_front_left",
                "rgb_front_right",
                "rgb_back",
                "rgb_back_left",
                "rgb_back_right",
            ):
                try:
                    (self.save_path / folder).rmdir()
                except OSError:
                    pass
        self._configure_visual_output_path()

    def _configure_visual_output_path(self):
        visual_root = os.environ.get("VISUAL_OUTPUT_PATH")
        if visual_root is None or self.save_path is None:
            return
        self.visual_save_path = Path(visual_root).resolve() / self.save_path.name
        self.visual_save_path.mkdir(parents=True, exist_ok=True)
        (self.visual_save_path / "rgb_all").mkdir(exist_ok=True)
        (self.visual_save_path / "bev").mkdir(exist_ok=True)

    def sensors(self):
        sensors = super().sensors()
        if _strtobool(os.environ.get("F2D_VAD_USE_BEV_SENSOR", "0")):
            return sensors
        return [sensor for sensor in sensors if sensor.get("id") != "bev"]

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        if "bev" in input_data:
            bev = cv2.cvtColor(input_data["bev"][1][:, :, :3], cv2.COLOR_BGR2RGB)
            bev_is_dummy = False
        else:
            bev = np.zeros((512, 512, 3), dtype=np.uint8)
            bev_is_dummy = True

        gps = input_data["GPS"][1][:2]
        speed = input_data["SPEED"][1]["speed"]
        compass = input_data["IMU"][1][-1]
        acceleration = input_data["IMU"][1][:3]
        angular_velocity = input_data["IMU"][1][3:6]

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)

        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        return {
            "imgs": imgs,
            "gps": gps,
            "pos": pos,
            "speed": speed,
            "compass": compass,
            "bev": bev,
            "bev_is_dummy": bev_is_dummy,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "command_near": near_command,
            "command_near_xy": near_node,
        }

    def _project_plan_to_front(self, front_img):
        plan = getattr(self, "pid_metadata", {}).get("plan")
        if not plan:
            return front_img

        vis = front_img.copy()
        points = np.asarray(plan, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return vis

        height, width = vis.shape[:2]
        focal = width / (2.0 * math.tan(math.radians(70.0) / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        camera_height = 1.60

        projected = []
        for lateral, forward in points[:, :2]:
            if forward <= 0.5:
                continue
            u = int(round(cx + focal * lateral / forward))
            v = int(round(cy + focal * camera_height / forward))
            if 0 <= u < width and 0 <= v < height:
                projected.append((u, v))

        for start, end in zip(projected[:-1], projected[1:]):
            cv2.line(vis, start, end, (255, 255, 0), 6, cv2.LINE_AA)
        for index, point in enumerate(projected):
            radius = 8 if index == len(projected) - 1 else 6
            cv2.circle(vis, point, radius, (255, 0, 0), -1, cv2.LINE_AA)

        return vis

    def _make_sensor_mosaic(self, imgs):
        tile_w, tile_h = 533, 300
        layout = [
            ("CAM_FRONT_LEFT", "front_left"),
            ("CAM_FRONT", "front + plan"),
            ("CAM_FRONT_RIGHT", "front_right"),
            ("CAM_BACK_LEFT", "back_left"),
            ("CAM_BACK", "back"),
            ("CAM_BACK_RIGHT", "back_right"),
        ]

        tiles = []
        for cam, label in layout:
            img = imgs[cam]
            if cam == "CAM_FRONT":
                img = self._project_plan_to_front(img)
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(tile, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)

        top = np.hstack(tiles[:3])
        bottom = np.hstack(tiles[3:])
        return np.vstack([top, bottom])

    def save(self, tick_data):
        frame = self.step
        image_save_path = self.visual_save_path if self.visual_save_path is not None else self.save_path

        mosaic = self._make_sensor_mosaic(tick_data["imgs"])
        Image.fromarray(mosaic).save(image_save_path / "rgb_all" / ("%06d.png" % frame))
        if not tick_data.get("bev_is_dummy", False):
            Image.fromarray(tick_data["bev"]).save(image_save_path / "bev" / ("%06d.png" % frame))

        with open(self.save_path / "meta" / ("%06d.json" % frame), "w") as outfile:
            json.dump(self.pid_metadata, outfile, indent=4)

        with open(self.save_path / "metric_info.json", "w") as outfile:
            json.dump(self.metric_info, outfile, indent=4)

    def get_metric_info(self):
        hero_actor = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.attributes.get("role_name") == "hero":
                hero_actor = actor
                break

        if hero_actor is None:
            return {}

        def vector2list(vector, rotation=False):
            if rotation:
                return [vector.roll, vector.pitch, vector.yaw]
            return [vector.x, vector.y, vector.z]

        transform = hero_actor.get_transform()
        return {
            "acceleration": vector2list(hero_actor.get_acceleration()),
            "angular_velocity": vector2list(hero_actor.get_angular_velocity()),
            "forward_vector": vector2list(transform.get_forward_vector()),
            "right_vector": vector2list(transform.get_right_vector()),
            "location": vector2list(transform.location),
            "rotation": vector2list(transform.rotation, rotation=True),
        }

    def destroy(self, results=None):
        try:
            super().destroy()
        except AttributeError:
            pass

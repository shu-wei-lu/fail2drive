"""Fail2Drive adapter for HiP-AD's Bench2Drive closed-loop agent."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

from activation_steering.policy import policy_from_env


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAIL2DRIVE_ROOT = _REPO_ROOT / "fail2drive"
_HIPAD_ROOT = _REPO_ROOT / "hip-ad"
_HIPAD_LEADERBOARD = _HIPAD_ROOT / "bench2drive" / "leaderboard"
_HIPAD_TEAM_CODE = _HIPAD_LEADERBOARD / "team_code"
_HIPAD_AGENT = _HIPAD_LEADERBOARD / "team_code" / "hipad_b2d_agent.py"
_F2D_LEADERBOARD = _FAIL2DRIVE_ROOT / "leaderboard"
_F2D_SCENARIO_RUNNER = _FAIL2DRIVE_ROOT / "scenario_runner"
_F2D_CARLA_API = _FAIL2DRIVE_ROOT / "f2d_carla" / "PythonAPI"
_F2D_CARLA_AGENTS = _F2D_CARLA_API / "carla"


def _prepend_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


for _path in reversed(
    (
        _F2D_LEADERBOARD,
        _HIPAD_LEADERBOARD,
        _HIPAD_ROOT,
        _F2D_SCENARIO_RUNNER,
        _F2D_CARLA_API,
        _F2D_CARLA_AGENTS,
    )
):
    _prepend_path(_path)

os.environ.setdefault("IS_BENCH2DRIVE", "1")
os.environ.setdefault("SAVE_PATH", str(_FAIL2DRIVE_ROOT / "results" / "hipad_f2d" / "vis"))
os.environ.setdefault("LEADERBOARD_ROOT", str(_F2D_LEADERBOARD))
os.environ.setdefault("SCENARIO_RUNNER_ROOT", str(_F2D_SCENARIO_RUNNER))

if not _HIPAD_AGENT.exists():
    raise FileNotFoundError(f"HiP-AD agent not found: {_HIPAD_AGENT}")

_team_code_module = types.ModuleType("team_code")
_team_code_module.__path__ = [str(_HIPAD_TEAM_CODE)]
sys.modules["team_code"] = _team_code_module

_spec = importlib.util.spec_from_file_location("_hipad_b2d_agent", _HIPAD_AGENT)
_hipad_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _hipad_module
_spec.loader.exec_module(_hipad_module)

_BaseHiPADAgent = _hipad_module.SparseAgent


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "t", "yes", "y")


def get_entry_point():
    return "Fail2DriveHiPADAgent"


class Fail2DriveHiPADAgent(_BaseHiPADAgent):
    """Adapts HiP-AD to fail2drive's local leaderboard API."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.route_index = route_index
        self.traffic_manager = traffic_manager
        self.visual_save_path = None
        self.activation_action_log = None
        self.activation_policy = policy_from_env(vlm_enabled=False)
        self.hipad_activation_alpha = 0.0
        self.hipad_activation_plan_shift = 0.0
        os.environ["ROUTES"] = str(route_index or os.environ.get("ROUTES", "fail2drive_hipad"))
        save_path = Path(os.environ["SAVE_PATH"]).resolve()
        os.environ["SAVE_PATH"] = str(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        parts = path_to_conf_file.split("+")
        if len(parts) < 2:
            raise ValueError("HiP-AD agent config must be '<config.py>+<checkpoint.pth>[+save_name]'.")
        if len(parts) == 2:
            path_to_conf_file = path_to_conf_file + "+hipad_f2d"

        cwd = os.getcwd()
        os.chdir(_HIPAD_ROOT)
        try:
            super().setup(path_to_conf_file)
        finally:
            os.chdir(cwd)
        self.is_visualize = _env_flag("DEBUG_CHALLENGE")
        if not self.is_visualize:
            self._remove_empty_log_visual_dirs()
        self.activation_action_log = self.save_path / "activation_actions.jsonl"
        self._configure_visual_output_path()

    def tick(self, input_data):
        if "bev" not in input_data:
            input_data = dict(input_data)
            input_data["bev"] = (None, np.zeros((512, 512, 4), dtype=np.uint8))
        return super().tick(input_data)

    def run_step(self, input_data, timestamp):
        self._set_plan_feature_env()
        self._set_activation_env()
        control = super().run_step(input_data, timestamp)
        self._log_activation_action(control)
        return control

    def _set_plan_feature_env(self):
        if getattr(self, "save_path", None) is not None:
            os.environ["HIPAD_PLAN_FEATURE_RUN_ID"] = self.save_path.name
        os.environ["HIPAD_PLAN_FEATURE_FRAME"] = str(int(getattr(self, "step", -1)) + 1)

    def _set_activation_env(self):
        frame = int(getattr(self, "step", -1)) + 1
        self.hipad_activation_alpha = self.activation_policy.alpha(frame)
        os.environ["HIPAD_ACTIVATION_ALPHA"] = self._format_activation_alpha(self.hipad_activation_alpha)
        self.hipad_activation_plan_shift = self._activation_plan_shift(self.hipad_activation_alpha)
        os.environ["HIPAD_ACTIVATION_PLAN_SHIFT"] = str(float(self.hipad_activation_plan_shift))

    def _configure_visual_output_path(self):
        if not getattr(self, "is_visualize", False):
            return
        visual_root = os.environ.get("VISUAL_OUTPUT_PATH")
        if visual_root is None or getattr(self, "save_path", None) is None:
            return
        self.visual_save_path = Path(visual_root).resolve() / self.save_path.name
        self.visual_save_path.mkdir(parents=True, exist_ok=True)
        (self.visual_save_path / "metas").mkdir(exist_ok=True)
        (self.visual_save_path / "images").mkdir(exist_ok=True)
        self._remove_empty_log_visual_dirs()

    def _remove_empty_log_visual_dirs(self):
        for folder in ("images", "metas"):
            try:
                (self.save_path / folder).rmdir()
            except OSError:
                pass

    def visualize(self, *args, **kwargs):
        if self.visual_save_path is None:
            return super().visualize(*args, **kwargs)
        log_save_path = self.save_path
        self.save_path = self.visual_save_path
        try:
            return super().visualize(*args, **kwargs)
        finally:
            self.save_path = log_save_path

    def destroy(self, results=None):
        super().destroy()

    def _log_activation_action(self, control):
        if self.activation_action_log is None:
            return
        metadata = getattr(self, "pid_metadata", {})
        frame = int(getattr(self, "step", -1))
        record = {
            "frame": frame,
            "speed": float(metadata.get("speed", 0.0)),
            "pred_target_speed": float(metadata.get("desired_speed", 0.0)),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "stop_for_stop_sign": False,
            "activation_alpha": self._json_activation_alpha(self.hipad_activation_alpha),
            "activation_plan_shift": float(self.hipad_activation_plan_shift),
            "feature_path": self._plan_feature_path(frame),
        }
        with open(self.activation_action_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _json_activation_alpha(activation_alpha):
        if activation_alpha is None:
            return None
        if hasattr(activation_alpha, "detach"):
            activation_alpha = activation_alpha.detach().cpu().flatten().tolist()
        elif isinstance(activation_alpha, np.ndarray):
            activation_alpha = activation_alpha.reshape(-1).tolist()
        elif isinstance(activation_alpha, (list, tuple)):
            activation_alpha = list(activation_alpha)
        else:
            return float(activation_alpha)
        return [float(value) for value in activation_alpha]

    @classmethod
    def _format_activation_alpha(cls, activation_alpha):
        value = cls._json_activation_alpha(activation_alpha)
        if value is None:
            return "0.0"
        if isinstance(value, list):
            return ",".join(str(float(item)) for item in value)
        return str(float(value))

    @classmethod
    def _activation_plan_shift(cls, activation_alpha):
        value = cls._json_activation_alpha(activation_alpha)
        if value is None:
            return 0.0
        if isinstance(value, list):
            return cls._activation_plan_shift_from_action_alpha(value)
        if float(value) <= 0.0:
            return 0.0
        direction = cls._infer_scalar_activation_direction()
        # if direction == "left":
        #     return -1.5
        # if direction == "right":
        #     return 1.5
        return 0.0

    @staticmethod
    def _activation_plan_shift_from_action_alpha(alpha_values):
        if len(alpha_values) < 3:
            return 0.0
        left_alpha = abs(float(alpha_values[1]))
        right_alpha = abs(float(alpha_values[2]))
        if left_alpha == 0.0 and right_alpha == 0.0:
            return 0.0
        # if right_alpha > left_alpha:
        #     return 1.5
        # if left_alpha > right_alpha:
        #     return -1.5
        return 0.0

    @classmethod
    def _infer_scalar_activation_direction(cls):
        for name in ("HIPAD_ACTIVATION_ACTION", "ACTIVATION_ACTION", "ORACLE_ACTION", "STEERING_ACTION"):
            action = cls._normalize_activation_direction(os.environ.get(name))
            if action is not None:
                return action

        for name in (
            "HIPAD_ACTIVATION_VECTOR_PATH",
            "ACTIVATION_VECTOR_PATH",
            "LEFT_ACTIVATION_VECTOR_PATH",
            "RIGHT_ACTIVATION_VECTOR_PATH",
            "ACTIVATION_VECTOR_PATH_LEFT",
            "ACTIVATION_VECTOR_PATH_RIGHT",
        ):
            action = cls._normalize_activation_direction(os.environ.get(name))
            if action is not None:
                return action
        return None

    @staticmethod
    def _normalize_activation_direction(value):
        if value is None:
            return None
        normalized = str(value).lower().replace("-", "_")
        if "right_change_lane" in normalized or "change_lane_right" in normalized:
            return "right"
        if "left_change_lane" in normalized or "change_lane_left" in normalized:
            return "left"
        parts = [part for part in normalized.replace("/", "_").split("_") if part]
        if "right" in parts or "rchange" in parts:
            return "right"
        if "left" in parts or "lchange" in parts:
            return "left"
        return None

    def _plan_feature_path(self, frame):
        if str(os.environ.get("SAVE_HIPAD_PLAN_FEATURES", "")).lower() not in ("1", "true", "t", "yes", "y"):
            return None
        root = os.environ.get("FUSED_FEATURES_PATH")
        if root is None or getattr(self, "save_path", None) is None:
            return None
        return str(Path(root) / self.save_path.name / f"{int(frame):06d}.pt")

    def get_metric_info(self):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

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
            "location": vector2list(hero_actor.get_location()),
            "rotation": vector2list(transform.rotation, rotation=True),
            "speed": hero_actor.get_velocity().length(),
        }

#!/usr/bin/env python3
"""Recreate the steering feature route split from post-processed summaries."""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "fail2drive_split"
OUTPUT_DIR = ROOT / "steering_split"

DEFAULT_DISTANCE = "30"
DEFAULT_STOP_TIME = "20"
DEFAULT_VEHICLE = "vehicle.*"
LANE_CHANGE_LATERAL_SHIFT_METERS = 1.5

LANE_CHANGE_ROUTE_IDS = [
    0,
    1,
    2,
    3,
    4,
    35,
    36,
    37,
    38,
    39,
    1085,
    1086,
    1087,
    1088,
    1089,
]


SPLITS = {
    "Brake": {
        "stem": "SteeringVehicleOnRoad",
        "ids": [0, 37, 38, 1085],
        "lateral_offset": 0.0,
    },
    "LaneChangeLeft": {
        "stem": "SteeringChangeLaneLeft",
        "ids": LANE_CHANGE_ROUTE_IDS,
        # Positive VehicleOnRoad lateral offsets move right in CARLA.
        # Place the obstacle to the right so the ego steers left.
        "lateral_offset": LANE_CHANGE_LATERAL_SHIFT_METERS,
    },
    "LaneChangeRight": {
        "stem": "SteeringChangeLaneRight",
        "ids": LANE_CHANGE_ROUTE_IDS,
        # Place the obstacle to the left so the ego steers right.
        "lateral_offset": -LANE_CHANGE_LATERAL_SHIFT_METERS,
    },
    "Normal": {
        "stem": "Normal",
        "ids": [0, 1085],
        "lateral_offset": None,
    },
}


def route_id_from_file(path: Path) -> int:
    tree = ET.parse(path)
    route = tree.getroot().find("route")
    if route is None or route.get("id") is None:
        raise ValueError(f"No route id in {path}")
    return int(route.get("id"))


def source_file_for(route_id: int) -> Path:
    candidates = sorted(SOURCE_DIR.glob(f"*_{route_id:04d}.xml"))
    for candidate in candidates:
        if route_id_from_file(candidate) == route_id:
            return candidate
    raise FileNotFoundError(f"No source XML found for route id {route_id:04d}")


def first_trigger_point(route_elem: ET.Element) -> ET.Element:
    trigger = route_elem.find("./scenarios/scenario/trigger_point")
    if trigger is None:
        raise ValueError(f"Route {route_elem.get('id')} has no trigger point to reuse")
    return trigger


def add_value(parent: ET.Element, tag: str, value: str) -> None:
    ET.SubElement(parent, tag, {"value": value})


def build_vehicle_on_road_scenario(
    stem: str,
    trigger_point: ET.Element,
    lateral_offset: float,
) -> ET.Element:
    scenario = ET.Element("scenario", {"name": stem, "type": "VehicleOnRoad"})
    add_value(scenario, "distance", DEFAULT_DISTANCE)
    add_value(scenario, "stop_time", DEFAULT_STOP_TIME)
    add_value(scenario, "vehicle", DEFAULT_VEHICLE)
    add_value(scenario, "lateral_offset", f"{lateral_offset:g}")
    scenario.append(copy.deepcopy(trigger_point))
    return scenario


def build_route_xml(source_file: Path, stem: str, lateral_offset: float | None) -> ET.ElementTree:
    source_tree = ET.parse(source_file)
    source_route = source_tree.getroot().find("route")
    if source_route is None:
        raise ValueError(f"No route element in {source_file}")

    root = ET.Element("routes")
    route = ET.SubElement(root, "route", dict(source_route.attrib))

    for tag in ("weathers", "waypoints"):
        child = source_route.find(tag)
        if child is None:
            raise ValueError(f"No {tag} element in {source_file}")
        route.append(copy.deepcopy(child))

    scenarios = ET.SubElement(route, "scenarios")
    if lateral_offset is not None:
        trigger_point = first_trigger_point(source_route)
        scenarios.append(build_vehicle_on_road_scenario(stem, trigger_point, lateral_offset))

    return ET.ElementTree(root)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for group_name, spec in SPLITS.items():
        group_dir = OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for route_id in spec["ids"]:
            source_file = source_file_for(route_id)
            output_file = group_dir / f"{spec['stem']}_{route_id:04d}.xml"
            tree = build_route_xml(source_file, spec["stem"], spec["lateral_offset"])
            ET.indent(tree, space="  ")
            tree.write(output_file, encoding="unicode", short_empty_elements=True)
            written.append((output_file, source_file))

    for output_file, source_file in written:
        print(f"{output_file.relative_to(ROOT)} <- {source_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

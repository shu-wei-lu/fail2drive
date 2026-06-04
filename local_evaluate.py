#!/usr/bin/env python3
"""Sequential Fail2Drive evaluation runner for a local machine.

This is the non-SLURM counterpart of slurm_evaluate.py. Start one CARLA
server yourself, then this script runs routes one by one and writes the same
seed/res, seed/out, seed/err layout consumed by tools/f2d_result_parser.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes",
        default="fail2drive_split",
        help="Path to folder containing Fail2Drive split route XML files.",
    )
    parser.add_argument(
        "--out_root",
        default="results/fail2drive_local",
        help="Path where results should be stored.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--lb_script",
        default="leaderboard/leaderboard/leaderboard_evaluator_local.py",
        help="Path to leaderboard evaluator script.",
    )
    parser.add_argument("--agent_file", required=True, help="Path to agent entry file.")
    parser.add_argument("--agent_config", required=True, help="Path to agent config/checkpoint.")
    parser.add_argument("--track", default="SENSORS", help="Leaderboard track passed to evaluator.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--route-filter",
        default="",
        help="Only run routes whose filename contains this text.",
    )
    parser.add_argument(
        "--route-limit",
        type=int,
        default=0,
        help="Run only the first N routes after sorting/filtering. Useful for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun routes even when an existing non-retryable result is present.",
    )
    return parser.parse_args()


def route_files(route_dir: Path, route_filter: str, route_limit: int) -> list[Path]:
    routes = sorted(route_dir.glob("*.xml"))
    if route_filter:
        routes = [route for route in routes if route_filter in route.name]
    if route_limit > 0:
        routes = routes[:route_limit]
    if not routes:
        raise RuntimeError(f"No route XML files found in {route_dir}")
    return routes


def result_is_done(result_file: Path) -> bool:
    if not result_file.exists():
        return False
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    checkpoint = data.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return False

    progress = checkpoint.get("progress")
    records = checkpoint.get("records")
    if not isinstance(progress, list) or not isinstance(records, list):
        return False
    if len(progress) < 2 or progress[0] < progress[1] or not records:
        return False

    return not any(record.get("status") in RETRYABLE_STATUSES for record in records)


def tee_command(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def run_route(args: argparse.Namespace, route: Path, seed: int, attempt: int) -> int:
    route_id = route.stem.rsplit("_", maxsplit=1)[-1]
    route_seed = int(route_id) % 1000 + (10000 * seed)

    base_dir = Path(args.out_root) / str(seed)
    result_dir = base_dir / "res"
    out_dir = base_dir / "out"
    err_dir = base_dir / "err"
    for path in (result_dir, out_dir, err_dir):
        path.mkdir(parents=True, exist_ok=True)

    result_file = result_dir / f"{route_id}_res.json"
    out_file = out_dir / f"{route_id}_out.log"
    err_file = err_dir / f"{route_id}_err.log"
    debug_file = out_dir / f"{route_id}_live.txt"

    command = [
        sys.executable,
        "-u",
        args.lb_script,
        f"--routes={route}",
        "--repetitions=1",
        f"--track={args.track}",
        f"--checkpoint={result_file}",
        f"--debug-checkpoint={debug_file}",
        f"--timeout={args.timeout}",
        f"--agent={args.agent_file}",
        f"--agent-config={args.agent_config}",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--traffic-manager-seed={route_seed}",
    ]

    with out_file.open("a", encoding="utf-8") as stdout, err_file.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n[local_evaluate] attempt={attempt} seed={seed} route={route.name}\n")
        stdout.write(f"[local_evaluate] command={tee_command(command)}\n")
        stdout.flush()
        return subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode


def main() -> int:
    args = parse_args()
    routes = route_files(Path(args.routes), args.route_filter, args.route_limit)
    total = len(routes) * len(args.seeds)
    current = 0

    print(f"Running {len(routes)} routes x {len(args.seeds)} seeds = {total} evaluations")
    print("CARLA must already be running and reachable at " f"{args.host}:{args.port}")

    for seed in args.seeds:
        for route in routes:
            current += 1
            route_id = route.stem.rsplit("_", maxsplit=1)[-1]
            result_file = Path(args.out_root) / str(seed) / "res" / f"{route_id}_res.json"

            if not args.force and result_is_done(result_file):
                print(f"[{current}/{total}] skip seed={seed} route={route.name}")
                continue

            for attempt in range(1, args.retries + 1):
                print(f"[{current}/{total}] run seed={seed} route={route.name} attempt={attempt}")
                return_code = run_route(args, route, seed, attempt)
                if result_is_done(result_file):
                    break
                print(
                    f"[warn] route did not finish cleanly "
                    f"(returncode={return_code}, result={result_file})"
                )
            else:
                print(f"[fail] exhausted retries for seed={seed} route={route.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

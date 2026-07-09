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
import signal
import shlex
import socket
import subprocess
import sys
import time
from typing import Iterable


RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}

FATAL_EVALUATOR_LOG_PATTERNS = (
    "Watchdog exception - Timeout",
    "The simulation took longer than",
    "time-out of 300000ms while waiting for the simulator",
)


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
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--evaluator-wall-timeout",
        type=float,
        default=0.0,
        help="Optional wall-clock timeout for the evaluator subprocess. 0 disables it.",
    )
    parser.add_argument(
        "--evaluator-fatal-grace",
        type=float,
        default=60.0,
        help=(
            "Seconds to wait for the evaluator to exit after a fatal watchdog/CARLA "
            "timeout appears in its logs before killing the evaluator process group."
        ),
    )
    parser.add_argument(
        "--restart-carla",
        action="store_true",
        help="Start a fresh CARLA server for every route attempt and stop it afterwards.",
    )
    parser.add_argument(
        "--carla-root",
        default=os.environ.get("CARLA_ROOT", ""),
        help="CARLA root directory. Defaults to $CARLA_ROOT.",
    )
    parser.add_argument(
        "--carla-executable",
        default="",
        help="Path to CarlaUE4.sh. Defaults to <carla-root>/CarlaUE4.sh.",
    )
    parser.add_argument(
        "--carla-startup-wait",
        type=float,
        default=60.0,
        help="Seconds to wait after launching CARLA before starting the evaluator.",
    )
    parser.add_argument(
        "--carla-shutdown-wait",
        type=float,
        default=5.0,
        help="Seconds to wait after stopping CARLA before the next attempt.",
    )
    parser.add_argument(
        "--port-free-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for CARLA/evaluator ports to become free before launching CARLA.",
    )
    parser.add_argument(
        "--carla-streaming-port",
        type=int,
        default=0,
        help="Optional CARLA streaming port. 0 lets CARLA use its default.",
    )
    parser.add_argument(
        "--carla-nullrhi",
        action="store_true",
        help="Start CARLA with -nullrhi. Do not use this if the agent needs RGB cameras.",
    )
    parser.add_argument(
        "--graphics-adapter",
        default="",
        help="Optional GPU index passed as -graphicsadapter=<index>.",
    )
    parser.add_argument(
        "--carla-extra-args",
        default="",
        help="Extra CARLA launch args, parsed with shell-like quoting.",
    )
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


def remove_incomplete_result(result_file: Path) -> None:
    if not result_file.exists() or result_is_done(result_file):
        return
    try:
        result_file.unlink()
    except OSError as exc:
        print(f"[warn] could not remove incomplete result {result_file}: {exc}")


def tee_command(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def log_contains_any(log_files: Iterable[Path], patterns: Iterable[str], max_bytes: int = 65536) -> bool:
    for log_file in log_files:
        try:
            with log_file.open("rb") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(max(0, size - max_bytes), os.SEEK_SET)
                text = file.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        if any(pattern in text for pattern in patterns):
            return True
    return False


def stop_process_group(
    process: subprocess.Popen,
    stdout,
    reason: str,
    terminate_timeout: float = 20.0,
) -> int:
    if process.poll() is not None:
        return process.returncode

    stdout.write(f"[local_evaluate] {reason}; killing evaluator process group\n")
    stdout.flush()
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=terminate_timeout)
        except subprocess.TimeoutExpired:
            stdout.write("[local_evaluate] evaluator ignored SIGTERM; sending SIGKILL\n")
            stdout.flush()
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=terminate_timeout)
    except ProcessLookupError:
        process.wait(timeout=terminate_timeout)
    return process.returncode


def carla_executable(args: argparse.Namespace) -> Path:
    if args.carla_executable:
        return Path(args.carla_executable)
    if not args.carla_root:
        raise RuntimeError("--restart-carla requires --carla-root or CARLA_ROOT")
    return Path(args.carla_root) / "CarlaUE4.sh"


def carla_command(args: argparse.Namespace) -> list[str]:
    executable = carla_executable(args)
    if not executable.exists():
        raise RuntimeError(f"CARLA executable not found: {executable}")

    binary = executable.parent / "CarlaUE4" / "Binaries" / "Linux" / "CarlaUE4-Linux-Shipping"
    if executable.name == "CarlaUE4.sh" and binary.exists():
        binary.chmod(binary.stat().st_mode | 0o111)
        command = [str(binary), "CarlaUE4"]
    elif executable.suffix == ".sh":
        command = ["bash", str(executable)]
    else:
        command = [str(executable)]

    command.extend(
        [
            f"-carla-rpc-port={args.port}",
            "-nosound",
            "-carla-primary-port=0",
        ]
    )
    if args.carla_streaming_port:
        command.append(f"-carla-streaming-port={args.carla_streaming_port}")
    if args.carla_nullrhi:
        command.append("-nullrhi")
    else:
        command.append("-RenderOffscreen")
    if args.graphics_adapter:
        command.append(f"-graphicsadapter={args.graphics_adapter}")
    if args.carla_extra_args:
        command.extend(shlex.split(args.carla_extra_args))
    return command


def socket_port_is_free(port: int, socket_type: int) -> bool:
    if port <= 0:
        return True
    with socket.socket(socket.AF_INET, socket_type) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def port_is_free(port: int) -> bool:
    return socket_port_is_free(port, socket.SOCK_STREAM) and socket_port_is_free(port, socket.SOCK_DGRAM)


def required_ports(args: argparse.Namespace) -> list[int]:
    ports = [args.port, args.traffic_manager_port]
    if args.carla_streaming_port:
        ports.append(args.carla_streaming_port)
    return sorted(set(ports))


def wait_for_free_ports(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.port_free_timeout
    ports = required_ports(args)
    while True:
        busy_ports = [port for port in ports if not port_is_free(port)]
        if not busy_ports:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"ports still in use after {args.port_free_timeout:.1f}s: {busy_ports}")
        time.sleep(1.0)


def start_carla(args: argparse.Namespace) -> subprocess.Popen:
    wait_for_free_ports(args)
    command = carla_command(args)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(args.carla_startup_wait)
    if process.poll() is not None:
        raise RuntimeError(f"CARLA exited during startup with code {process.returncode}")
    return process


def stop_carla(process: subprocess.Popen, args: argparse.Namespace, timeout: float = 20.0) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + args.port_free_timeout
            while process.poll() is None and time.monotonic() < deadline:
                if not [port for port in required_ports(args) if not port_is_free(port)]:
                    break
                time.sleep(1.0)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=timeout)
        except ProcessLookupError:
            pass
    try:
        wait_for_free_ports(args)
    except RuntimeError as exc:
        print(f"[warn] after stopping CARLA: {exc}")


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
        f"--traffic-manager-port={args.traffic_manager_port}",
        f"--traffic-manager-seed={route_seed}",
    ]

    with out_file.open("a", encoding="utf-8") as stdout, err_file.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n[local_evaluate] attempt={attempt} seed={seed} route={route.name}\n")
        stdout.write(f"[local_evaluate] command={tee_command(command)}\n")
        stdout.flush()
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        wall_deadline = None
        if args.evaluator_wall_timeout > 0:
            wall_deadline = time.monotonic() + args.evaluator_wall_timeout
        fatal_seen_at = None

        while True:
            if process.poll() is not None:
                return process.returncode

            now = time.monotonic()
            if wall_deadline is not None and now >= wall_deadline:
                return stop_process_group(
                    process,
                    stdout,
                    f"evaluator timed out after {args.evaluator_wall_timeout:.1f}s",
                )

            if fatal_seen_at is None and log_contains_any(
                (out_file, err_file),
                FATAL_EVALUATOR_LOG_PATTERNS,
            ):
                fatal_seen_at = now
                stdout.write(
                    "[local_evaluate] detected fatal evaluator timeout in logs; "
                    f"waiting {args.evaluator_fatal_grace:.1f}s for clean exit\n"
                )
                stdout.flush()

            if fatal_seen_at is not None and now - fatal_seen_at >= args.evaluator_fatal_grace:
                return stop_process_group(
                    process,
                    stdout,
                    "evaluator did not exit after fatal timeout",
                )

            time.sleep(1.0)


def main() -> int:
    args = parse_args()
    routes = route_files(Path(args.routes), args.route_filter, args.route_limit)
    total = len(routes) * len(args.seeds)
    current = 0

    print(f"Running {len(routes)} routes x {len(args.seeds)} seeds = {total} evaluations")
    if args.restart_carla:
        print("CARLA will be restarted for every route attempt at " f"{args.host}:{args.port}")
    else:
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
                if attempt > 1 or args.force:
                    remove_incomplete_result(result_file)
                print(f"[{current}/{total}] run seed={seed} route={route.name} attempt={attempt}")
                carla_process = None
                return_code = 1
                try:
                    if args.restart_carla:
                        carla_process = start_carla(args)
                    return_code = run_route(args, route, seed, attempt)
                except RuntimeError as exc:
                    print(f"[warn] CARLA failed to start for seed={seed} route={route.name}: {exc}")
                finally:
                    if carla_process is not None:
                        stop_carla(carla_process, args)
                        time.sleep(args.carla_shutdown_wait)
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

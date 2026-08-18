#!/usr/bin/env python3
"""Submit one steering-feature collection Slurm job per route.

Each job starts its own CARLA server, runs ``create_steering_features.py`` for
one route XML, and stops CARLA on exit.  ``--routes`` accepts either one route
XML or a directory containing route XML files, matching
``create_steering_features.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CREATE_SCRIPT = REPO_ROOT / "create_steering_features.py"
DEFAULT_CONDA_SH = REPO_ROOT.parent / "miniconda3" / "etc" / "profile.d" / "conda.sh"
DEFAULT_SLURM_LOG_ROOT = REPO_ROOT / "results" / "slurm_create_steering_features"

# Edit these values to match the cluster.
SBATCH_OPTIONS = {
    "partition": "devq",
    "qos": "normal",
    "nodes": "1",
    "ntasks": "1",
    "cpus-per-task": "8",
    "mem": "32gb",
    "time": "8:00:00",
    "gres": "gpu:nvidia_geforce_rtx_4090:1",
}


def safe_name(value: str) -> str:
    """Return a value safe for Slurm job names and generated filenames."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not result:
        raise ValueError(f"Invalid empty name derived from {value!r}")
    return result


def default_slurm_log_root(output_root: Path) -> Path:
    """Keep scheduler artifacts outside the steering-feature collection tree."""
    try:
        output_label = output_root.relative_to(REPO_ROOT)
    except ValueError:
        output_label = output_root
    return DEFAULT_SLURM_LOG_ROOT / safe_name(str(output_label))


def list_routes(routes_arg: str, route_filter: str, route_limit: int) -> list[Path]:
    """Select routes using the same file/directory rules as the local script."""
    route_path = Path(routes_arg).expanduser()
    if not route_path.is_absolute():
        route_path = REPO_ROOT / route_path
    route_path = route_path.resolve()

    if route_path.is_file():
        if route_path.suffix.lower() != ".xml":
            raise ValueError(f"Route is not an XML file: {route_path}")
        routes = [route_path]
    elif route_path.is_dir():
        routes = sorted(route.resolve() for route in route_path.glob("*.xml"))
    else:
        raise FileNotFoundError(f"Route path does not exist: {route_path}")

    if route_filter:
        routes = [route for route in routes if route_filter in route.name]
    if route_limit > 0:
        routes = routes[:route_limit]
    if not routes:
        raise RuntimeError(f"No route XML files found for {routes_arg}")
    return routes


def validate_env_overrides(items: list[str]) -> list[str]:
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --env, got: {item}")
        key, _ = item.split("=", maxsplit=1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
    return items


def make_feature_command(args: argparse.Namespace, route: Path) -> str:
    agent_file = Path(os.path.expanduser(os.path.expandvars(args.agent_file)))
    if not agent_file.is_absolute():
        agent_file = REPO_ROOT / agent_file
    command = [
        "python",
        "-u",
        str(CREATE_SCRIPT),
        "--routes",
        str(route),
        "--agent-file",
        str(agent_file.resolve()),
        "--agent-config",
        os.path.expanduser(os.path.expandvars(args.agent_config)),
        "--output-root",
        str(args.output_root),
        "--port",
        '"$FREE_WORLD_PORT"',
        "--timeout",
        str(args.timeout),
        "--repetitions",
        str(args.repetitions),
    ]
    if args.max_frames > 0:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.save_visual_output:
        command.append("--save-visual-output")
    if args.no_other_vehicles:
        command.append("--no-other-vehicles")
    for item in args.env:
        command.extend(["--env", item])

    # shlex.join would quote "$FREE_WORLD_PORT" and prevent shell expansion.
    rendered = shlex.join(command)
    return rendered.replace("'\"$FREE_WORLD_PORT\"'", '"$FREE_WORLD_PORT"')


def make_job_script(
    args: argparse.Namespace,
    route: Path,
    job_output: Path,
) -> str:
    route_name = safe_name(route.stem)
    directives = "\n".join(
        f"#SBATCH --{key}={value}" for key, value in SBATCH_OPTIONS.items()
    )
    feature_command = make_feature_command(args, route)

    return f"""#!/bin/bash
#SBATCH --job-name=steer_{route_name}
{directives}
#SBATCH --output={job_output}/slurm-%j.out
#SBATCH --error={job_output}/slurm-%j.err

set -eo pipefail

source {shlex.quote(str(args.conda_sh))}
conda activate {shlex.quote(args.conda_env)}

set -u

: "${{CARLA_ROOT:?Export CARLA_ROOT before running the Slurm submitter.}}"
: "${{SCENARIO_RUNNER_ROOT:?Export SCENARIO_RUNNER_ROOT before running the Slurm submitter.}}"

cd {shlex.quote(str(REPO_ROOT))}

echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: $(hostname)"
echo "Route: {shlex.quote(str(route))}"
echo "Python: $(which python)"
echo "Conda environment: ${{CONDA_DEFAULT_ENV:-unknown}}"
echo "Output root: {shlex.quote(str(args.output_root))}"

free_port() {{
  local start=$1
  comm -23 <(seq "$start" "$((start + 99))" | sort) \\
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) \\
    | shuf | head -n 1
}}

FREE_WORLD_PORT=$(free_port 10000)
FREE_STREAMING_PORT=$(free_port 20000)
echo "World port: $FREE_WORLD_PORT"
echo "Streaming port: $FREE_STREAMING_PORT"

CARLA_LOG={shlex.quote(str(job_output / "carla.log"))}
"$CARLA_ROOT/CarlaUE4.sh" \\
  -carla-rpc-port="$FREE_WORLD_PORT" \\
  -carla-streaming-port="$FREE_STREAMING_PORT" \\
  -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 \\
  >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!

cleanup() {{
  kill "$CARLA_PID" 2>/dev/null || true
  wait "$CARLA_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

sleep {args.carla_startup_seconds}
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "CARLA exited before feature collection started. Last log lines:" >&2
  tail -100 "$CARLA_LOG" >&2 || true
  exit 1
fi

test -f "$SCENARIO_RUNNER_ROOT/speed_limits/Town13_speed_limits.npy"

{feature_command}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes",
        required=True,
        help="One route XML file, or a directory containing route XML files.",
    )
    parser.add_argument(
        "--route-filter",
        default="",
        help="Only submit route filenames containing this text.",
    )
    parser.add_argument(
        "--route-limit",
        type=int,
        default=0,
        help="Submit at most this many selected routes. 0 means no limit.",
    )
    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--slurm-log-root",
        type=Path,
        help=(
            "Directory for generated job scripts and Slurm/CARLA logs. "
            "Defaults to results/slurm_create_steering_features/<output-name>, "
            "outside the feature collection root."
        ),
    )
    parser.add_argument(
        "--conda-env",
        required=True,
        help="Conda environment used inside every job, e.g. hipad or fail2drive.",
    )
    parser.add_argument("--conda-sh", type=Path, default=DEFAULT_CONDA_SH)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--carla-startup-seconds", type=int, default=60)
    parser.add_argument("--save-visual-output", action="store_true")
    parser.add_argument("--no-other-vehicles", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override forwarded to create_steering_features.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.route_limit < 0:
        parser.error("--route-limit must be >= 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.carla_startup_seconds < 0:
        parser.error("--carla-startup-seconds must be >= 0")

    args.conda_sh = args.conda_sh.expanduser().resolve()
    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = (REPO_ROOT / args.output_root).resolve()
    if args.slurm_log_root is None:
        args.slurm_log_root = default_slurm_log_root(args.output_root)
    else:
        args.slurm_log_root = args.slurm_log_root.expanduser()
        if not args.slurm_log_root.is_absolute():
            args.slurm_log_root = (REPO_ROOT / args.slurm_log_root).resolve()
    args.env = validate_env_overrides(args.env)
    return args


def main() -> None:
    args = parse_args()
    routes = list_routes(args.routes, args.route_filter, args.route_limit)
    if not CREATE_SCRIPT.is_file():
        raise FileNotFoundError(f"Feature creation script not found: {CREATE_SCRIPT}")
    if not args.conda_sh.is_file():
        raise FileNotFoundError(f"Conda activation script not found: {args.conda_sh}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_root = args.slurm_log_root
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {len(routes)} independent steering-feature job(s).")
    print(f"Feature collection root: {args.output_root}")
    print(f"Slurm log root: {run_root}")
    for route in routes:
        route_name = safe_name(route.stem)
        job_output = run_root / route_name
        job_output.mkdir(parents=True, exist_ok=True)
        job_file = job_output / "job.sh"
        job_file.write_text(
            make_job_script(args, route, job_output),
            encoding="utf-8",
        )
        job_file.chmod(0o755)

        if args.dry_run:
            print(f"[dry-run] {route.name}: {job_file}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"{route.name}: {result.stdout.strip()}")


if __name__ == "__main__":
    main()

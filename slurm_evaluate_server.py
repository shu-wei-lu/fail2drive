#!/usr/bin/env python3
"""Submit Fail2Drive server-backed evaluation as one Slurm job per route.

Each submitted job runs the selected Depth-TTC or Alpamayo server, CARLA, and
one leaderboard route on the same node. This mirrors slurm_evaluate.py's
parallel per-route execution while keeping localhost server URLs valid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


JOB_PREFIX = "F2DServer_"
RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}
FATAL_LOG_PATTERNS = (
    "Watchdog exception",
    "Engine crash handling finished",
    "Stopping the route, the agent has crashed",
    "Error during the simulation",
    "The simulation took longer than",
)
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class EvaluationJob:
    route: Path
    route_id: str
    seed: int
    result_file: Path
    stdout_file: Path
    stderr_file: Path
    evaluator_stdout_file: Path
    evaluator_stderr_file: Path
    debug_dir: Path
    attempts_left: int
    job_id: str | None = None

    @property
    def name(self) -> str:
        return f"{JOB_PREFIX}{self.seed}_{self.route_id}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Submit one two-GPU Slurm job per Fail2Drive route. Each job starts "
            "its own Depth-TTC or Alpamayo server on the same node as CARLA."
        )
    )
    parser.add_argument("--eval-mode", choices=("depth_ttc", "vlm"), default="depth_ttc")
    parser.add_argument("--routes", default="fail2drive_split")
    parser.add_argument("--out-root", default="results/depth_ttc_slurm")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Maximum Slurm submissions per route/seed, including the first attempt.",
    )
    parser.add_argument("--route-filter", default="")
    parser.add_argument("--route-limit", type=int, default=0)
    parser.add_argument("--agent-file", default="team_code/sensor_agent.py")
    parser.add_argument("--agent-config", default="checkpoints/tfpp")
    parser.add_argument("--track", default="SENSORS")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--template",
        default=str(root / "sbatch_evaluate_server.sh"),
        help="Two-GPU Slurm batch template used for every route.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum active jobs. Defaults to eval_num_jobs.txt.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra environment variable inherited by every submitted job; repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_extra_env(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must use NAME=VALUE syntax: {item!r}")
        name, value = item.split("=", 1)
        if not ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        result[name] = value
    return result


def route_files(route_dir: Path, route_filter: str, route_limit: int) -> list[Path]:
    routes = sorted(route_dir.glob("*.xml"))
    if route_filter:
        routes = [route for route in routes if route_filter in route.name]
    if route_limit > 0:
        routes = routes[:route_limit]
    if not routes:
        raise RuntimeError(f"No route XML files found under {route_dir}")
    route_ids = [route.stem.rsplit("_", maxsplit=1)[-1] for route in routes]
    duplicate_ids = sorted({route_id for route_id in route_ids if route_ids.count(route_id) > 1})
    if duplicate_ids:
        raise RuntimeError(
            "Route ids must be unique because result filenames use the final id; "
            f"duplicates: {duplicate_ids}"
        )
    return routes


def result_is_complete(result_file: Path) -> bool:
    if not result_file.exists():
        return False
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    checkpoint = data.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return False
    progress = checkpoint.get("progress")
    records = checkpoint.get("records")
    if not isinstance(progress, list) or len(progress) < 2:
        return False
    if not isinstance(records, list) or not records:
        return False
    if progress[0] < progress[1]:
        return False
    return not any(
        not isinstance(record, dict) or record.get("status") in RETRYABLE_STATUSES
        for record in records
    )


def max_parallel_jobs(repo_root: Path, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--max-parallel must be at least 1")
        return override
    path = repo_root / "eval_num_jobs.txt"
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        print(f"[warn] Could not read {path}: {exc}; using max_parallel=1")
        return 1
    return max(1, value)


def validate_mode_dependencies(
    args: argparse.Namespace, repo_root: Path, extra_env: dict[str, str]
) -> None:
    env = dict(os.environ)
    env.update(extra_env)
    root = Path(env.get("ROOT", str(repo_root.parent))).expanduser()
    if args.eval_mode == "depth_ttc":
        depth_root = Path(env.get("DEPTH_ROOT", str(root / "Depth-Anything-V2"))).expanduser()
        checkpoint = Path(
            env.get(
                "DEPTH_CHECKPOINT",
                str(
                    depth_root
                    / "metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"
                ),
            )
        ).expanduser()
        if not (depth_root / "metric_depth/depth_anything_v2").is_dir():
            raise FileNotFoundError(f"Depth Anything code not found under {depth_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Depth Anything checkpoint not found: {checkpoint}")
        return

    alpamayo_root = Path(
        env.get("ALPAMAYO_ROOT", str(root / "alpamayo1.5"))
    ).expanduser()
    alpamayo_python = Path(
        env.get("ALPAMAYO_PY", str(alpamayo_root / "a1_5_venv/bin/python"))
    ).expanduser()
    if not alpamayo_python.is_file() or not os.access(alpamayo_python, os.X_OK):
        raise FileNotFoundError(
            f"Alpamayo Python not found/executable: {alpamayo_python}; "
            "provide --env ALPAMAYO_ROOT=... or --env ALPAMAYO_PY=..."
        )


def slurm_jobs() -> dict[str, str]:
    try:
        process = subprocess.run(
            ["squeue", "--me", "--noheader", "--format", "%A|%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to query Slurm queue: {exc}") from exc

    jobs: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, name = line.split("|", 1)
        if name.startswith(JOB_PREFIX):
            jobs[job_id.strip()] = name.strip()
    return jobs


def log_has_fatal_pattern(log_file: Path, max_bytes: int = 65536) -> bool:
    try:
        with log_file.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            text = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return False
    return any(pattern in text for pattern in FATAL_LOG_PATTERNS)


def cancel_dead_jobs(jobs: list[EvaluationJob], running: dict[str, str]) -> None:
    for job in jobs:
        if job.job_id not in running:
            continue
        log_files = (
            job.stdout_file,
            job.stderr_file,
            job.evaluator_stdout_file,
            job.evaluator_stderr_file,
        )
        if not any(log_has_fatal_pattern(path) for path in log_files):
            continue
        print(f"[cancel] fatal log pattern job={job.job_id} route={job.route.name}")
        subprocess.run(["scancel", str(job.job_id)], check=False)


def build_jobs(
    routes: list[Path], seeds: list[int], out_root: Path, retries: int
) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for seed in seeds:
        base = out_root / str(seed)
        for directory in (
            base / "res",
            base / "out",
            base / "err",
            base / "slurm_out",
            base / "slurm_err",
            base / "debug",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for route in routes:
            route_id = route.stem.rsplit("_", maxsplit=1)[-1]
            jobs.append(
                EvaluationJob(
                    route=route,
                    route_id=route_id,
                    seed=seed,
                    result_file=base / "res" / f"{route_id}_res.json",
                    stdout_file=base / "slurm_out" / f"{route_id}_slurm.out",
                    stderr_file=base / "slurm_err" / f"{route_id}_slurm.err",
                    evaluator_stdout_file=base / "out" / f"{route_id}_out.log",
                    evaluator_stderr_file=base / "err" / f"{route_id}_err.log",
                    debug_dir=base / "debug" / route_id,
                    attempts_left=retries,
                )
            )
    return jobs


def submission_environment(
    base_env: dict[str, str],
    args: argparse.Namespace,
    job: EvaluationJob,
    route_dir: Path,
    out_root: Path,
    extra_env: dict[str, str],
) -> dict[str, str]:
    env = dict(base_env)
    env.update(extra_env)
    env.update(
        {
            "EVAL_MODE": args.eval_mode,
            "ROUTES": str(route_dir),
            "ROUTE_FILTER": job.route.stem,
            "ROUTE_LIMIT": "1",
            "SEEDS": str(job.seed),
            "RETRIES": "1",
            "OUT_ROOT": str(out_root),
            "FORCE": "1",
            "AGENT_FILE": args.agent_file,
            "AGENT_CONFIG": args.agent_config,
            "EVAL_TRACK": args.track,
            "EVAL_TIMEOUT": str(args.timeout),
        }
    )
    if args.eval_mode == "depth_ttc":
        env["DEPTH_TTC_DEBUG_DIR"] = str(job.debug_dir)
    else:
        env["ALPAMAYO_INPUT_SAVE_DIR"] = str(job.debug_dir / "alpamayo_inputs")
        env["ALPAMAYO_RAW_LOG_PATH"] = str(job.debug_dir / "alpamayo_raw_responses.txt")
    return env


def archive_existing_log(path: Path, previous_job_id: str | None) -> Path | None:
    """Move a previous attempt's log out of the path monitored for fatal errors."""
    if not path.exists():
        return None

    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    job_label = previous_job_id or "unknown"
    archive_path = archive_dir / f"{path.name}.{timestamp}.job-{job_label}"
    sequence = 1
    while archive_path.exists():
        archive_path = archive_dir / (
            f"{path.name}.{timestamp}.job-{job_label}.{sequence}"
        )
        sequence += 1

    path.replace(archive_path)
    return archive_path


def submit_job(
    template: Path,
    args: argparse.Namespace,
    job: EvaluationJob,
    route_dir: Path,
    out_root: Path,
    extra_env: dict[str, str],
) -> str:
    for path in (job.stdout_file, job.stderr_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    for path in (job.evaluator_stdout_file, job.evaluator_stderr_file):
        archived_path = archive_existing_log(path, job.job_id)
        if archived_path is not None:
            print(f"[archive] {path} -> {archived_path}", flush=True)

    env = submission_environment(
        os.environ, args, job, route_dir, out_root, extra_env
    )
    command = [
        "sbatch",
        "--job-name",
        job.name,
        "--output",
        str(job.stdout_file),
        "--error",
        str(job.stderr_file),
        "--export=ALL",
        str(template),
    ]
    output = subprocess.check_output(command, env=env, text=True).strip()
    job_id = output.rsplit(maxsplit=1)[-1]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from: {output!r}")
    return job_id


def dry_run_summary(
    jobs: list[EvaluationJob], args: argparse.Namespace, route_dir: Path, out_root: Path
) -> None:
    print(f"mode={args.eval_mode}")
    print(f"routes={route_dir}")
    print(f"out_root={out_root}")
    print(f"planned_jobs={len(jobs)}")
    print(f"seeds={args.seeds}")
    print(f"template={Path(args.template).resolve()}")
    for job in jobs[:5]:
        print(
            f"  {job.name}: route={job.route.name} seed={job.seed} "
            f"result={job.result_file}"
        )
    if len(jobs) > 5:
        print(f"  ... {len(jobs) - 5} more jobs")


def main() -> int:
    args = parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")

    repo_root = Path(__file__).resolve().parent
    route_dir = Path(args.routes).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(f"Slurm template not found: {template}")

    extra_env = parse_extra_env(args.env)
    routes = route_files(route_dir, args.route_filter, args.route_limit)
    jobs = build_jobs(routes, args.seeds, out_root, args.retries)
    total = len(jobs)

    if args.dry_run:
        dry_run_summary(jobs, args, route_dir, out_root)
        return 0

    validate_mode_dependencies(args, repo_root, extra_env)
    limit = max_parallel_jobs(repo_root, args.max_parallel)
    pending = [job for job in jobs if not result_is_complete(job.result_file)]
    completed = total - len(pending)
    exhausted: list[EvaluationJob] = []
    print(
        f"Evaluation mode={args.eval_mode}: total={total} already_complete={completed} "
        f"pending={len(pending)} max_parallel={limit}",
        flush=True,
    )

    while pending:
        try:
            running = slurm_jobs()
        except RuntimeError as exc:
            print(f"[warn] {exc}; retrying in {args.poll_seconds:.1f}s", flush=True)
            time.sleep(args.poll_seconds)
            continue

        cancel_dead_jobs(pending, running)
        next_pending: list[EvaluationJob] = []
        for job in pending:
            if result_is_complete(job.result_file):
                completed += 1
                print(
                    f"[done {completed}/{total}] seed={job.seed} route={job.route.name}",
                    flush=True,
                )
                continue
            if job.job_id in running:
                next_pending.append(job)
                continue
            if job.attempts_left <= 0:
                exhausted.append(job)
                print(
                    f"[failed] exhausted attempts seed={job.seed} route={job.route.name}",
                    flush=True,
                )
                continue
            next_pending.append(job)
        pending = next_pending
        if not pending:
            break

        running_count = len(running)
        capacity = max(0, limit - running_count)
        for job in pending:
            if capacity <= 0:
                break
            if job.job_id in running or job.attempts_left <= 0:
                continue
            try:
                job.job_id = submit_job(
                    template, args, job, route_dir, out_root, extra_env
                )
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                print(f"[warn] submit failed route={job.route.name}: {exc}", flush=True)
                continue
            job.attempts_left -= 1
            capacity -= 1
            print(
                f"[submit] job={job.job_id} seed={job.seed} route={job.route.name} "
                f"attempts_left={job.attempts_left}",
                flush=True,
            )

        time.sleep(max(0.5, args.poll_seconds))

    print(
        f"Finished: complete={completed}/{total} exhausted={len(exhausted)} "
        f"out_root={out_root}",
        flush=True,
    )
    if exhausted:
        for job in exhausted:
            print(f"  seed={job.seed} route={job.route.name}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

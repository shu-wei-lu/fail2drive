#!/usr/bin/env python3
"""Submit a small, explicitly listed set of Fail2Drive validation jobs.

This is intentionally simpler than ``slurm_evaluate.py``: there is no route
discovery, retry, result inspection, or job monitoring.  Edit ``TESTS`` and
``SBATCH_OPTIONS`` below, then run:

    python slurm_quick_validate.py

Use ``--dry-run`` to only generate and print the sbatch files.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate"

# Edit the Slurm resources here.
SBATCH_OPTIONS = {
    "partition": "devq",
    "qos": "normal",
    "nodes": "1",
    "ntasks": "1",
    "cpus-per-task": "8",
    "mem": "16gb",
    "time": "2:00:00",
    "gres": "gpu:nvidia_geforce_rtx_4090:1",
}

# Edit, add, or remove entries here.  Each command is submitted as one
# independent Slurm job.  Keeping the full command in every entry makes it easy
# to change agent/config/environment variables for just one test.
TESTS = [
#     {
#         "name": "f2d_1021",
#         "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
# python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
#   --agent ./team_code/hipad_f2d_agent.py \
#   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
#   --routes ./fail2drive_split/Generalization_CustomObstacles_1021.xml \
#   --port "$FREE_WORLD_PORT"''',
#     },
#     {
#         "name": "f2d_1022",
#         "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
# python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
#   --agent ./team_code/hipad_f2d_agent.py \
#   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
#   --routes ./fail2drive_split/Generalization_CustomObstacles_1022.xml \
#   --port "$FREE_WORLD_PORT"''',
#     },
#     {
#         "name": "f2d_1056",
#         "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
# python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
#   --agent ./team_code/hipad_f2d_agent.py \
#   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
#   --routes ./fail2drive_split/Generalization_RightOfWay_1056.xml \
#   --port "$FREE_WORLD_PORT"''',
#     },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Base_CustomObstacles_0020.xml \
  --port "$FREE_WORLD_PORT"''',
    },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Generalization_ConstructionPermutations_1019.xml \
  --port "$FREE_WORLD_PORT"''',
    },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Base_PedestrianCrowd_0069.xml \
  --port "$FREE_WORLD_PORT"''',
    },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Generalization_CustomObstacles_1028.xml \
  --port "$FREE_WORLD_PORT"''',
    },
#     {
#         "name": "f2d_test",
#         "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
# python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
#   --agent ./team_code/hipad_f2d_agent.py \
#   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
#   --routes ./fail2drive_split/Generalization_CustomObstacles_1021.xml \
#   --port "$FREE_WORLD_PORT"''',
#     },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Base_PedestrianCrowd_0067.xml \
  --port "$FREE_WORLD_PORT"''',
    },
#     {
#         "name": "f2d_test",
#         "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
# python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
#   --agent ./team_code/hipad_f2d_agent.py \
#   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
#   --routes ./fail2drive_split/Base_ImageOnObject_0040.xml \
#   --port "$FREE_WORLD_PORT"''',
#     },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Base_PedestrianCrowd_0066.xml \
  --port "$FREE_WORLD_PORT"''',
    },
    {
        "name": "f2d_test",
        "command": r'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ./team_code/hipad_f2d_agent.py \
  --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --routes ./fail2drive_split/Generalization_Animals_1079.xml \
  --port "$FREE_WORLD_PORT"''',
    },
    # {
    #     "name": "f2d_left_q2",
    #     "command": r'''ACTIVATION_VECTOR_PATH="$F2D/steering/hipad_new/post_process/left_change_lane_align_q/steering_vector.pt" NO_OTHER_VEHICLES=1 LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1086.xml --port "$FREE_WORLD_PORT"''',
    # },
    # {
    #     "name": "f2d_final_left",
    #     "command": r'''ACTIVATION_VECTOR_PATH="$F2D/steering/hipad_new/post_process/left_change_lane_align_q/steering_vector.pt" NO_OTHER_VEHICLES=1 LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml --port "$FREE_WORLD_PORT"''',
    # },
    # {
    #     "name": "f2d_final_right",
    #     "command": r'''ACTIVATION_ALPHA=2.0 ACTIVATION_VECTOR_PATH="$F2D/steering/hipad_new/post_process/right_align_q/steering_vector.pt" NO_OTHER_VEHICLES=1 LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml --port "$FREE_WORLD_PORT"''',
    # },
    # {
    #     "name": "f2d_final_brake",
    #     "command": r'''ACTIVATION_VECTOR_PATH="$F2D/steering/hipad_new/post_process/brake_align_q/steering_vector.pt" NO_OTHER_VEHICLES=1 LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent ./team_code/hipad_f2d_agent.py   --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml --port "$FREE_WORLD_PORT"''',
    # },
]

def safe_name(name: str) -> str:
    """Return a name safe for paths and Slurm job names."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
    if not value:
        raise ValueError(f"Invalid empty test name derived from {name!r}")
    return value


def make_job_script(test: dict[str, str], test_output: Path) -> str:
    name = safe_name(test["name"])
    directives = "\n".join(
        f"#SBATCH --{key}={value}" for key, value in SBATCH_OPTIONS.items()
    )

    return fr'''#!/bin/bash
#SBATCH --job-name=f2d_val_{name}
{directives}
#SBATCH --output={test_output}/slurm-%j.out
#SBATCH --error={test_output}/slurm-%j.err

set -euo pipefail

REPO_ROOT={shlex.quote(str(REPO_ROOT))}
TEST_OUTPUT={shlex.quote(str(test_output))}
mkdir -p "$TEST_OUTPUT/viz_vehicle"
cd "$REPO_ROOT"

echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: $(hostname)"
echo "Test: {name}"
echo "Output: $TEST_OUTPUT"

: "${{CARLA_ROOT:?CARLA_ROOT is not set. Submit from the activated fail2drive environment.}}"
: "${{HIP:?HIP is not set. Export HIP before submitting.}}"

free_port() {{
  local start=$1
  comm -23 <(seq "$start" "$((start + 49))" | sort) \
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) \
    | shuf | head -n 1
}}

FREE_WORLD_PORT=$(free_port 10000)
FREE_STREAMING_PORT=$(free_port 20000)
echo "World port: $FREE_WORLD_PORT"
echo "Streaming port: $FREE_STREAMING_PORT"

CARLA_LOG="$TEST_OUTPUT/carla.log"
"$CARLA_ROOT/CarlaUE4.sh" \
  -carla-rpc-port="$FREE_WORLD_PORT" \
  -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 \
  -carla-streaming-port="$FREE_STREAMING_PORT" \
  >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!

cleanup() {{
  kill "$CARLA_PID" 2>/dev/null || true
  wait "$CARLA_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

sleep 60
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "CARLA exited before validation started. Last log lines:" >&2
  tail -100 "$CARLA_LOG" >&2 || true
  exit 1
fi

{test["command"]}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without calling sbatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT_ROOT / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # seen_names: set[str] = set()
    for test in TESTS:
        name = safe_name(test["name"])
        # if name in seen_names:
        #     raise ValueError(f"Duplicate test name: {name}")
        # seen_names.add(name)

        test_output = OUTPUT_ROOT / name
        test_output.mkdir(parents=True, exist_ok=True)
        job_file = run_dir / f"{name}.sh"
        job_file.write_text(make_job_script(test, test_output), encoding="utf-8")
        job_file.chmod(0o755)

        if args.dry_run:
            print(f"[dry-run] {job_file}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"{name}: {result.stdout.strip()}")


if __name__ == "__main__":
    main()

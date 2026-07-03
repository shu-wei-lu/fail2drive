#!/usr/bin/env python3
"""Collect steering features for all steering route splits.

This is a small wrapper around create_steering_features.py. It runs the four
standard steering splits into sibling output directories:

  <output-root>/Normal
  <output-root>/Brake
  <output-root>/LaneChangeLeft
  <output-root>/LaneChangeRight

Any unknown arguments are forwarded to create_steering_features.py, so options
like --max-frames, --route-filter, --save-visual-output, --env, --host, and
--port can still be used from this wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = ("Normal", "Brake", "LaneChangeLeft", "LaneChangeRight")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(
      description="Run create_steering_features.py for all standard steering splits.",
  )
  parser.add_argument("--agent-file", required=True, help="Agent entry file.")
  parser.add_argument("--agent-config", required=True, help="Agent checkpoint/config path.")
  parser.add_argument(
      "--output-root",
      required=True,
      help="Parent output directory. Each split is written to <output-root>/<split>.",
  )
  parser.add_argument(
      "--steering-split-root",
      default=str(ROOT / "steering_split"),
      help="Directory containing Normal/Brake/LaneChangeLeft/LaneChangeRight route folders.",
  )
  parser.add_argument(
      "--feature-script",
      default=str(ROOT / "create_steering_features.py"),
      help="Path to create_steering_features.py.",
  )
  parser.add_argument(
      "--splits",
      nargs="+",
      default=list(DEFAULT_SPLITS),
      help="Split names to run. Defaults to all four standard splits.",
  )
  parser.add_argument(
      "--keep-going",
      action="store_true",
      help="Continue running remaining splits after a split fails.",
  )
  args, forwarded_args = parser.parse_known_args()
  return args, forwarded_args


def require_split_dirs(split_root: Path, splits: list[str]) -> None:
  missing = [split for split in splits if not (split_root / split).is_dir()]
  if missing:
    missing_text = ", ".join(missing)
    raise FileNotFoundError(f"Missing steering split directory/directories: {missing_text}")


def run_split(args: argparse.Namespace, forwarded_args: list[str], split: str, index: int, total: int) -> int:
  routes = Path(args.steering_split_root) / split
  output_root = Path(args.output_root) / split
  command = [
      sys.executable,
      "-u",
      args.feature_script,
      "--routes",
      str(routes),
      "--agent-file",
      args.agent_file,
      "--agent-config",
      args.agent_config,
      "--output-root",
      str(output_root),
      *forwarded_args,
  ]

  print(f"[{index}/{total}] Collecting {split} -> {output_root}", flush=True)
  print(" ".join(command), flush=True)
  return subprocess.run(command, check=False).returncode


def main() -> int:
  args, forwarded_args = parse_args()
  split_root = Path(args.steering_split_root)
  splits = list(args.splits)
  require_split_dirs(split_root, splits)

  failures: list[tuple[str, int]] = []
  for index, split in enumerate(splits, start=1):
    return_code = run_split(args, forwarded_args, split, index, len(splits))
    if return_code != 0:
      failures.append((split, return_code))
      print(f"[warn] {split} exited with return code {return_code}", flush=True)
      if not args.keep_going:
        break

  if failures:
    failed_text = ", ".join(f"{split}={return_code}" for split, return_code in failures)
    print(f"Failed split(s): {failed_text}", file=sys.stderr)
    return 1

  print("Finished all requested steering splits.", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

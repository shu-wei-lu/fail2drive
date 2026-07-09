#!/usr/bin/env python3
"""Run routes with activation feature logging enabled.

This is a thin collection wrapper around the local leaderboard evaluator. It
keeps feature collection separate from evaluator code and writes two parallel
trees:

  output/logs/<route_run>/activation_actions.jsonl
  output/features/<route_run>/<frame>.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--routes', required=True, help='Route XML file or directory of route XML files.')
  parser.add_argument('--agent-file', required=True, help='Agent entry file, e.g. team_code/sensor_agent.py.')
  parser.add_argument('--agent-config', required=True, help='Agent checkpoint/config directory.')
  parser.add_argument('--output-root', default='results/steering_features')
  parser.add_argument('--lb-script', default='leaderboard/leaderboard/leaderboard_evaluator_local.py')
  parser.add_argument('--track', default='SENSORS')
  parser.add_argument('--host', default='localhost')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--timeout', type=float, default=300.0)
  parser.add_argument('--traffic-manager-seed', type=int, default=0)
  parser.add_argument('--repetitions', type=int, default=1)
  parser.add_argument('--route-limit', type=int, default=0)
  parser.add_argument('--route-filter', default='')
  parser.add_argument(
      '--max-frames',
      type=int,
      default=0,
      help='Stop each route/repetition run after this many evaluator frames. 0 means no limit.',
  )
  parser.add_argument(
      '--save-visual-output',
      action='store_true',
      help='Save per-frame model debug/pygame-style images under <output-root>/images.',
  )
  parser.add_argument(
      '--live-visual-output',
      action='store_true',
      help='Open the pygame live visualizer while collecting features.',
  )
  parser.add_argument(
      '--no-other-vehicles',
      action='store_true',
      help='Set NO_OTHER_VEHICLES=1 for the evaluator to disable background and parked vehicles.',
  )
  parser.add_argument(
      '--env',
      action='append',
      default=[],
      metavar='KEY=VALUE',
      help='Extra environment override passed to the agent. Can be repeated.',
  )
  args = parser.parse_args()
  if args.max_frames < 0:
    parser.error('--max-frames must be >= 0')
  return args


def list_routes(routes_arg: str, route_filter: str, route_limit: int) -> list[Path]:
  route_path = Path(routes_arg)
  if route_path.is_file():
    routes = [route_path]
  else:
    routes = sorted(route_path.glob('*.xml'))

  if route_filter:
    routes = [route for route in routes if route_filter in route.name]
  if route_limit > 0:
    routes = routes[:route_limit]
  if not routes:
    raise RuntimeError(f'No route XML files found for {routes_arg}')
  return routes


def parse_env_overrides(items: list[str]) -> dict[str, str]:
  result = {}
  for item in items:
    if '=' not in item:
      raise ValueError(f'Expected KEY=VALUE for --env, got: {item}')
    key, value = item.split('=', maxsplit=1)
    result[key] = value
  return result


def run_route(args: argparse.Namespace, route: Path, output_root: Path, env: dict[str, str], index: int) -> int:
  route_run_id = route.stem
  result_dir = output_root / 'results'
  stdout_dir = output_root / 'stdout'
  stderr_dir = output_root / 'stderr'
  debug_dir = output_root / 'debug'
  for path in (result_dir, stdout_dir, stderr_dir, debug_dir):
    path.mkdir(parents=True, exist_ok=True)

  command = [
      sys.executable,
      '-u',
      args.lb_script,
      f'--routes={route}',
      f'--repetitions={args.repetitions}',
      f'--track={args.track}',
      f'--checkpoint={result_dir / f"{route_run_id}_res.json"}',
      f'--debug-checkpoint={debug_dir / f"{route_run_id}_live.txt"}',
      f'--timeout={args.timeout}',
      f'--agent={args.agent_file}',
      f'--agent-config={args.agent_config}',
      f'--host={args.host}',
      f'--port={args.port}',
      f'--traffic-manager-seed={args.traffic_manager_seed + index}',
  ]
  if args.max_frames > 0:
    command.append(f'--max-frames={args.max_frames}')

  with (stdout_dir / f'{route_run_id}.log').open('a', encoding='utf-8') as stdout, \
      (stderr_dir / f'{route_run_id}.log').open('a', encoding='utf-8') as stderr:
    stdout.write('[create_steering_features] ' + ' '.join(command) + '\n')
    stdout.write(f'[create_steering_features] NO_OTHER_VEHICLES={env.get("NO_OTHER_VEHICLES", "")}\n')
    stdout.flush()
    return subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False).returncode


def main() -> int:
  args = parse_args()
  output_root = Path(args.output_root)
  logs_root = output_root / 'logs'
  features_root = output_root / 'features'
  images_root = output_root / 'images'
  logs_root.mkdir(parents=True, exist_ok=True)
  features_root.mkdir(parents=True, exist_ok=True)

  env = os.environ.copy()
  env.update({
      'SAVE_PATH': str(logs_root),
      'SAVE_FUSED_FEATURES': 'True',
      'FUSED_FEATURES_PATH': str(features_root),
  })
  if args.save_visual_output:
    env['DEBUG_CHALLENGE'] = '1'
    env['VISUAL_OUTPUT_PATH'] = str(images_root)
    images_root.mkdir(parents=True, exist_ok=True)
  if args.live_visual_output:
    env['LIVE_VISU'] = 'True'
  if args.no_other_vehicles:
    env['NO_OTHER_VEHICLES'] = '1'
  env.update(parse_env_overrides(args.env))

  routes = list_routes(args.routes, args.route_filter, args.route_limit)
  print(f'Collecting activation features for {len(routes)} route(s).')
  print(f'Logs: {logs_root}')
  print(f'Features: {features_root}')
  if args.save_visual_output:
    print(f'Visual output: {images_root}')
  if args.live_visual_output:
    print('Live pygame visualizer: enabled via LIVE_VISU=True')
  if env.get('NO_OTHER_VEHICLES', '0').lower() in ('1', 'true', 'yes', 'y'):
    print('Other vehicles: disabled via NO_OTHER_VEHICLES=1')
  if args.max_frames > 0:
    print(f'Max frames per run: {args.max_frames}')

  failed = 0
  for index, route in enumerate(routes, start=1):
    print(f'[{index}/{len(routes)}] {route}')
    return_code = run_route(args, route, output_root, env, index)
    if return_code != 0:
      failed += 1
      print(f'[warn] route exited with return code {return_code}: {route}')

  return 1 if failed else 0


if __name__ == '__main__':
  raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrate the steering feature alpha that forces braking on a clean route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / 'team_code'))
from activation_steering.registry import get_adapter


def parse_args() -> argparse.Namespace:
  base_parser = argparse.ArgumentParser(add_help=False)
  base_parser.add_argument('--adapter', default='transfuser_target_speed')
  known, _ = base_parser.parse_known_args()
  adapter = get_adapter(known.adapter)

  parser = argparse.ArgumentParser(parents=[base_parser])
  parser.add_argument('--routes', default='steering_split/normal/Steering_Normal_1085.xml')
  parser.add_argument('--agent-file', required=True, help='Agent entry file, e.g. team_code/sensor_agent.py.')
  parser.add_argument('--agent-config', required=True, help='Agent checkpoint/config directory.')
  parser.add_argument('--output-root', default='results/steering_alpha_calibration')
  parser.add_argument(
      '--activation-vector-path',
      default=None,
      help='Path to brake_minus_normal.pt. Defaults to the agent/model default.',
  )
  parser.add_argument('--lb-script', default='leaderboard/leaderboard/leaderboard_evaluator_local.py')
  parser.add_argument('--track', default='SENSORS')
  parser.add_argument('--host', default='localhost')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--timeout', type=float, default=300.0)
  parser.add_argument('--traffic-manager-seed', type=int, default=0)
  parser.add_argument('--starting-alpha', type=float, default=0.5)
  parser.add_argument('--alpha-interval', type=float, default=0.5)
  parser.add_argument('--max-alpha', type=float, default=20.0)
  parser.add_argument('--start-steering-frame', type=int, default=100)
  parser.add_argument(
      '--fixed-eval-frames',
      type=int,
      default=120,
      help='Stop each run after this many frames at/after --start-steering-frame. Use 0 to run to route end.',
  )
  parser.add_argument('--min-eval-frames', type=int, default=20)
  parser.add_argument('--poll-interval', type=float, default=1.0)
  parser.add_argument('--stop-timeout', type=float, default=20.0)
  parser.add_argument(
      '--live-visu',
      action='store_true',
      help='Set LIVE_VISU=True for the agent subprocess to open the pygame debug window.',
  )
  parser.add_argument(
      '--binary-refine-steps',
      type=int,
      default=0,
      help='After the first successful linear alpha, run this many binary refinement steps.',
  )
  parser.add_argument(
      '--keep-other-vehicles',
      action='store_true',
      help='Do not set NO_OTHER_VEHICLES=1.',
  )
  parser.add_argument(
      '--keep-stop-control',
      action='store_true',
      help='Do not set STOP_CONTROL=0.',
  )
  parser.add_argument(
      '--normalize',
      action='store_true',
      help='Normalize the activation vector.',
  )
  parser.add_argument(
      '--env',
      action='append',
      default=[],
      metavar='KEY=VALUE',
      help='Extra environment override passed to the agent. Can be repeated.',
  )
  adapter.add_calibration_args(parser)
  return parser.parse_args()


def parse_env_overrides(items: list[str]) -> dict[str, str]:
  result = {}
  for item in items:
    if '=' not in item:
      raise ValueError(f'Expected KEY=VALUE for --env, got: {item}')
    key, value = item.split('=', maxsplit=1)
    result[key] = value
  return result


def alpha_slug(alpha: float) -> str:
  return f'{alpha:.6f}'.rstrip('0').rstrip('.').replace('.', 'p')


def read_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open('r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def latest_activation_log(run_dir: Path) -> Path:
  logs = sorted(
      run_dir.rglob('activation_actions.jsonl'),
      key=lambda path: path.stat().st_mtime,
      reverse=True,
  )
  if not logs:
    raise FileNotFoundError(f'No activation_actions.jsonl found under {run_dir}')
  return logs[0]


def eval_rows(log_path: Path, args: argparse.Namespace) -> list[dict]:
  rows = [
      row for row in read_jsonl(log_path)
      if int(row.get('frame', -1)) >= args.start_steering_frame
  ]
  if args.fixed_eval_frames > 0:
    rows = rows[:args.fixed_eval_frames]
  return rows


def analyze_log(log_path: Path, args: argparse.Namespace, adapter, baseline_rows: list[dict] | None = None) -> dict:
  rows = eval_rows(log_path, args)
  total = len(rows)

  forced = []
  for index, row in enumerate(rows):
    baseline_row = baseline_rows[index] if baseline_rows is not None and index < len(baseline_rows) else None
    if adapter.is_forced_frame(row, args, baseline_row):
      forced.append(row)

  metrics = [adapter.calibration_metric(row) for row in rows]
  baseline_metrics = [adapter.calibration_metric(row) for row in (baseline_rows or [])[:total]]
  ratio = len(forced) / total if total else 0.0
  return {
      'activation_log': str(log_path),
      'eval_frames': total,
      'forced_frames': len(forced),
      'forced_ratio': ratio,
      'success': total >= args.min_eval_frames and ratio >= args.success_ratio,
      'first_eval_frame': int(rows[0]['frame']) if rows else None,
      'last_eval_frame': int(rows[-1]['frame']) if rows else None,
      'mean_metric': sum(metrics) / total if total else None,
      'min_metric': min(metrics) if metrics else None,
      'max_metric': max(metrics) if metrics else None,
      'mean_baseline_metric': (
          sum(baseline_metrics) / len(baseline_metrics) if baseline_metrics else None
      ),
  }


def has_enough_eval_frames(logs_dir: Path, args: argparse.Namespace) -> bool:
  try:
    log_path = latest_activation_log(logs_dir)
    return len(eval_rows(log_path, args)) >= args.fixed_eval_frames
  except Exception:
    return False


def wait_for_fixed_frames(process: subprocess.Popen, logs_dir: Path, args: argparse.Namespace) -> int:
  if args.fixed_eval_frames <= 0:
    return process.wait()

  while process.poll() is None:
    if has_enough_eval_frames(logs_dir, args):
      process.send_signal(signal.SIGINT)
      try:
        return process.wait(timeout=args.stop_timeout)
      except subprocess.TimeoutExpired:
        process.terminate()
        try:
          return process.wait(timeout=args.stop_timeout)
        except subprocess.TimeoutExpired:
          process.kill()
          return process.wait()
    time.sleep(args.poll_interval)

  return process.returncode


def run_alpha(
    args: argparse.Namespace,
    adapter,
    alpha: float,
    attempt_index: int,
    baseline_rows: list[dict] | None = None,
) -> dict:
  output_root = Path(args.output_root)
  timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
  run_dir = output_root / 'runs' / f'{attempt_index:03d}_alpha_{alpha_slug(alpha)}_{timestamp}'
  result_dir = run_dir / 'results'
  stdout_dir = run_dir / 'stdout'
  stderr_dir = run_dir / 'stderr'
  logs_dir = run_dir / 'logs'
  debug_dir = run_dir / 'debug'
  for path in (result_dir, stdout_dir, stderr_dir, logs_dir, debug_dir):
    path.mkdir(parents=True, exist_ok=True)

  env = os.environ.copy()
  env.update({
      'SAVE_PATH': str(logs_dir),
      'STEERING_ALPHA': str(alpha),
      'START_STEERING_FRAME': str(args.start_steering_frame),
      'NORMALIZE_STEERING_VECTOR': '1' if args.normalize else '0',
  })
  if args.activation_vector_path:
    env['ACTIVATION_VECTOR_PATH'] = str(args.activation_vector_path)
  if args.live_visu:
    env['LIVE_VISU'] = 'True'
  if not args.keep_other_vehicles:
    env['NO_OTHER_VEHICLES'] = '1'
  if not args.keep_stop_control:
    env['STOP_CONTROL'] = '0'
  env.update(parse_env_overrides(args.env))

  command = [
      sys.executable,
      '-u',
      args.lb_script,
      f'--routes={args.routes}',
      '--repetitions=1',
      f'--track={args.track}',
      f'--checkpoint={result_dir / "result.json"}',
      f'--debug-checkpoint={debug_dir / "live.txt"}',
      f'--timeout={args.timeout}',
      f'--agent={args.agent_file}',
      f'--agent-config={args.agent_config}',
      f'--host={args.host}',
      f'--port={args.port}',
      f'--traffic-manager-seed={args.traffic_manager_seed}',
  ]

  out_file = stdout_dir / 'run.log'
  err_file = stderr_dir / 'run.log'
  with out_file.open('w', encoding='utf-8') as stdout, err_file.open('w', encoding='utf-8') as stderr:
    stdout.write('[calibrate_steering_alpha] ' + ' '.join(command) + '\n')
    stdout.write(f'[calibrate_steering_alpha] alpha={alpha}\n')
    stdout.flush()
    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
    return_code = wait_for_fixed_frames(process, logs_dir, args)

  result = {
      'alpha': alpha,
      'return_code': return_code,
      'run_dir': str(run_dir),
      'stdout': str(out_file),
      'stderr': str(err_file),
  }
  try:
    result.update(analyze_log(latest_activation_log(logs_dir), args, adapter, baseline_rows))
  except Exception as exc:
    result.update({
        'eval_frames': 0,
        'forced_frames': 0,
        'forced_ratio': 0.0,
        'success': False,
        'analysis_error': str(exc),
    })
  return result


def write_summary(args: argparse.Namespace, attempts: list[dict], alpha_max: float | None) -> None:
  output_root = Path(args.output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  summary = {
      'adapter': args.adapter,
      'alpha_max': alpha_max,
      'success_ratio': args.success_ratio,
      'success_metric': getattr(args, 'success_metric', None),
      'start_steering_frame': args.start_steering_frame,
      'fixed_eval_frames': args.fixed_eval_frames,
      'args': vars(args),
      'attempts': attempts,
  }
  summary_path = output_root / 'summary.json'
  summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main() -> int:
  args = parse_args()
  adapter = get_adapter(args.adapter)
  attempts = []
  attempt_index = 1
  alpha = args.starting_alpha
  previous_failed_alpha = 0.0
  first_success = None
  baseline_rows = None

  print('CARLA must already be running and reachable at ' f'{args.host}:{args.port}')
  print(f'Route: {args.routes}')

  if getattr(args, 'success_metric', None) in ('target_speed_drop', 'target_speed_ratio'):
    print(f'[baseline] alpha={args.baseline_alpha:.6g}')
    baseline_result = run_alpha(args, adapter, args.baseline_alpha, 0)
    attempts.append({'baseline': True, **baseline_result})
    try:
      baseline_rows = eval_rows(Path(baseline_result['activation_log']), args)
    except Exception as exc:
      print(f'[fail] could not read baseline rows: {exc}')
      write_summary(args, attempts, None)
      return 1

  while alpha <= args.max_alpha + 1e-9:
    print(f'[{attempt_index}] alpha={alpha:.6g}')
    result = run_alpha(args, adapter, alpha, attempt_index, baseline_rows)
    attempts.append(result)
    print(
        f"    forced={result['forced_frames']}/{result['eval_frames']} "
        f"ratio={result['forced_ratio']:.3f} "
        f"mean_metric={result.get('mean_metric')} success={result['success']}"
    )
    if result['success']:
      first_success = alpha
      break
    previous_failed_alpha = alpha
    alpha += args.alpha_interval
    attempt_index += 1

  alpha_max = first_success
  if first_success is not None and args.binary_refine_steps > 0:
    low = previous_failed_alpha
    high = first_success
    for _ in range(args.binary_refine_steps):
      attempt_index += 1
      mid = (low + high) / 2.0
      print(f'[{attempt_index}] refine alpha={mid:.6g}')
      result = run_alpha(args, adapter, mid, attempt_index, baseline_rows)
      attempts.append(result)
      print(
          f"    forced={result['forced_frames']}/{result['eval_frames']} "
          f"ratio={result['forced_ratio']:.3f} "
          f"mean_metric={result.get('mean_metric')} success={result['success']}"
      )
      if result['success']:
        high = mid
      else:
        low = mid
    alpha_max = high

  write_summary(args, attempts, alpha_max)
  if alpha_max is None:
    print(f'No alpha reached the criterion up to max-alpha={args.max_alpha}.')
    return 1

  print(f'alpha_max={alpha_max:.6g}')
  print(f'Summary: {Path(args.output_root) / "summary.json"}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

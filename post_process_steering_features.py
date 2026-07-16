#!/usr/bin/env python3
"""Build action-vs-normal activation vectors from collected features."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'team_code'))
from activation_steering.registry import get_adapter


ACTION_ALIASES = {
    'brake': 'brake',
    'stop': 'brake',
    'left': 'left_change_lane',
    'left_change_lane': 'left_change_lane',
    'left-change-lane': 'left_change_lane',
    'left change lane': 'left_change_lane',
    'left_lane_change': 'left_change_lane',
    'right': 'right_change_lane',
    'right_change_lane': 'right_change_lane',
    'right-change-lane': 'right_change_lane',
    'right change lane': 'right_change_lane',
    'right_lane_change': 'right_change_lane',
}


def parse_args() -> argparse.Namespace:
  base_parser = argparse.ArgumentParser(add_help=False)
  base_parser.add_argument('--adapter', default='transfuser_target_speed')
  known, _ = base_parser.parse_known_args()
  adapter = get_adapter(known.adapter)

  parser = argparse.ArgumentParser(parents=[base_parser])
  parser.add_argument('--collection-root', default='results/steering_features')
  parser.add_argument('--logs-root', default=None, help='Defaults to <collection-root>/logs.')
  parser.add_argument('--features-root', default=None, help='Defaults to <collection-root>/features.')
  parser.add_argument(
      '--feature-name',
      '--feature_name',
      default=None,
      help=(
          'Optional nested feature name under each run, e.g. align_query for '
          '<features-root>/<run>/<feature-name>/<layer-name>/<frame>.pt.'
      ),
  )
  parser.add_argument(
      '--layer-name',
      '--layer_name',
      default=None,
      help=(
          'Optional nested layer name under each feature name, e.g. layer_02 for '
          '<features-root>/<run>/<feature-name>/<layer-name>/<frame>.pt.'
      ),
  )
  parser.add_argument('--output-dir', default=None, help='Defaults to <collection-root>/post_process.')
  parser.add_argument(
      '--folder-name',
      '--folder_name',
      default=None,
      help='Deprecated: subfolder under the default output root when --output-dir is not set.',
  )
  parser.add_argument(
      '--action',
      type=normalize_action,
      choices=('brake', 'left_change_lane', 'right_change_lane'),
      default='brake',
      help='Target action to use as the negative class: brake, left_change_lane, or right_change_lane.',
  )
  parser.add_argument('--model-index', type=int, default=0, help='Feature subdir to use for ensembles.')
  parser.add_argument('--max-frames-per-class', type=int, default=0)
  parser.add_argument('--flatten', action='store_true', help='Flatten each feature tensor before averaging.')
  parser.add_argument(
      '--manual',
      action='store_true',
      help=(
          'Use manually picked positive frames from '
          '<collection-root>/<ActionDir>/picked_frames.json. Negative selection is unchanged.'
      ),
  )
  parser.add_argument('--steer-threshold', type=float, default=0.2)
  parser.add_argument('--normal-max-abs-steer', type=float, default=0.05)
  parser.add_argument(
      '--positive-include-pattern',
      action='append',
      default=[],
      help='Only use positive/target-action frames from run names or log paths containing this substring. Can be repeated.',
  )
  parser.add_argument(
      '--positive-exclude-pattern',
      action='append',
      default=[],
      help='Exclude positive/target-action frames from run names or log paths containing this substring. Can be repeated.',
  )
  parser.add_argument(
      '--negative-include-pattern',
      action='append',
      default=[],
      help='Only use negative/normal frames from run names or log paths containing this substring. Can be repeated.',
  )
  parser.add_argument(
      '--negative-exclude-pattern',
      action='append',
      default=[],
      help='Exclude negative/normal frames from run names or log paths containing this substring. Can be repeated.',
  )
  adapter.add_post_process_args(parser)
  return parser.parse_args()


def normalize_action(action: str) -> str:
  key = action.strip().lower().replace(' ', '_')
  normalized = ACTION_ALIASES.get(action.strip().lower(), ACTION_ALIASES.get(key))
  if normalized is None:
    options = ', '.join(('brake', 'left_change_lane', 'right_change_lane'))
    raise argparse.ArgumentTypeError(f"Unsupported action '{action}'. Options: {options}")
  return normalized


def matches_patterns(name: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
  if include_patterns and not any(pattern in name for pattern in include_patterns):
    return False
  if any(pattern in name for pattern in exclude_patterns):
    return False
  return True


def action_dir_name(action: str) -> str:
  if action == 'brake':
    return 'Brake'
  if action == 'left_change_lane':
    return 'LaneChangeLeft'
  if action == 'right_change_lane':
    return 'LaneChangeRight'
  return action


def manual_action_dir_candidates(action: str) -> list[str]:
  candidates = [action_dir_name(action)]
  if action == 'left_change_lane':
    candidates.append('Left')
  elif action == 'right_change_lane':
    candidates.append('Right')
  return candidates


def manual_positive_frame_path(
    collection_root: Path,
    action: str,
    include_patterns: list[str],
) -> Path:
  candidates: list[Path] = []
  for dirname in manual_action_dir_candidates(action):
    candidates.append(collection_root / dirname / 'picked_frames.json')

  for pattern in include_patterns:
    candidates.append(collection_root / pattern / 'picked_frames.json')

  for path in candidates:
    if path.exists():
      return path

  matching_paths = []
  for path in sorted(collection_root.glob('*/picked_frames.json')):
    match_context = f'{path.parent.name}\n{path.parent}'
    if matches_patterns(match_context, include_patterns, []):
      matching_paths.append(path)

  if len(matching_paths) == 1:
    return matching_paths[0]
  if len(matching_paths) > 1:
    shown = ', '.join(str(path) for path in matching_paths)
    raise ValueError(
        f'--manual matched multiple picked_frames.json files: {shown}. '
        'Use a more specific --positive-include-pattern.')

  searched = ', '.join(str(path) for path in candidates)
  raise FileNotFoundError(f'--manual expected picked frames at one of: {searched}')


def load_manual_positive_frames(
    collection_root: Path,
    action: str,
    include_patterns: list[str],
) -> tuple[dict[str, set[int]], Path]:
  path = manual_positive_frame_path(collection_root, action, include_patterns)

  with path.open('r', encoding='utf-8') as f:
    raw = json.load(f)
  if not isinstance(raw, dict):
    raise ValueError(f'--manual picked frames must be a JSON object: {path}')

  picked: dict[str, set[int]] = {}
  for run_name, frames in raw.items():
    if not isinstance(run_name, str) or not isinstance(frames, list):
      raise ValueError(f'--manual expected Dict[str, List[int]] in {path}')
    picked[run_name] = {int(frame) for frame in frames}
  return picked, path


def read_jsonl(path: Path) -> Iterable[dict]:
  with path.open('r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def read_measurements(path: Path) -> Iterable[dict]:
  for measurement in sorted(path.glob('measurements/*.json.gz')):
    frame = int(measurement.stem.split('.')[0])
    with gzip.open(measurement, 'rt', encoding='utf-8') as f:
      row = json.load(f)
    row['frame'] = frame
    yield row


def read_hipad_metas(path: Path) -> Iterable[dict]:
  for meta_path in sorted(path.glob('*.json')):
    with meta_path.open('r', encoding='utf-8') as f:
      meta = json.load(f)
    row = dict(meta)
    row['frame'] = int(meta_path.stem)
    row['_hipad_meta'] = meta
    row['_hipad_meta_path'] = str(meta_path)
    yield row


def action_logs(logs_root: Path) -> list[Path]:
  logs = sorted(logs_root.rglob('activation_actions.jsonl'))
  if logs:
    return logs
  return sorted({path.parent for path in logs_root.rglob('measurements/*.json.gz')})


def post_process_sources(logs_root: Path, collection_root: Path, adapter_name: str) -> list[Path]:
  sources = action_logs(logs_root)
  if adapter_name not in ('hipad', 'hipad_plan'):
    return sources

  existing_runs = {run_name_for(source) for source in sources}
  meta_dirs = sorted(
      path for path in collection_root.rglob('metas')
      if path.is_dir() and path.parent.parent.name == 'images'
  )
  sources.extend(path for path in meta_dirs if run_name_for(path) not in existing_runs)
  return sources


def run_name_for(log_source: Path) -> str:
  if log_source.name == 'metas' and log_source.parent.parent.name == 'images':
    return log_source.parent.name
  return log_source.parent.name if log_source.name == 'activation_actions.jsonl' else log_source.name


def match_context_for(log_source: Path, run_name: str) -> str:
  return f'{run_name}\n{log_source}\n{log_source.parent}'


def feature_search_root(feature_dir: Path, model_index: int) -> Path:
  model_dir = feature_dir / f'model_{model_index:02d}'
  if model_dir.exists():
    return model_dir
  return feature_dir


def feature_file_in_dir(feature_dir: Path, frame: int, model_index: int) -> Path:
  return feature_search_root(feature_dir, model_index) / f'{frame:06d}.pt'


def nested_feature_files_in_dir(
    feature_dir: Path,
    frame: int,
    model_index: int,
    feature_name: str | None,
    layer_name: str | None,
) -> list[Path]:
  search_root = feature_search_root(feature_dir, model_index)
  filename = f'{frame:06d}.pt'

  if feature_name and layer_name:
    return [search_root / feature_name / layer_name / filename]
  if feature_name:
    return sorted(path for path in (search_root / feature_name).glob(f'*/{filename}') if path.is_file())
  if layer_name:
    return sorted(path for path in search_root.glob(f'*/{layer_name}/{filename}') if path.is_file())
  return sorted(path for path in search_root.glob(f'*/*/{filename}') if path.is_file())


def feature_path_in_run_dir(
    feature_dir: Path,
    frame: int,
    model_index: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
  path = feature_file_in_dir(feature_dir, frame, model_index)
  if path.exists():
    return path

  candidates = [path for path in nested_feature_files_in_dir(
      feature_dir, frame, model_index, feature_name, layer_name) if path.exists()]
  if len(candidates) == 1:
    return candidates[0]
  if len(candidates) > 1:
    shown = ', '.join(str(path) for path in candidates[:5])
    suffix = '' if len(candidates) <= 5 else f', ... ({len(candidates)} total)'
    raise ValueError(
        f'Multiple feature files found for frame {frame:06d} under {feature_dir}: '
        f'{shown}{suffix}. Pass --feature-name and --layer-name to choose one.')
  return None


def sibling_feature_path_for(
    log_source: Path,
    run_name: str,
    model_index: int,
    frame: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
  if log_source.name == 'metas' and log_source.parent.parent.name == 'images':
    return feature_path_in_run_dir(
        log_source.parent.parent.parent / 'features' / run_name,
        frame,
        model_index,
        feature_name,
        layer_name,
    )

  log_dir = log_source.parent if log_source.name == 'activation_actions.jsonl' else log_source
  if log_dir.parent.name != 'logs':
    return None
  return feature_path_in_run_dir(
      log_dir.parent.parent / 'features' / run_name, frame, model_index, feature_name, layer_name)


def nested_feature_path_for(
    features_root: Path,
    run_name: str,
    model_index: int,
    frame: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
  for feature_dir in sorted(features_root.rglob(run_name)):
    if not feature_dir.is_dir():
      continue
    path = feature_path_in_run_dir(feature_dir, frame, model_index, feature_name, layer_name)
    if path is not None:
      return path
  return None


def feature_path_for(row: dict, log_source: Path, logs_root: Path, features_root: Path, args: argparse.Namespace, adapter) -> Path:
  path = adapter.feature_path_for(row, log_source, features_root, args.model_index)
  if path.exists():
    return path

  run_name = run_name_for(log_source)
  frame = int(row['frame'])
  sibling_path = sibling_feature_path_for(
      log_source, run_name, args.model_index, frame, args.feature_name, args.layer_name)
  if sibling_path is not None:
    return sibling_path

  nested_path = nested_feature_path_for(
      features_root, run_name, args.model_index, frame, args.feature_name, args.layer_name)
  if nested_path is not None:
    return nested_path

  return path


def default_child_or_root(root: Path, child_name: str) -> Path:
  child = root / child_name
  return child if child.exists() else root


def add_feature(accumulator: dict, label: str, feature: torch.Tensor, max_count: int) -> bool:
  if max_count > 0 and accumulator[f'{label}_count'] >= max_count:
    return False

  feature = feature.detach().cpu().float()
  if accumulator['flatten']:
    feature = feature.reshape(-1)

  key = f'{label}_sum'
  if accumulator[key] is None:
    accumulator[key] = torch.zeros_like(feature)
  accumulator[key] += feature
  accumulator[f'{label}_count'] += 1
  return True


def basic_frame_allowed(row: dict, args: argparse.Namespace) -> bool:
  speed = float(row.get('speed', 0.0))
  stop_for_stop_sign = bool(row.get('stop_for_stop_sign', row.get('stop_sign_hazard', False)))
  if args.exclude_stop_sign and stop_for_stop_sign:
    return False
  return speed >= args.min_speed


def classify_normal(row: dict, match_context: str, args: argparse.Namespace) -> str | None:
  if not matches_patterns(match_context, args.negative_include_pattern, args.negative_exclude_pattern):
    return None
  if not matches_patterns(match_context, args.normal_include_pattern, args.normal_exclude_pattern):
    return None
  if not basic_frame_allowed(row, args):
    return None

  steer = float(row.get('steer', 0.0))
  throttle = float(row.get('throttle', 0.0))
  brake = float(row.get('brake', row.get('control_brake', 0.0)))
  if abs(steer) <= args.normal_max_abs_steer and brake < args.brake_threshold and throttle >= args.normal_throttle_threshold:
    return 'negative'
  return None


def classify_target_action(row: dict, run_name: str, match_context: str, args: argparse.Namespace, adapter) -> str | None:
  if not matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
    return None
  if not basic_frame_allowed(row, args):
    return None

  if args.action == 'brake':
    if not matches_patterns(match_context, args.brake_include_pattern, args.brake_exclude_pattern):
      return None
    if adapter.classify_frame(row, run_name, args) == 'brake':
      return 'positive'
    return None

  steer = float(row.get('steer', 0.0))
  brake = float(row.get('brake', row.get('control_brake', 0.0)))
  if brake >= args.brake_threshold:
    return None
  if args.action == 'right_change_lane' and steer > args.steer_threshold:
    return 'positive'
  if args.action == 'left_change_lane' and steer < -args.steer_threshold:
    return 'positive'
  return None


def classify_frame(row: dict, run_name: str, match_context: str, args: argparse.Namespace, adapter) -> str | None:
  if args.manual:
    manual_frames = getattr(args, '_manual_positive_frames', {})
    frame = int(row['frame'])
    if frame in manual_frames.get(run_name, set()):
      if matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
        return adapter.filter_post_process_label(row, 'positive', args.action, run_name, args)
      return None

  custom_label = adapter.classify_post_process_label(row, args.action, run_name, args)
  if custom_label == 'skip':
    return None
  if custom_label == 'positive':
    if args.manual:
      return None
    if matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
      return adapter.filter_post_process_label(row, 'positive', args.action, run_name, args)
    return None
  if custom_label == 'negative':
    if matches_patterns(match_context, args.negative_include_pattern, args.negative_exclude_pattern):
      if matches_patterns(match_context, args.normal_include_pattern, args.normal_exclude_pattern):
        return adapter.filter_post_process_label(row, 'negative', args.action, run_name, args)
    return None

  if not args.manual:
    positive = classify_target_action(row, run_name, match_context, args, adapter)
    if positive is not None:
      return adapter.filter_post_process_label(row, positive, args.action, run_name, args)
  normal = classify_normal(row, match_context, args)
  if normal is not None:
    return adapter.filter_post_process_label(row, normal, args.action, run_name, args)
  return None


def main() -> int:
  args = parse_args()
  adapter = get_adapter(args.adapter)
  collection_root = Path(args.collection_root)
  logs_root = Path(args.logs_root) if args.logs_root else default_child_or_root(collection_root, 'logs')
  features_root = Path(args.features_root) if args.features_root else default_child_or_root(collection_root, 'features')
  output_dir = Path(args.output_dir) if args.output_dir else collection_root / 'post_process' / (args.folder_name or args.action)
  output_dir.mkdir(parents=True, exist_ok=True)

  manual_path = None
  if args.manual:
    manual_frames, manual_path = load_manual_positive_frames(
        collection_root, args.action, args.positive_include_pattern)
    setattr(args, '_manual_positive_frames', manual_frames)

  accumulator = {
      'positive_sum': None,
      'negative_sum': None,
      'positive_count': 0,
      'negative_count': 0,
      'flatten': args.flatten,
  }
  manifest_path = output_dir / 'selected_frames.jsonl'
  missing_features = 0
  total_rows = 0

  with manifest_path.open('w', encoding='utf-8') as manifest:
    for log_source in post_process_sources(logs_root, collection_root, args.adapter):
      run_name = run_name_for(log_source)
      match_context = match_context_for(log_source, run_name)
      if log_source.name == 'activation_actions.jsonl':
        rows_iter = read_jsonl(log_source)
      elif log_source.name == 'metas':
        rows_iter = read_hipad_metas(log_source)
      else:
        rows_iter = read_measurements(log_source)
      rows = adapter.augment_rows(list(rows_iter), log_source, collection_root)
      rows = adapter.annotate_rows(rows, args)
      for row in rows:
        total_rows += 1
        label = classify_frame(row, run_name, match_context, args, adapter)
        if label is None:
          continue

        path = feature_path_for(row, log_source, logs_root, features_root, args, adapter)
        if not path.exists():
          missing_features += 1
          continue

        feature = adapter.load_feature(path)
        if add_feature(accumulator, label, feature, args.max_frames_per_class):
          record = {
              'label': label,
              'action': args.action,
              'run_name': run_name,
              'frame': int(row['frame']),
              'feature_path': str(path),
              'speed': float(row.get('speed', 0.0)),
              'steer': float(row.get('steer', 0.0)),
          }
          record.update(adapter.manifest_extra(row))
          manifest.write(json.dumps(record) + '\n')

  if accumulator['positive_count'] == 0 or accumulator['negative_count'] == 0:
    raise RuntimeError(
        f"Need both classes, got positive={accumulator['positive_count']} "
        f"negative={accumulator['negative_count']} missing_features={missing_features}")

  positive_mean = accumulator['positive_sum'] / accumulator['positive_count']
  negative_mean = accumulator['negative_sum'] / accumulator['negative_count']
  steering_vector = positive_mean - negative_mean

  torch.save(positive_mean, output_dir / 'positive_mean.pt')
  torch.save(negative_mean, output_dir / 'negative_mean.pt')
  torch.save(steering_vector, output_dir / 'steering_vector.pt')

  summary = {
      'adapter': args.adapter,
      'action': args.action,
      'positive_label': args.action,
      'negative_label': 'normal',
      'positive_count': accumulator['positive_count'],
      'negative_count': accumulator['negative_count'],
      'total_rows': total_rows,
      'missing_features': missing_features,
      'flatten': args.flatten,
      'vector_formula': 'positive_mean - negative_mean',
      'manual_positive_frames_path': None if manual_path is None else str(manual_path),
      'manual_positive_frame_count': (
          0 if manual_path is None
          else sum(len(frames) for frames in getattr(args, '_manual_positive_frames', {}).values())
      ),
      'output_files': {
          'positive_mean': str(output_dir / 'positive_mean.pt'),
          'negative_mean': str(output_dir / 'negative_mean.pt'),
          'steering_vector': str(output_dir / 'steering_vector.pt'),
          'selected_frames': str(manifest_path),
      },
      'args': {key: value for key, value in vars(args).items() if not key.startswith('_')},
  }
  (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

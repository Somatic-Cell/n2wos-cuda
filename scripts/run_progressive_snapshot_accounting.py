#!/usr/bin/env python3
"""Measure progressive snapshot accounting for NC-WoS and NC+2LMC.

This script uses the existing fixed-snapshot executable
`n2wos_eval_tcnn_nc_wos` to measure several cache snapshots. It then reports
how the same snapshots would be charged under two schedules:

1. offline_serial: train a snapshot, then evaluate with that snapshot.
2. progressive_overlap_model: evaluate snapshot k while training snapshot k+1,
   with block time estimated as max(eval_time(k), train_time(k+1)).

This is not yet the final in-process online implementation: it does not reuse a
model state between snapshots and does not run training/evaluation kernels in
separate CUDA streams. It is intended to answer whether progressive snapshot
accounting can plausibly hide the cache-training cost behind NC+2LMC work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shlex
import subprocess
import sys
from typing import Any, Dict, List


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in text.split(','):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError('expected a comma-separated list of integers')
    return values


def as_float(x: Any, default: float = float('nan')) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
        return float('nan')
    return num / den


def get_2lmc_run(runs: Dict[str, Any], depth_m: int) -> Dict[str, Any]:
    for name in ('nc_2lmc', f'nc_2lmc_m{depth_m}', 'nc_2lmc_m1'):
        value = runs.get(name)
        if isinstance(value, dict):
            return value
    return {}


def mean_bias(run: Dict[str, Any]) -> float:
    if not run:
        return float('nan')
    if 'mean_bias' in run:
        return as_float(run.get('mean_bias'))
    return as_float(run.get('mean_estimate')) - as_float(run.get('mean_exact'))


def add_mesh_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.mesh_path:
        cmd += ['--mesh-path', args.mesh_path]
    if args.normalize is not None:
        cmd += ['--normalize', '1' if args.normalize else '0']
    if args.mesh == 'procedural_bumpy_sphere':
        cmd += [
            '--bumpy-stacks', str(args.bumpy_stacks),
            '--bumpy-slices', str(args.bumpy_slices),
            '--bumpy-amplitude', str(args.bumpy_amplitude),
        ]


def add_eval_args(cmd: List[str], args: argparse.Namespace, output_prefix: pathlib.Path) -> None:
    cmd += ['--eval-mode', args.eval_mode]
    if args.eval_mode == 'ball':
        cmd += ['--eval-points', str(args.eval_points)]
    elif args.eval_mode == 'slice':
        cmd += [
            '--slice-width', str(args.slice_width),
            '--slice-height', str(args.slice_height),
            '--slice-view', args.slice_view,
            '--slice-plane', str(args.slice_plane),
            '--slice-preserve-world-aspect', '1' if args.slice_preserve_world_aspect else '0',
            '--slice-padding-fraction', str(args.slice_padding_fraction),
            '--slice-output-prefix', str(output_prefix),
        ]
        if args.slice_frame:
            cmd += ['--slice-frame', args.slice_frame]
    else:
        raise ValueError(args.eval_mode)


def build_command(args: argparse.Namespace, train_steps_per_refresh: int, output_json: pathlib.Path, seed: int) -> List[str]:
    cmd: List[str] = [
        args.executable,
        '--mesh', args.mesh,
        '--boundary', args.boundary,
        '--label-source', args.label_source,
        '--cache-preset', args.cache_preset,
        '--train-points', str(args.train_points),
        '--label-refreshes', str(args.label_refreshes),
        '--walks-per-label-refresh', str(args.walks_per_label_refresh),
        '--train-steps-per-refresh', str(train_steps_per_refresh),
        '--pure-walks-per-point', str(args.pure_walks_per_point),
        '--hybrid-walks-per-point', str(args.hybrid_walks_per_point),
        '--enable-2lmc', '1' if args.enable_2lmc else '0',
        '--coarse-walks-per-point', str(args.coarse_walks_per_point),
        '--residual-walks-per-point', str(args.residual_walks_per_point),
        '--depth-m', str(args.depth_m),
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--seed', str(seed),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--output', str(output_json),
        '--jit', '1' if args.jit else '0',
    ]
    add_mesh_args(cmd, args)
    add_eval_args(cmd, args, output_json.with_suffix(''))
    return cmd


def run_command(cmd: List[str], dry_run: bool) -> None:
    print('+', shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def row_from_result(label: str, path: pathlib.Path, data: Dict[str, Any], depth_m: int) -> Dict[str, Any]:
    options = data.get('options', {})
    training = data.get('training', {})
    runs = data.get('runs', {})
    pure = runs.get('pure_wos', {}) if isinstance(runs, dict) else {}
    nc = runs.get('nc_wos', {}) if isinstance(runs, dict) else {}
    two = get_2lmc_run(runs if isinstance(runs, dict) else {}, depth_m)
    comparison = data.get('comparison', {})

    pure_var = as_float(pure.get('mean_sample_variance'))
    residual_var = as_float(two.get('mean_residual_sample_variance'))
    train_ms = as_float(training.get('total_training_ms'))
    nc_ms = as_float(nc.get('elapsed_ms'))
    two_ms = as_float(two.get('elapsed_ms'))

    return {
        'label': label,
        'json_path': str(path),
        'train_steps_per_refresh': as_int(options.get('train_steps_per_refresh')),
        'total_train_steps': as_int(options.get('train_steps_per_refresh')) * as_int(options.get('label_refreshes'), 1),
        'depth_m': as_int(options.get('depth_m'), depth_m),
        'cache_preset': options.get('cache_preset'),
        'boundary_condition': options.get('boundary_condition'),
        'label_source': options.get('label_source'),
        'eval_mode': options.get('eval_mode', 'ball'),
        'eval_points': as_int(options.get('eval_points')),
        'inside_pixels': as_int(options.get('inside_pixels')) if 'inside_pixels' in options else '',
        'train_points_padded': as_int(options.get('train_points_padded')),
        'label_update_ms': as_float(training.get('label_update_ms')),
        'tcnn_training_ms': as_float(training.get('tcnn_training_ms')),
        'total_training_ms': train_ms,
        'pure_rmse': as_float(pure.get('rmse')),
        'pure_mean_bias': mean_bias(pure),
        'pure_mean_sample_variance': pure_var,
        'pure_elapsed_ms': as_float(pure.get('elapsed_ms')),
        'nc_wos_rmse': as_float(nc.get('rmse')),
        'nc_wos_mean_bias': mean_bias(nc),
        'nc_wos_elapsed_ms': nc_ms,
        'nc_wos_training_plus_elapsed_ms': as_float(nc.get('training_plus_elapsed_ms'), train_ms + nc_ms),
        'nc_wos_rmse_div_pure_rmse': as_float(comparison.get('nc_wos_rmse_div_pure_wos_rmse'), safe_ratio(as_float(nc.get('rmse')), as_float(pure.get('rmse')))),
        'nc_2lmc_rmse': as_float(two.get('rmse')),
        'nc_2lmc_mean_bias': mean_bias(two),
        'nc_2lmc_elapsed_ms': two_ms,
        'nc_2lmc_training_plus_elapsed_ms': as_float(two.get('training_plus_elapsed_ms'), train_ms + two_ms),
        'nc_2lmc_mean_coarse_sample_variance': as_float(two.get('mean_coarse_sample_variance')),
        'nc_2lmc_mean_residual_sample_variance': residual_var,
        'residual_variance_ratio_vs_pure': safe_ratio(residual_var, pure_var),
        'nc_2lmc_rmse_div_pure_rmse': as_float(comparison.get('nc_2lmc_rmse_div_pure_wos_rmse'), safe_ratio(as_float(two.get('rmse')), as_float(pure.get('rmse')))),
        'nc_2lmc_rmse_div_nc_wos_rmse': as_float(comparison.get('nc_2lmc_rmse_div_nc_wos_rmse'), safe_ratio(as_float(two.get('rmse')), as_float(nc.get('rmse')))),
        'pure_elapsed_div_nc_wos_elapsed': safe_ratio(as_float(pure.get('elapsed_ms')), nc_ms),
        'pure_elapsed_div_nc_2lmc_elapsed': safe_ratio(as_float(pure.get('elapsed_ms')), two_ms),
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def progressive_accounting(rows: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    eval_key = 'nc_2lmc_elapsed_ms' if method == 'nc_2lmc' else 'nc_wos_elapsed_ms'
    serial_cum = 0.0
    overlap_cum = 0.0
    output: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        train_ms = as_float(row.get('total_training_ms'), 0.0)
        eval_ms = as_float(row.get(eval_key), 0.0)
        next_train_ms = as_float(rows[i + 1].get('total_training_ms'), 0.0) if i + 1 < len(rows) else 0.0

        serial_block_ms = train_ms + eval_ms
        # Block k evaluates theta_k while training theta_{k+1}. The last block has
        # no following training charge in this accounting model.
        overlap_block_ms = max(eval_ms, next_train_ms)
        serial_cum += serial_block_ms
        overlap_cum += overlap_block_ms

        out = dict(row)
        out['progressive_method'] = method
        out['progressive_eval_ms'] = eval_ms
        out['progressive_training_ms_for_this_snapshot'] = train_ms
        out['progressive_training_ms_for_next_snapshot'] = next_train_ms
        out['serial_block_ms'] = serial_block_ms
        out['serial_cumulative_ms'] = serial_cum
        out['overlap_block_ms'] = overlap_block_ms
        out['overlap_cumulative_ms'] = overlap_cum
        out['overlap_speedup_vs_serial_cumulative'] = safe_ratio(serial_cum, overlap_cum)
        out['training_hidden_by_eval_ms'] = min(eval_ms, next_train_ms) if next_train_ms > 0.0 else 0.0
        output.append(out)
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', default='./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos')
    p.add_argument('--output-dir', default='results/progressive_snapshot_bumpy_nano_medium_m4')
    p.add_argument('--snapshot-train-steps-per-refresh-list', type=parse_int_list, default=[0, 50, 100, 250])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--progressive-method', choices=['nc_2lmc', 'nc_wos'], default='nc_2lmc')

    p.add_argument('--mesh', default='procedural_bumpy_sphere')
    p.add_argument('--mesh-path', default='')
    p.add_argument('--normalize', type=lambda x: bool(int(x)), default=True)
    p.add_argument('--bumpy-stacks', type=int, default=128)
    p.add_argument('--bumpy-slices', type=int, default=256)
    p.add_argument('--bumpy-amplitude', type=float, default=0.15)

    p.add_argument('--boundary', default='external_charges_medium')
    p.add_argument('--label-source', default='wos_supervision')
    p.add_argument('--cache-preset', default='nano')
    p.add_argument('--train-points', type=int, default=20000)
    p.add_argument('--label-refreshes', type=int, default=4)
    p.add_argument('--walks-per-label-refresh', type=int, default=50)

    p.add_argument('--eval-mode', choices=['ball', 'slice'], default='slice')
    p.add_argument('--eval-points', type=int, default=8192)
    p.add_argument('--slice-width', type=int, default=512)
    p.add_argument('--slice-height', type=int, default=512)
    p.add_argument('--slice-view', choices=['xy', 'xz', 'yz'], default='xy')
    p.add_argument('--slice-plane', type=float, default=0.0)
    p.add_argument('--slice-frame', default='')
    p.add_argument('--slice-padding-fraction', type=float, default=0.02)
    p.add_argument('--slice-preserve-world-aspect', type=lambda x: bool(int(x)), default=True)

    p.add_argument('--depth-m', type=int, default=4)
    p.add_argument('--pure-walks-per-point', type=int, default=64)
    p.add_argument('--hybrid-walks-per-point', type=int, default=4)
    p.add_argument('--enable-2lmc', type=lambda x: bool(int(x)), default=True)
    p.add_argument('--coarse-walks-per-point', type=int, default=64)
    p.add_argument('--residual-walks-per-point', type=int, default=32)
    p.add_argument('--max-steps', type=int, default=256)
    p.add_argument('--epsilon', default='1e-4')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--seed-stride', type=int, default=0)
    p.add_argument('--cubql-build-method', default='sah')
    p.add_argument('--cubql-leaf-size', type=int, default=8)
    p.add_argument('--jit', type=lambda x: bool(int(x)), default=False)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for idx, train_steps in enumerate(args.snapshot_train_steps_per_refresh_list):
        label = f'snapshot_{idx:02d}_steps_per_refresh_{train_steps}'
        out_json = out_dir / f'{label}.json'
        seed = args.seed + idx * args.seed_stride
        cmd = build_command(args, train_steps, out_json, seed)
        commands.append({
            'snapshot_index': idx,
            'train_steps_per_refresh': train_steps,
            'seed': seed,
            'output': str(out_json),
            'command': cmd,
        })
        if out_json.exists() and args.skip_existing:
            print(f'skipping existing {out_json}', flush=True)
        else:
            try:
                run_command(cmd, args.dry_run)
            except subprocess.CalledProcessError as exc:
                if args.continue_on_error:
                    print(f'command failed: {exc.returncode}: {shlex.join(cmd)}', file=sys.stderr)
                    continue
                raise
        if not args.dry_run and out_json.exists():
            rows.append(row_from_result(label, out_json, load_json(out_json), args.depth_m))

    prog_rows = progressive_accounting(rows, args.progressive_method) if rows else []
    summary = {
        'schema': 'n2wos_progressive_snapshot_accounting_v1',
        'purpose': 'Estimate serial versus overlapped wall-clock accounting for progressive NC+2LMC snapshots.',
        'limitations': {
            'actual_same_process_cuda_stream_overlap': False,
            'model_state_reused_between_snapshots': False,
            'residual_samples_reused_for_training': False,
            'training_distribution': 'global training points from n2wos_eval_tcnn_nc_wos',
            'interpretation': 'Each snapshot is measured as an independent fixed-snapshot executable run. The overlap model is post-hoc accounting.'
        },
        'arguments': vars(args),
        'commands': commands,
        'rows': rows,
        'progressive_rows': prog_rows,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if rows:
        write_csv(out_dir / 'summary.csv', rows)
    if prog_rows:
        write_csv(out_dir / 'progressive_summary.csv', prog_rows)

    if prog_rows:
        last = prog_rows[-1]
        print('progressive accounting summary:')
        print(f"  serial cumulative ms:  {as_float(last.get('serial_cumulative_ms')):.3f}")
        print(f"  overlap cumulative ms: {as_float(last.get('overlap_cumulative_ms')):.3f}")
        print(f"  overlap speedup model: {as_float(last.get('overlap_speedup_vs_serial_cumulative')):.3f}")
    print(f'wrote {out_dir / "summary.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

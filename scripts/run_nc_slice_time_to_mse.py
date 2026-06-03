#!/usr/bin/env python3
"""Run a fixed-slice NC / NC+2LMC / pure-WoS time-to-MSE sweep.

This is an orchestration/accounting tool around n2wos_eval_tcnn_nc_wos. It does
not introduce a new solver. The goal is to replace fixed-64-wpp comparisons with
MSE-vs-time points on the same slice and training schedule.

Timing fields written by the underlying executable are interpreted as:
  * solve_only_ms: method evaluation after a cache snapshot exists
  * end_to_end_ms: total_training_ms + solve_only_ms for that independent run
  * single_snapshot_total_ms: one representative training cost for the family +
    solve_only_ms. This approximates training once and reusing the snapshot for
    all budgets in the curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def parse_allocations(text: str) -> List[Tuple[int, int]]:
    allocs: List[Tuple[int, int]] = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f"allocation must be coarse:residual, got {part!r}")
        c, r = part.split(':', 1)
        allocs.append((int(c), int(r)))
    return allocs


def as_float(value: Any, default: float = float('nan')) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def get_run(data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    runs = data.get('runs', {})
    if name in runs:
        return runs[name]
    if name == 'nc_2lmc':
        for key, value in runs.items():
            if key.startswith('nc_2lmc'):
                return value
    return None


def run_command(cmd: Sequence[str], *, dry_run: bool, continue_on_error: bool) -> int:
    print(' '.join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def add_common_args(cmd: List[str], args: argparse.Namespace, output: Path) -> List[str]:
    cmd.extend(['--mesh', args.mesh])
    if args.mesh_path:
        cmd.extend(['--mesh-path', args.mesh_path])
    cmd.extend(['--boundary', args.boundary])
    cmd.extend(['--label-source', args.label_source])
    cmd.extend(['--cache-preset', args.cache_preset])
    cmd.extend(['--train-points', str(args.train_points)])

    cmd.extend(['--eval-mode', 'slice'])
    cmd.extend(['--slice-width', str(args.slice_width)])
    cmd.extend(['--slice-height', str(args.slice_height)])
    cmd.extend(['--slice-view', args.slice_view])
    cmd.extend(['--slice-plane', str(args.slice_plane)])
    cmd.extend(['--slice-preserve-world-aspect', '1' if args.slice_preserve_world_aspect else '0'])
    cmd.extend(['--slice-padding-fraction', str(args.slice_padding_fraction)])

    if args.slice_frame:
        cmd.extend(['--slice-frame', args.slice_frame])

    cmd.extend(['--label-refreshes', str(args.label_refreshes)])
    cmd.extend(['--walks-per-label-refresh', str(args.walks_per_label_refresh)])
    cmd.extend(['--train-steps-per-refresh', str(args.train_steps_per_refresh)])

    cmd.extend(['--depth-m', str(args.depth_m)])
    cmd.extend(['--max-steps', str(args.max_steps)])
    cmd.extend(['--epsilon', str(args.epsilon)])
    cmd.extend(['--seed', str(args.seed)])
    cmd.extend(['--cubql-build-method', args.cubql_build_method])
    cmd.extend(['--cubql-leaf-size', str(args.cubql_leaf_size)])
    cmd.extend(['--output', str(output)])
    cmd.extend(['--normalize', '1' if args.normalize else '0'])
    cmd.extend(['--jit', '1' if args.jit else '0'])

    if args.mesh == 'procedural_bumpy_sphere':
        cmd.extend(['--bumpy-stacks', str(args.bumpy_stacks)])
        cmd.extend(['--bumpy-slices', str(args.bumpy_slices)])
        cmd.extend(['--bumpy-amplitude', str(args.bumpy_amplitude)])

    return cmd


def parse_result_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def make_row(
    *,
    method: str,
    label: str,
    path: Path,
    data: Dict[str, Any],
    run: Dict[str, Any],
    args: argparse.Namespace,
    pure_wpp: Optional[int] = None,
    hybrid_wpp: Optional[int] = None,
    coarse_wpp: Optional[int] = None,
    residual_wpp: Optional[int] = None,
) -> Dict[str, Any]:
    training = data.get('training', {})
    options = data.get('options', {})
    rmse = as_float(run.get('rmse'))
    elapsed_ms = as_float(run.get('elapsed_ms'))
    training_ms = as_float(training.get('total_training_ms'), 0.0)
    row: Dict[str, Any] = {
        'method': method,
        'label': label,
        'json_path': str(path),
        'cache_preset': args.cache_preset if method != 'pure_wos' else '',
        'boundary_condition': args.boundary,
        'depth_m': args.depth_m if method != 'pure_wos' else '',
        'train_points_requested': args.train_points if method != 'pure_wos' else '',
        'train_points_padded': options.get('train_points_padded', ''),
        'total_train_steps': int(args.label_refreshes) * int(args.train_steps_per_refresh),
        'pure_walks_per_point': pure_wpp if pure_wpp is not None else '',
        'hybrid_walks_per_point': hybrid_wpp if hybrid_wpp is not None else '',
        'coarse_walks_per_point': coarse_wpp if coarse_wpp is not None else '',
        'residual_walks_per_point': residual_wpp if residual_wpp is not None else '',
        'slice_width': args.slice_width,
        'slice_height': args.slice_height,
        'slice_view': args.slice_view,
        'slice_plane': args.slice_plane,
        'eval_points': run.get('eval_points', options.get('eval_points', '')),
        'inside_pixels': data.get('slice', {}).get('inside_pixels', run.get('eval_points', '')),
        'rmse': rmse,
        'mse': rmse * rmse if math.isfinite(rmse) else float('nan'),
        'mean_bias': as_float(run.get('mean_bias')),
        'mean_sample_variance': as_float(run.get('mean_sample_variance')),
        'mean_coarse_sample_variance': as_float(run.get('mean_coarse_sample_variance')),
        'mean_residual_sample_variance': as_float(run.get('mean_residual_sample_variance')),
        'residual_variance_ratio_vs_pure': float('nan'),
        'solve_only_ms': elapsed_ms,
        'training_ms': training_ms if method != 'pure_wos' else 0.0,
        'end_to_end_ms': as_float(run.get('training_plus_elapsed_ms'), training_ms + elapsed_ms),
        'single_snapshot_total_ms': float('nan'),
        'label_update_ms': as_float(training.get('label_update_ms'), 0.0) if method != 'pure_wos' else 0.0,
        'tcnn_training_ms': as_float(training.get('tcnn_training_ms'), 0.0) if method != 'pure_wos' else 0.0,
    }
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_thresholds(rows: List[Dict[str, Any]], thresholds: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    time_modes = ['solve_only_ms', 'single_snapshot_total_ms', 'end_to_end_ms']
    methods = sorted(set(str(r['method']) for r in rows))
    for threshold in thresholds:
        threshold_mse = threshold * threshold
        for method in methods:
            method_rows = [r for r in rows if r.get('method') == method]
            for time_mode in time_modes:
                candidates = [r for r in method_rows if as_float(r.get('mse')) <= threshold_mse and math.isfinite(as_float(r.get(time_mode)))]
                if not candidates:
                    out.append({
                        'threshold_rmse': threshold,
                        'threshold_mse': threshold_mse,
                        'method': method,
                        'time_mode': time_mode,
                        'hit': False,
                        'best_time_ms': '',
                        'best_label': '',
                        'best_rmse': '',
                    })
                    continue
                best = min(candidates, key=lambda r: as_float(r.get(time_mode)))
                out.append({
                    'threshold_rmse': threshold,
                    'threshold_mse': threshold_mse,
                    'method': method,
                    'time_mode': time_mode,
                    'hit': True,
                    'best_time_ms': best.get(time_mode),
                    'best_label': best.get('label'),
                    'best_rmse': best.get('rmse'),
                })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--mesh', default='procedural_bumpy_sphere')
    parser.add_argument('--mesh-path', default='')
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--bumpy-stacks', type=int, default=128)
    parser.add_argument('--bumpy-slices', type=int, default=256)
    parser.add_argument('--bumpy-amplitude', type=float, default=0.15)

    parser.add_argument('--boundary', default='external_charges_medium')
    parser.add_argument('--label-source', default='wos_supervision')
    parser.add_argument('--cache-preset', default='nano')
    parser.add_argument('--train-points', type=int, default=20000)
    parser.add_argument('--label-refreshes', type=int, default=4)
    parser.add_argument('--walks-per-label-refresh', type=int, default=50)
    parser.add_argument('--train-steps-per-refresh', type=int, default=5000)

    parser.add_argument('--depth-m', type=int, default=4)
    parser.add_argument('--slice-width', type=int, default=512)
    parser.add_argument('--slice-height', type=int, default=512)
    parser.add_argument('--slice-view', default='xy', choices=['xy', 'xz', 'yz'])
    parser.add_argument('--slice-plane', type=float, default=0.0)
    parser.add_argument('--slice-frame', default='')
    parser.add_argument('--slice-padding-fraction', type=float, default=0.02)
    parser.add_argument('--slice-preserve-world-aspect', type=int, default=1)

    parser.add_argument('--pure-wpp-list', default='8,16,32,64,128')
    parser.add_argument('--nc-hybrid-wpp-list', default='1,2,4,8,16')
    parser.add_argument('--lmc-allocations', default='16:8,32:16,32:32,64:32')
    parser.add_argument('--pure-probe-train-points', type=int, default=512, help='Cheap dummy training points used for pure-only probe runs; pure timing/RMSE are independent of cache training.')

    parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--epsilon', default='1e-4')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--cubql-build-method', default='sah')
    parser.add_argument('--cubql-leaf-size', type=int, default=8)
    parser.add_argument('--jit', type=int, default=0)
    parser.add_argument('--threshold-rmse-list', default='0.05,0.06,0.075,0.1')

    parser.add_argument('--run-pure', type=int, default=1)
    parser.add_argument('--run-nc', type=int, default=1)
    parser.add_argument('--run-2lmc', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')

    args = parser.parse_args(argv)
    args.normalize = bool(args.normalize)
    args.jit = bool(args.jit)
    args.slice_preserve_world_aspect = bool(args.slice_preserve_world_aspect)

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)

    pure_wpps = parse_int_list(args.pure_wpp_list)
    nc_wpps = parse_int_list(args.nc_hybrid_wpp_list)
    allocations = parse_allocations(args.lmc_allocations)
    thresholds = parse_float_list(args.threshold_rmse_list)

    manifest: Dict[str, Any] = {
        'script': 'run_nc_slice_time_to_mse.py',
        'purpose': 'Build MSE-vs-time points for pure WoS, NC-only, and NC+2LMC on a fixed slice.',
        'limitations': {
            'underlying_solver': 'n2wos_eval_tcnn_nc_wos fixed-snapshot executable',
            'actual_progressive_online_training': False,
            'single_snapshot_total_ms_is_accounting': True,
            'pure_probe_uses_dummy_cache_training': True,
            'reference': 'Current executable RMSE is used as reported. High-sample pure-WoS numerical reference is not created by this runner.',
        },
        'arguments': vars(args),
        'commands': [],
        'errors': [],
    }

    all_rows: List[Dict[str, Any]] = []
    raw_json_paths: List[Path] = []

    def maybe_run(label: str, cmd: List[str], output: Path) -> Optional[Dict[str, Any]]:
        manifest['commands'].append({'label': label, 'output': str(output), 'command': cmd})
        if output.exists() and args.skip_existing:
            print(f"[skip] {output}", flush=True)
        else:
            rc = run_command(cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
            if rc != 0:
                manifest['errors'].append({'label': label, 'returncode': rc})
                return None
        if args.dry_run:
            return None
        if not output.exists():
            manifest['errors'].append({'label': label, 'error': 'output_missing'})
            return None
        raw_json_paths.append(output)
        return parse_result_json(output)

    # Pure-WoS budget curve. Use minimal dummy cache training because the current executable
    # always contains NC plumbing; only pure_wos elapsed/RMSE are retained.
    if args.run_pure:
        saved_train_points = args.train_points
        saved_label_refreshes = args.label_refreshes
        saved_walks_per_label_refresh = args.walks_per_label_refresh
        saved_train_steps_per_refresh = args.train_steps_per_refresh
        args.train_points = args.pure_probe_train_points
        args.label_refreshes = 1
        args.walks_per_label_refresh = 1
        args.train_steps_per_refresh = 0
        for wpp in pure_wpps:
            label = f'pure_wpp{wpp}'
            output = runs_dir / f'{label}.json'
            cmd = [args.executable]
            add_common_args(cmd, args, output)
            cmd.extend(['--pure-walks-per-point', str(wpp)])
            cmd.extend(['--hybrid-walks-per-point', '1'])
            cmd.extend(['--enable-2lmc', '0'])
            cmd.extend(['--coarse-walks-per-point', '1'])
            cmd.extend(['--residual-walks-per-point', '1'])
            data = maybe_run(label, cmd, output)
            if data:
                run = get_run(data, 'pure_wos')
                if run:
                    all_rows.append(make_row(method='pure_wos', label=label, path=output, data=data, run=run, args=args, pure_wpp=wpp))
        args.train_points = saved_train_points
        args.label_refreshes = saved_label_refreshes
        args.walks_per_label_refresh = saved_walks_per_label_refresh
        args.train_steps_per_refresh = saved_train_steps_per_refresh

    if args.run_nc:
        for wpp in nc_wpps:
            label = f'nc_wpp{wpp}'
            output = runs_dir / f'{label}.json'
            cmd = [args.executable]
            add_common_args(cmd, args, output)
            cmd.extend(['--pure-walks-per-point', str(max(pure_wpps) if pure_wpps else 64)])
            cmd.extend(['--hybrid-walks-per-point', str(wpp)])
            cmd.extend(['--enable-2lmc', '0'])
            cmd.extend(['--coarse-walks-per-point', '1'])
            cmd.extend(['--residual-walks-per-point', '1'])
            data = maybe_run(label, cmd, output)
            if data:
                run = get_run(data, 'nc_wos')
                if run:
                    all_rows.append(make_row(method='nc_wos', label=label, path=output, data=data, run=run, args=args, hybrid_wpp=wpp))

    if args.run_2lmc:
        for coarse, residual in allocations:
            label = f'nc2lmc_c{coarse}_r{residual}'
            output = runs_dir / f'{label}.json'
            cmd = [args.executable]
            add_common_args(cmd, args, output)
            cmd.extend(['--pure-walks-per-point', str(max(pure_wpps) if pure_wpps else 64)])
            cmd.extend(['--hybrid-walks-per-point', str(min(nc_wpps) if nc_wpps else 4)])
            cmd.extend(['--enable-2lmc', '1'])
            cmd.extend(['--coarse-walks-per-point', str(coarse)])
            cmd.extend(['--residual-walks-per-point', str(residual)])
            data = maybe_run(label, cmd, output)
            if data:
                run = get_run(data, 'nc_2lmc')
                if run:
                    all_rows.append(make_row(method='nc_2lmc', label=label, path=output, data=data, run=run, args=args, coarse_wpp=coarse, residual_wpp=residual))

    # Fill single-snapshot accounting using a representative training cost from NC/2LMC runs.
    training_candidates = [as_float(r.get('training_ms')) for r in all_rows if r.get('method') != 'pure_wos' and as_float(r.get('training_ms')) > 0]
    representative_training_ms = statistics.median(training_candidates) if training_candidates else 0.0
    for row in all_rows:
        if row['method'] == 'pure_wos':
            row['single_snapshot_total_ms'] = row['solve_only_ms']
            row['end_to_end_ms'] = row['solve_only_ms']
        else:
            row['single_snapshot_total_ms'] = representative_training_ms + as_float(row['solve_only_ms'])

    # Compute residual variance ratio against nearest pure sample variance if not present.
    pure_variances = [as_float(r.get('mean_sample_variance')) for r in all_rows if r['method'] == 'pure_wos' and math.isfinite(as_float(r.get('mean_sample_variance')))]
    pure_var_ref = pure_variances[-1] if pure_variances else float('nan')
    for row in all_rows:
        if row['method'] == 'nc_2lmc' and math.isfinite(pure_var_ref) and pure_var_ref > 0:
            rv = as_float(row.get('mean_residual_sample_variance'))
            if math.isfinite(rv):
                row['residual_variance_ratio_vs_pure'] = rv / pure_var_ref

    fieldnames = [
        'method', 'label', 'json_path', 'cache_preset', 'boundary_condition', 'depth_m',
        'train_points_requested', 'train_points_padded', 'total_train_steps',
        'pure_walks_per_point', 'hybrid_walks_per_point', 'coarse_walks_per_point', 'residual_walks_per_point',
        'slice_width', 'slice_height', 'slice_view', 'slice_plane', 'eval_points', 'inside_pixels',
        'rmse', 'mse', 'mean_bias', 'mean_sample_variance', 'mean_coarse_sample_variance',
        'mean_residual_sample_variance', 'residual_variance_ratio_vs_pure',
        'solve_only_ms', 'training_ms', 'single_snapshot_total_ms', 'end_to_end_ms',
        'label_update_ms', 'tcnn_training_ms',
    ]
    write_csv(output_dir / 'time_mse_points.csv', all_rows, fieldnames)

    thresholds_rows = compute_thresholds(all_rows, thresholds)
    write_csv(output_dir / 'time_to_threshold.csv', thresholds_rows, [
        'threshold_rmse', 'threshold_mse', 'method', 'time_mode', 'hit', 'best_time_ms', 'best_label', 'best_rmse'
    ])

    summary = {
        'manifest': manifest,
        'representative_training_ms_for_single_snapshot_total': representative_training_ms,
        'rows': all_rows,
        'thresholds': thresholds_rows,
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {output_dir / 'time_mse_points.csv'}")
    print(f"Wrote {output_dir / 'time_to_threshold.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

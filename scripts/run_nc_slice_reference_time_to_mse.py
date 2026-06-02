#!/usr/bin/env python3
"""Reference-based slice time-to-MSE runner for NC/2LMC WoS.

This runner uses a high-sample Pure WoS run as a numerical reference, then
compares low-budget Pure WoS, NC-WoS, and NC+2LMC estimates against that same
reference point set. It intentionally avoids analytic exact solutions for the
reported MSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_list_int(text: str) -> List[int]:
    return [int(x) for x in text.split(',') if x.strip()]


def parse_allocs(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        a, b = item.split(':')
        out.append((int(a), int(b)))
    return out


def parse_list_float(text: str) -> List[float]:
    return [float(x) for x in text.split(',') if x.strip()]


def run(cmd: List[str], dry_run: bool) -> None:
    print('+', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_estimates(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def mse_vs_reference(rows: List[Dict[str, str]], ref: List[float], column: str) -> Tuple[float, float, float, float]:
    if len(rows) != len(ref):
        raise RuntimeError(f'row count mismatch for {column}: {len(rows)} vs reference {len(ref)}')
    mse = 0.0
    mae = 0.0
    max_abs = 0.0
    mean_err = 0.0
    for row, r in zip(rows, ref):
        e = float(row[column]) - r
        mse += e * e
        ae = abs(e)
        mae += ae
        max_abs = max(max_abs, ae)
        mean_err += e
    n = float(len(ref))
    mse /= n
    return mse, math.sqrt(max(0.0, mse)), mae / n, mean_err / n


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description='Reference-based slice time-to-MSE runner.')
    ap.add_argument('--executable', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mesh', default='procedural_bumpy_sphere')
    ap.add_argument('--mesh-path', default='')
    ap.add_argument('--normalize', type=int, default=1)
    ap.add_argument('--bumpy-stacks', type=int, default=128)
    ap.add_argument('--bumpy-slices', type=int, default=256)
    ap.add_argument('--bumpy-amplitude', type=float, default=0.15)
    ap.add_argument('--boundary', default='external_charges_shell_k16')
    ap.add_argument('--label-source', default='wos_supervision')
    ap.add_argument('--cache-preset', default='nano')
    ap.add_argument('--train-points', type=int, default=5000)
    ap.add_argument('--label-refreshes', type=int, default=4)
    ap.add_argument('--walks-per-label-refresh', type=int, default=16)
    ap.add_argument('--train-steps-per-refresh', type=int, default=50)
    ap.add_argument('--depth-m', type=int, default=4)
    ap.add_argument('--slice-width', type=int, default=512)
    ap.add_argument('--slice-height', type=int, default=512)
    ap.add_argument('--slice-view', default='xy')
    ap.add_argument('--slice-plane', default='0')
    ap.add_argument('--slice-padding-fraction', default='0.02')
    ap.add_argument('--slice-preserve-world-aspect', type=int, default=1)
    ap.add_argument('--reference-wpp', type=int, default=512)
    ap.add_argument('--reference-seed', type=int, default=987654321)
    ap.add_argument('--pure-wpp-list', default='8,16,32,64,128')
    ap.add_argument('--nc-hybrid-wpp-list', default='1,2,4,8,16')
    ap.add_argument('--lmc-allocations', default='16:8,32:16,32:32,64:32')
    ap.add_argument('--threshold-rmse-list', default='0.03,0.05,0.075,0.1')
    ap.add_argument('--max-steps', type=int, default=256)
    ap.add_argument('--epsilon', default='1e-4')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--cubql-build-method', default='sah')
    ap.add_argument('--cubql-leaf-size', type=int, default=8)
    ap.add_argument('--jit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    runs_dir = out_dir / 'runs'
    estimates_dir = out_dir / 'estimates'
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    estimates_dir.mkdir(parents=True, exist_ok=True)

    common = [
        args.executable,
        '--mesh', args.mesh,
        '--boundary', args.boundary,
        '--label-source', args.label_source,
        '--cache-preset', args.cache_preset,
        '--eval-mode', 'slice',
        '--slice-width', str(args.slice_width),
        '--slice-height', str(args.slice_height),
        '--slice-view', args.slice_view,
        '--slice-plane', str(args.slice_plane),
        '--slice-preserve-world-aspect', str(args.slice_preserve_world_aspect),
        '--slice-padding-fraction', str(args.slice_padding_fraction),
        '--depth-m', str(args.depth_m),
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--normalize', str(args.normalize),
        '--jit', str(args.jit),
        '--bumpy-stacks', str(args.bumpy_stacks),
        '--bumpy-slices', str(args.bumpy_slices),
        '--bumpy-amplitude', str(args.bumpy_amplitude),
    ]
    if args.mesh_path:
        common += ['--mesh-path', args.mesh_path]

    # 1. High-sample Pure WoS reference.
    ref_json = runs_dir / 'reference_pure.json'
    ref_prefix = estimates_dir / 'reference_pure'
    if not (args.skip_existing and ref_json.exists() and Path(str(ref_prefix) + '_estimates.csv').exists()):
        cmd = common + [
            '--train-points', '128',
            '--label-refreshes', '1',
            '--walks-per-label-refresh', '1',
            '--train-steps-per-refresh', '0',
            '--skip-training', '1',
            '--pure-walks-per-point', str(args.reference_wpp),
            '--hybrid-walks-per-point', '1',
            '--enable-2lmc', '0',
            '--coarse-walks-per-point', '1',
            '--residual-walks-per-point', '1',
            '--seed', str(args.reference_seed),
            '--slice-output-prefix', str(out_dir / 'slice'),
            '--save-estimates-prefix', str(ref_prefix),
            '--output', str(ref_json),
        ]
        run(cmd, args.dry_run)

    ref_rows = read_estimates(Path(str(ref_prefix) + '_estimates.csv')) if not args.dry_run else []
    ref_values = [float(r['pure_mean']) for r in ref_rows]
    ref_var = [float(r.get('pure_sample_variance', 0.0)) for r in ref_rows]
    mean_ref_var = sum(ref_var) / len(ref_var) if ref_var else 0.0
    reference_rmse_floor = math.sqrt(max(0.0, mean_ref_var / float(args.reference_wpp))) if ref_var else 0.0

    points: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []

    def add_point(method: str, label: str, json_path: Path, prefix: Path, value_column: str, time_json_path: Tuple[str, ...], training_included: bool) -> None:
        j = read_json(json_path)
        rows = read_estimates(Path(str(prefix) + '_estimates.csv'))
        mse, rmse, mae, mean_error = mse_vs_reference(rows, ref_values, value_column)
        node: Any = j
        for key in time_json_path:
            node = node[key]
        solve_ms = float(node.get('elapsed_ms', 0.0))
        training_ms = float(j.get('training', {}).get('total_training_ms', 0.0))
        total_ms = solve_ms + training_ms if training_included else solve_ms
        points.append({
            'method': method,
            'label': label,
            'rmse_vs_reference': rmse,
            'mse_vs_reference': mse,
            'mae_vs_reference': mae,
            'mean_error_vs_reference': mean_error,
            'solve_only_ms': solve_ms,
            'training_ms': training_ms,
            'training_plus_solve_ms': solve_ms + training_ms,
            'default_time_ms': total_ms,
            'reference_wpp': args.reference_wpp,
            'reference_rmse_floor_estimate': reference_rmse_floor,
            'json_path': str(json_path),
            'estimates_csv': str(prefix) + '_estimates.csv',
        })

    # 2. Low-budget Pure WoS curve.
    for wpp in parse_list_int(args.pure_wpp_list):
        label = f'pure_wpp{wpp}'
        json_path = runs_dir / f'{label}.json'
        prefix = estimates_dir / label
        cmd = common + [
            '--train-points', '128',
            '--label-refreshes', '1',
            '--walks-per-label-refresh', '1',
            '--train-steps-per-refresh', '0',
            '--skip-training', '1',
            '--pure-walks-per-point', str(wpp),
            '--hybrid-walks-per-point', '1',
            '--enable-2lmc', '0',
            '--coarse-walks-per-point', '1',
            '--residual-walks-per-point', '1',
            '--seed', str(args.seed + 1000 + wpp),
            '--slice-output-prefix', str(out_dir / 'slice'),
            '--save-estimates-prefix', str(prefix),
            '--output', str(json_path),
        ]
        commands.append({'method': 'pure_wos', 'label': label, 'command': cmd})
        if not (args.skip_existing and json_path.exists() and Path(str(prefix) + '_estimates.csv').exists()):
            run(cmd, args.dry_run)
        if not args.dry_run:
            add_point('pure_wos', label, json_path, prefix, 'pure_mean', ('runs', 'pure_wos'), False)

    # 3. NC-only curve.
    for wpp in parse_list_int(args.nc_hybrid_wpp_list):
        label = f'nc_wpp{wpp}'
        json_path = runs_dir / f'{label}.json'
        prefix = estimates_dir / label
        cmd = common + [
            '--train-points', str(args.train_points),
            '--label-refreshes', str(args.label_refreshes),
            '--walks-per-label-refresh', str(args.walks_per_label_refresh),
            '--train-steps-per-refresh', str(args.train_steps_per_refresh),
            '--pure-walks-per-point', '1',
            '--hybrid-walks-per-point', str(wpp),
            '--enable-2lmc', '0',
            '--coarse-walks-per-point', '1',
            '--residual-walks-per-point', '1',
            '--seed', str(args.seed + 2000 + wpp),
            '--slice-output-prefix', str(out_dir / 'slice'),
            '--save-estimates-prefix', str(prefix),
            '--output', str(json_path),
        ]
        commands.append({'method': 'nc_wos', 'label': label, 'command': cmd})
        if not (args.skip_existing and json_path.exists() and Path(str(prefix) + '_estimates.csv').exists()):
            run(cmd, args.dry_run)
        if not args.dry_run:
            add_point('nc_wos', label, json_path, prefix, 'nc_wos_mean', ('runs', 'nc_wos'), True)

    # 4. NC+2LMC curve.
    for coarse, residual in parse_allocs(args.lmc_allocations):
        label = f'nc2lmc_c{coarse}_r{residual}'
        json_path = runs_dir / f'{label}.json'
        prefix = estimates_dir / label
        cmd = common + [
            '--train-points', str(args.train_points),
            '--label-refreshes', str(args.label_refreshes),
            '--walks-per-label-refresh', str(args.walks_per_label_refresh),
            '--train-steps-per-refresh', str(args.train_steps_per_refresh),
            '--pure-walks-per-point', '1',
            '--hybrid-walks-per-point', '1',
            '--enable-2lmc', '1',
            '--coarse-walks-per-point', str(coarse),
            '--residual-walks-per-point', str(residual),
            '--seed', str(args.seed + 3000 + coarse * 17 + residual),
            '--slice-output-prefix', str(out_dir / 'slice'),
            '--save-estimates-prefix', str(prefix),
            '--output', str(json_path),
        ]
        commands.append({'method': 'nc_2lmc', 'label': label, 'command': cmd})
        if not (args.skip_existing and json_path.exists() and Path(str(prefix) + '_estimates.csv').exists()):
            run(cmd, args.dry_run)
        if not args.dry_run:
            add_point('nc_2lmc', label, json_path, prefix, 'nc_2lmc_mean', ('runs', 'nc_2lmc'), True)

    if args.dry_run:
        manifest = {'arguments': vars(args), 'commands': commands, 'dry_run': True}
        (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return 0

    fields = ['method','label','rmse_vs_reference','mse_vs_reference','mae_vs_reference','mean_error_vs_reference','solve_only_ms','training_ms','training_plus_solve_ms','default_time_ms','reference_wpp','reference_rmse_floor_estimate','json_path','estimates_csv']
    write_csv(out_dir / 'time_mse_points_reference.csv', points, fields)

    thresholds = parse_list_float(args.threshold_rmse_list)
    thresh_rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        for method in ['pure_wos', 'nc_wos', 'nc_2lmc']:
            candidates = [p for p in points if p['method'] == method and p['rmse_vs_reference'] <= thr]
            for time_mode in ['solve_only_ms', 'training_plus_solve_ms']:
                if candidates:
                    best = min(candidates, key=lambda p: float(p[time_mode]))
                    thresh_rows.append({'threshold_rmse': thr, 'method': method, 'time_mode': time_mode, 'hit': True, 'best_time_ms': best[time_mode], 'best_label': best['label'], 'best_rmse_vs_reference': best['rmse_vs_reference']})
                else:
                    thresh_rows.append({'threshold_rmse': thr, 'method': method, 'time_mode': time_mode, 'hit': False, 'best_time_ms': '', 'best_label': '', 'best_rmse_vs_reference': ''})
    write_csv(out_dir / 'time_to_threshold_reference.csv', thresh_rows, ['threshold_rmse','method','time_mode','hit','best_time_ms','best_label','best_rmse_vs_reference'])

    summary = {
        'schema': 'n2wos_reference_slice_time_to_mse_v1',
        'arguments': vars(args),
        'reference': {
            'json_path': str(ref_json),
            'estimates_csv': str(ref_prefix) + '_estimates.csv',
            'reference_wpp': args.reference_wpp,
            'reference_rmse_floor_estimate': reference_rmse_floor,
            'mean_reference_sample_variance': mean_ref_var,
        },
        'points': points,
        'thresholds': thresh_rows,
        'commands': commands,
        'limitations': {
            'reference_is_monte_carlo': True,
            'reference_seed_independent_from_method_seeds': True,
            'method_runs_retrain_cache_independently': True,
            'not_true_online_progressive_runner': True,
        },
    }
    (out_dir / 'summary_reference.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('wrote', out_dir / 'time_mse_points_reference.csv')
    print('wrote', out_dir / 'time_to_threshold_reference.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

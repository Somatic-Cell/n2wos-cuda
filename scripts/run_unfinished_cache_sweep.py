#!/usr/bin/env python3
"""Run an unfinished-cache NC-WoS / NC+2LMC diagnostic sweep.

The script launches n2wos_eval_tcnn_nc_wos multiple times with the same mesh,
cache preset, prefix depth m, and evaluation points, while varying the amount of
cache training. It tests whether NC+2LMC removes the bias of an intentionally
unfinished cache before any m>1 sweep is introduced.
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
from typing import Any, Dict, Iterable, List, Optional


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


def get_run(runs: Dict[str, Any], *names: str) -> Dict[str, Any]:
    for name in names:
        value = runs.get(name)
        if isinstance(value, dict):
            return value
    return {}


def mean_bias(run: Dict[str, Any]) -> float:
    if 'mean_bias' in run:
        return as_float(run.get('mean_bias'))
    return as_float(run.get('mean_estimate')) - as_float(run.get('mean_exact'))


def build_command(args: argparse.Namespace, train_steps_per_refresh: int, output_json: pathlib.Path, seed: int) -> List[str]:
    cmd: List[str] = [
        args.executable,
        '--mesh', args.mesh,
        '--boundary', args.boundary,
        '--label-source', args.label_source,
        '--cache-preset', args.cache_preset,
        '--train-points', str(args.train_points),
        '--eval-points', str(args.eval_points),
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
    ]
    if args.mesh_path:
        cmd.extend(['--mesh-path', args.mesh_path])
    if args.normalize is not None:
        cmd.extend(['--normalize', '1' if args.normalize else '0'])
    if args.mesh == 'procedural_bumpy_sphere':
        cmd.extend([
            '--bumpy-stacks', str(args.bumpy_stacks),
            '--bumpy-slices', str(args.bumpy_slices),
            '--bumpy-amplitude', str(args.bumpy_amplitude),
        ])
    if args.jit is not None:
        cmd.extend(['--jit', '1' if args.jit else '0'])
    return cmd


def summarize_one(path: pathlib.Path, train_steps_per_refresh: int) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    runs = data.get('runs', {}) if isinstance(data.get('runs'), dict) else {}
    pure = get_run(runs, 'pure_wos')
    nc = get_run(runs, 'nc_wos')
    two = get_run(runs, 'nc_2lmc_m1', 'nc_2lmc')
    training = data.get('training', {}) if isinstance(data.get('training'), dict) else {}
    options = data.get('options', {}) if isinstance(data.get('options'), dict) else {}
    comparison = data.get('comparison', {}) if isinstance(data.get('comparison'), dict) else {}

    pure_rmse = as_float(pure.get('rmse'))
    nc_rmse = as_float(nc.get('rmse'))
    two_rmse = as_float(two.get('rmse'))
    pure_var = as_float(pure.get('mean_sample_variance'))
    residual_var = as_float(two.get('mean_residual_sample_variance')) if two else float('nan')
    nc_bias = mean_bias(nc)
    two_bias = mean_bias(two) if two else float('nan')

    row: Dict[str, Any] = {
        'label': f'steps_per_refresh_{train_steps_per_refresh}',
        'json_path': str(path),
        'cache_preset': options.get('cache_preset', ''),
        'boundary_condition': options.get('boundary_condition', ''),
        'label_source': options.get('label_source', ''),
        'depth_m': as_int(options.get('depth_m')),
        'label_refreshes': as_int(options.get('label_refreshes')),
        'walks_per_label_refresh': as_int(options.get('walks_per_label_refresh')),
        'train_steps_per_refresh': train_steps_per_refresh,
        'total_train_steps': train_steps_per_refresh * as_int(options.get('label_refreshes')),
        'train_points_padded': as_int(options.get('train_points_padded')),
        'eval_points': as_int(options.get('eval_points')),
        'pure_walks_per_point': as_int(options.get('pure_walks_per_point')),
        'hybrid_walks_per_point': as_int(options.get('hybrid_walks_per_point')),
        'coarse_walks_per_point': as_int(options.get('coarse_walks_per_point')),
        'residual_walks_per_point': as_int(options.get('residual_walks_per_point')),
        'label_update_ms': as_float(training.get('label_update_ms')),
        'tcnn_training_ms': as_float(training.get('tcnn_training_ms')),
        'total_training_ms': as_float(training.get('total_training_ms')),
        'pure_rmse': pure_rmse,
        'pure_mean_bias': mean_bias(pure),
        'pure_mean_sample_variance': pure_var,
        'pure_elapsed_ms': as_float(pure.get('elapsed_ms')),
        'nc_wos_rmse': nc_rmse,
        'nc_wos_mean_bias': nc_bias,
        'nc_wos_abs_mean_bias': abs(nc_bias) if math.isfinite(nc_bias) else float('nan'),
        'nc_wos_mean_sample_variance': as_float(nc.get('mean_sample_variance')),
        'nc_wos_elapsed_ms': as_float(nc.get('elapsed_ms')),
        'nc_wos_total_ms': as_float(nc.get('training_plus_elapsed_ms')),
        'nc_wos_rmse_div_pure_rmse': safe_ratio(nc_rmse, pure_rmse),
        'nc_2lmc_rmse': two_rmse,
        'nc_2lmc_mean_bias': two_bias,
        'nc_2lmc_abs_mean_bias': abs(two_bias) if math.isfinite(two_bias) else float('nan'),
        'nc_2lmc_mean_coarse_sample_variance': as_float(two.get('mean_coarse_sample_variance')) if two else float('nan'),
        'nc_2lmc_mean_residual_sample_variance': residual_var,
        'residual_variance_ratio_vs_pure': safe_ratio(residual_var, pure_var),
        'nc_2lmc_elapsed_ms': as_float(two.get('elapsed_ms')) if two else float('nan'),
        'nc_2lmc_total_ms': as_float(two.get('training_plus_elapsed_ms')) if two else float('nan'),
        'nc_2lmc_rmse_div_pure_rmse': safe_ratio(two_rmse, pure_rmse),
        'nc_2lmc_rmse_div_nc_wos_rmse': safe_ratio(two_rmse, nc_rmse),
        'bias_abs_ratio_2lmc_over_nc': safe_ratio(abs(two_bias), abs(nc_bias)),
        'pure_elapsed_div_nc_inference_elapsed': as_float(comparison.get('pure_elapsed_div_nc_inference_elapsed')),
        'pure_elapsed_div_nc_total_with_training': as_float(comparison.get('pure_elapsed_div_nc_total_with_training')),
        'pure_elapsed_div_nc_2lmc_inference_elapsed': as_float(comparison.get('pure_elapsed_div_nc_2lmc_inference_elapsed')),
        'pure_elapsed_div_nc_2lmc_total_with_training': as_float(comparison.get('pure_elapsed_div_nc_2lmc_total_with_training')),
    }
    return row


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True, help='Path to n2wos_eval_tcnn_nc_wos')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mesh', default='procedural_bumpy_sphere')
    p.add_argument('--mesh-path', default='')
    p.add_argument('--normalize', type=int, choices=[0, 1], default=1)
    p.add_argument('--bumpy-stacks', type=int, default=128)
    p.add_argument('--bumpy-slices', type=int, default=256)
    p.add_argument('--bumpy-amplitude', type=float, default=0.15)
    p.add_argument('--boundary', default='external_charges_high')
    p.add_argument('--label-source', default='wos_supervision')
    p.add_argument('--cache-preset', default='light')
    p.add_argument('--train-points', type=int, default=20000)
    p.add_argument('--eval-points', type=int, default=8192)
    p.add_argument('--label-refreshes', type=int, default=4)
    p.add_argument('--walks-per-label-refresh', type=int, default=16)
    p.add_argument('--train-steps-per-refresh-list', type=parse_int_list, default=parse_int_list('0,50,100,250,500,1000'))
    p.add_argument('--pure-walks-per-point', type=int, default=64)
    p.add_argument('--hybrid-walks-per-point', type=int, default=4)
    p.add_argument('--enable-2lmc', type=int, choices=[0, 1], default=1)
    p.add_argument('--coarse-walks-per-point', type=int, default=64)
    p.add_argument('--residual-walks-per-point', type=int, default=32)
    p.add_argument('--depth-m', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=256)
    p.add_argument('--epsilon', default='1e-4')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--seed-stride', type=int, default=0, help='If nonzero, add k*stride to the seed for the k-th checkpoint')
    p.add_argument('--cubql-build-method', default='sah')
    p.add_argument('--cubql-leaf-size', type=int, default=8)
    p.add_argument('--jit', type=int, choices=[0, 1], default=0)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--skip-existing', action='store_true')
    args = p.parse_args(list(argv) if argv is not None else None)

    args.normalize = bool(args.normalize)
    args.enable_2lmc = bool(args.enable_2lmc)
    args.jit = bool(args.jit)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'script': 'run_unfinished_cache_sweep.py',
        'purpose': 'diagnose NC-only bias and NC+2LMC bias correction as cache training is intentionally left unfinished',
        'arguments': vars(args).copy(),
        'commands': [],
    }

    rows: List[Dict[str, Any]] = []
    for k, steps in enumerate(args.train_steps_per_refresh_list):
        seed = args.seed + (k * args.seed_stride if args.seed_stride else 0)
        output_json = out_dir / f'unfinished_cache_steps_per_refresh_{steps}.json'
        cmd = build_command(args, steps, output_json, seed)
        manifest['commands'].append({'train_steps_per_refresh': steps, 'seed': seed, 'output': str(output_json), 'command': cmd})
        print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and output_json.exists():
            print(f'skipping existing {output_json}', flush=True)
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                if args.continue_on_error:
                    print(f'command failed with code {exc.returncode}: {output_json}', file=sys.stderr)
                    continue
                raise
        if output_json.exists():
            rows.append(summarize_one(output_json, steps))

    if rows:
        write_csv(out_dir / 'summary.csv', rows)
        with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump({'manifest': manifest, 'rows': rows}, f, indent=2)
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    if rows:
        best_bias = min(rows, key=lambda r: as_float(r.get('nc_2lmc_abs_mean_bias'), float('inf')))
        best_rmse = min(rows, key=lambda r: as_float(r.get('nc_2lmc_rmse'), float('inf')))
        print('wrote', out_dir / 'summary.csv')
        print('best NC+2LMC abs mean bias:', best_bias.get('label'), best_bias.get('nc_2lmc_abs_mean_bias'))
        print('best NC+2LMC RMSE:', best_rmse.get('label'), best_rmse.get('nc_2lmc_rmse'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

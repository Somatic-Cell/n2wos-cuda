#!/usr/bin/env python3
"""Run NC-WoS / NC+2LMC on a fixed 2D slice grid.

This is the first paper-style evaluation wrapper: evaluation points are no
longer random 8192 interior samples. The executable builds a width x height
slice grid, masks pixels outside the mesh, and evaluates only interior pixel
centers. The slice mask and the exact list of evaluated points are written next
to each JSON result by n2wos_eval_tcnn_nc_wos.
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
    out: List[int] = []
    for part in text.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError('expected comma-separated integers')
    return out


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


def safe_ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or b == 0.0:
        return float('nan')
    return a / b


def get_two_level(runs: Dict[str, Any], depth_m: int) -> Dict[str, Any]:
    for key in ('nc_2lmc', f'nc_2lmc_m{depth_m}', 'nc_2lmc_m1'):
        val = runs.get(key)
        if isinstance(val, dict):
            return val
    return {}


def mean_bias(run: Dict[str, Any]) -> float:
    if not run:
        return float('nan')
    if 'mean_bias' in run:
        return as_float(run.get('mean_bias'))
    return as_float(run.get('mean_estimate')) - as_float(run.get('mean_exact'))


def build_command(args: argparse.Namespace, depth_m: int, output_json: pathlib.Path, seed: int) -> List[str]:
    prefix = output_json.with_suffix('')
    cmd = [
        args.executable,
        '--mesh', args.mesh,
        '--boundary', args.boundary,
        '--label-source', args.label_source,
        '--cache-preset', args.cache_preset,
        '--train-points', str(args.train_points),
        '--eval-mode', 'slice',
        '--slice-width', str(args.slice_width),
        '--slice-height', str(args.slice_height),
        '--slice-view', args.slice_view,
        '--slice-plane', str(args.slice_plane),
        '--slice-preserve-world-aspect', '1' if args.slice_preserve_world_aspect else '0',
        '--slice-padding-fraction', str(args.slice_padding_fraction),
        '--slice-output-prefix', str(prefix) + '_slice',
        '--label-refreshes', str(args.label_refreshes),
        '--walks-per-label-refresh', str(args.walks_per_label_refresh),
        '--train-steps-per-refresh', str(args.train_steps_per_refresh),
        '--pure-walks-per-point', str(args.pure_walks_per_point),
        '--hybrid-walks-per-point', str(args.hybrid_walks_per_point),
        '--enable-2lmc', '1' if args.enable_2lmc else '0',
        '--coarse-walks-per-point', str(args.coarse_walks_per_point),
        '--residual-walks-per-point', str(args.residual_walks_per_point),
        '--depth-m', str(depth_m),
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--seed', str(seed),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--output', str(output_json),
    ]
    if args.mesh_path:
        cmd += ['--mesh-path', args.mesh_path]
    cmd += ['--normalize', '1' if args.normalize else '0']
    if args.mesh == 'procedural_bumpy_sphere':
        cmd += [
            '--bumpy-stacks', str(args.bumpy_stacks),
            '--bumpy-slices', str(args.bumpy_slices),
            '--bumpy-amplitude', str(args.bumpy_amplitude),
        ]
    if args.slice_frame:
        cmd += ['--slice-frame', args.slice_frame]
    cmd += ['--jit', '1' if args.jit else '0']
    return cmd


def summarize_one(path: pathlib.Path, depth_m: int) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    runs = data.get('runs', {}) if isinstance(data.get('runs'), dict) else {}
    pure = runs.get('pure_wos', {}) if isinstance(runs.get('pure_wos'), dict) else {}
    nc = runs.get('nc_wos', {}) if isinstance(runs.get('nc_wos'), dict) else {}
    two = get_two_level(runs, depth_m)
    training = data.get('training', {}) if isinstance(data.get('training'), dict) else {}
    opts = data.get('options', {}) if isinstance(data.get('options'), dict) else {}
    comp = data.get('comparison', {}) if isinstance(data.get('comparison'), dict) else {}
    sl = data.get('slice_eval', {}) if isinstance(data.get('slice_eval'), dict) else {}
    pure_rmse = as_float(pure.get('rmse'))
    nc_rmse = as_float(nc.get('rmse'))
    two_rmse = as_float(two.get('rmse'))
    pure_var = as_float(pure.get('mean_sample_variance'))
    residual_var = as_float(two.get('mean_residual_sample_variance')) if two else float('nan')
    nc_bias = mean_bias(nc)
    two_bias = mean_bias(two)
    return {
        'label': f'm{depth_m}',
        'json_path': str(path),
        'depth_m': depth_m,
        'cache_preset': opts.get('cache_preset', ''),
        'boundary_condition': opts.get('boundary_condition', ''),
        'label_source': opts.get('label_source', ''),
        'slice_width': as_int(sl.get('width')),
        'slice_height': as_int(sl.get('height')),
        'candidate_pixels': as_int(sl.get('candidate_pixels')),
        'inside_pixels': as_int(sl.get('inside_pixels')),
        'mask_ppm': sl.get('mask_ppm', ''),
        'points_csv': sl.get('points_csv', ''),
        'train_points_padded': as_int(opts.get('train_points_padded')),
        'eval_points': as_int(opts.get('eval_points')),
        'total_train_steps': as_int(opts.get('label_refreshes')) * as_int(opts.get('train_steps_per_refresh')),
        'label_update_ms': as_float(training.get('label_update_ms')),
        'tcnn_training_ms': as_float(training.get('tcnn_training_ms')),
        'total_training_ms': as_float(training.get('total_training_ms')),
        'pure_rmse': pure_rmse,
        'pure_mean_bias': mean_bias(pure),
        'pure_mean_sample_variance': pure_var,
        'pure_elapsed_ms': as_float(pure.get('elapsed_ms')),
        'nc_wos_rmse': nc_rmse,
        'nc_wos_mean_bias': nc_bias,
        'nc_wos_elapsed_ms': as_float(nc.get('elapsed_ms')),
        'nc_wos_total_ms': as_float(nc.get('training_plus_elapsed_ms')),
        'nc_wos_rmse_div_pure_rmse': safe_ratio(nc_rmse, pure_rmse),
        'nc_2lmc_rmse': two_rmse,
        'nc_2lmc_mean_bias': two_bias,
        'nc_2lmc_mean_coarse_sample_variance': as_float(two.get('mean_coarse_sample_variance')) if two else float('nan'),
        'nc_2lmc_mean_residual_sample_variance': residual_var,
        'residual_variance_ratio_vs_pure': safe_ratio(residual_var, pure_var),
        'nc_2lmc_elapsed_ms': as_float(two.get('elapsed_ms')) if two else float('nan'),
        'nc_2lmc_total_ms': as_float(two.get('training_plus_elapsed_ms')) if two else float('nan'),
        'nc_2lmc_rmse_div_pure_rmse': safe_ratio(two_rmse, pure_rmse),
        'nc_2lmc_rmse_div_nc_wos_rmse': safe_ratio(two_rmse, nc_rmse),
        'pure_elapsed_div_nc_inference_elapsed': as_float(comp.get('pure_elapsed_div_nc_inference_elapsed')),
        'pure_elapsed_div_nc_2lmc_inference_elapsed': as_float(comp.get('pure_elapsed_div_nc_2lmc_inference_elapsed')),
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mesh', default='procedural_bumpy_sphere')
    ap.add_argument('--mesh-path', default='')
    ap.add_argument('--normalize', type=int, choices=[0, 1], default=1)
    ap.add_argument('--bumpy-stacks', type=int, default=128)
    ap.add_argument('--bumpy-slices', type=int, default=256)
    ap.add_argument('--bumpy-amplitude', type=float, default=0.15)
    ap.add_argument('--boundary', default='external_charges_medium')
    ap.add_argument('--label-source', default='wos_supervision')
    ap.add_argument('--cache-preset', default='nano')
    ap.add_argument('--train-points', type=int, default=20000)
    ap.add_argument('--label-refreshes', type=int, default=4)
    ap.add_argument('--walks-per-label-refresh', type=int, default=16)
    ap.add_argument('--train-steps-per-refresh', type=int, default=50)
    ap.add_argument('--depths', type=parse_int_list, default=parse_int_list('1,2,4'))
    ap.add_argument('--slice-width', type=int, default=512)
    ap.add_argument('--slice-height', type=int, default=512)
    ap.add_argument('--slice-view', default='xy', choices=['xy', 'xz', 'yz'])
    ap.add_argument('--slice-plane', type=float, default=0.0)
    ap.add_argument('--slice-frame', default='')
    ap.add_argument('--slice-padding-fraction', type=float, default=0.02)
    ap.add_argument('--slice-preserve-world-aspect', type=int, choices=[0, 1], default=1)
    ap.add_argument('--pure-walks-per-point', type=int, default=64)
    ap.add_argument('--hybrid-walks-per-point', type=int, default=4)
    ap.add_argument('--enable-2lmc', type=int, choices=[0, 1], default=1)
    ap.add_argument('--coarse-walks-per-point', type=int, default=64)
    ap.add_argument('--residual-walks-per-point', type=int, default=32)
    ap.add_argument('--max-steps', type=int, default=256)
    ap.add_argument('--epsilon', default='1e-4')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--seed-stride', type=int, default=0)
    ap.add_argument('--cubql-build-method', default='sah')
    ap.add_argument('--cubql-leaf-size', type=int, default=8)
    ap.add_argument('--jit', type=int, choices=[0, 1], default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)
    args.normalize = bool(args.normalize)
    args.enable_2lmc = bool(args.enable_2lmc)
    args.jit = bool(args.jit)
    args.slice_preserve_world_aspect = bool(args.slice_preserve_world_aspect)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'script': 'run_nc_slice_depth_sweep.py', 'purpose': 'fixed slice-grid NC/2LMC depth sweep', 'arguments': vars(args).copy(), 'commands': []}
    rows: List[Dict[str, Any]] = []
    for k, depth in enumerate(args.depths):
        seed = args.seed + (k * args.seed_stride if args.seed_stride else 0)
        output_json = out_dir / f'nc_slice_m{depth}.json'
        cmd = build_command(args, depth, output_json, seed)
        manifest['commands'].append({'depth_m': depth, 'seed': seed, 'output': str(output_json), 'command': cmd})
        print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
        if not args.dry_run:
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
            rows.append(summarize_one(output_json, depth))
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if rows:
        write_csv(out_dir / 'summary.csv', rows)
        (out_dir / 'summary.json').write_text(json.dumps({'manifest': manifest, 'rows': rows}, indent=2), encoding='utf-8')
        print(f'wrote {out_dir / "summary.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a boundary-texture / depth matrix using one high-sample Pure-WoS reference per boundary.

This orchestrator is intentionally a thin layer over two existing runners:

1. run_nc_slice_reference_time_to_mse.py
   Builds a high-sample Pure-WoS numerical reference once per boundary texture.
   It also produces the low-budget Pure-WoS curve against that reference.

2. run_nc_slice_reference_cache_sweep.py
   Reuses the same reference CSV and evaluates NC-only / NC+2LMC for each
   requested depth m and cache/training setting.

The purpose is to avoid rebuilding the expensive reference when sweeping m.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def parse_list_int(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def run(cmd: List[str], dry_run: bool) -> None:
    print('+', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sanitize_name(text: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in text)


def add_common_geometry_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd += [
        '--mesh', args.mesh,
        '--boundary', args.boundary_placeholder,  # replaced by caller
        '--label-source', args.label_source,
        '--cache-preset', args.cache_preset_for_reference,
        '--train-points', str(args.train_points_for_reference),
        '--label-refreshes', str(args.label_refreshes),
        '--walks-per-label-refresh', str(args.walks_per_label_refresh),
        '--train-steps-per-refresh', str(args.train_steps_per_refresh_for_reference),
        '--depth-m', str(args.reference_depth_m),
        '--slice-width', str(args.slice_width),
        '--slice-height', str(args.slice_height),
        '--slice-view', args.slice_view,
        '--slice-plane', str(args.slice_plane),
        '--reference-wpp', str(args.reference_wpp),
        '--reference-chunk-wpp', str(args.reference_chunk_wpp),
        '--pure-wpp-list', args.pure_wpp_list,
        '--nc-hybrid-wpp-list', args.reference_nc_hybrid_wpp_list,
        '--lmc-allocations', args.reference_lmc_allocations,
        '--threshold-rmse-list', args.threshold_rmse_list,
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--normalize', str(args.normalize),
        '--jit', str(args.jit),
        '--seed', str(args.seed),
    ]
    if args.mesh_path:
        cmd += ['--mesh-path', args.mesh_path]
    if args.mesh == 'procedural_bumpy_sphere':
        cmd += [
            '--bumpy-stacks', str(args.bumpy_stacks),
            '--bumpy-slices', str(args.bumpy_slices),
            '--bumpy-amplitude', str(args.bumpy_amplitude),
        ]
    if args.skip_existing:
        cmd += ['--skip-existing']


def main() -> int:
    ap = argparse.ArgumentParser(description='Boundary texture / depth matrix with one reference per boundary.')
    ap.add_argument('--executable', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--reference-runner', default='scripts/run_nc_slice_reference_time_to_mse.py')
    ap.add_argument('--cache-sweep-runner', default='scripts/run_nc_slice_reference_cache_sweep.py')

    ap.add_argument('--mesh', default='procedural_bumpy_sphere')
    ap.add_argument('--mesh-path', default='')
    ap.add_argument('--normalize', type=int, default=1)
    ap.add_argument('--bumpy-stacks', type=int, default=128)
    ap.add_argument('--bumpy-slices', type=int, default=256)
    ap.add_argument('--bumpy-amplitude', type=float, default=0.15)

    ap.add_argument('--boundaries', default='boundary_texture_stripes_k8,boundary_texture_checker_k8')
    ap.add_argument('--depths', default='4,8')
    ap.add_argument('--label-source', default='wos_supervision')
    ap.add_argument('--cache-presets', default='nano,light')
    ap.add_argument('--train-points-list', default='5000')
    ap.add_argument('--train-steps-per-refresh-list', default='50')
    ap.add_argument('--label-refreshes', type=int, default=4)
    ap.add_argument('--walks-per-label-refresh', type=int, default=16)

    # The reference runner itself needs a cache/training setting because it uses the same executable.
    # These settings are not used for reporting unless the caller inspects the reference runner's own output.
    ap.add_argument('--cache-preset-for-reference', default='nano')
    ap.add_argument('--train-points-for-reference', type=int, default=1024)
    ap.add_argument('--train-steps-per-refresh-for-reference', type=int, default=1)
    ap.add_argument('--reference-depth-m', type=int, default=4)
    ap.add_argument('--reference-nc-hybrid-wpp-list', default='1')
    ap.add_argument('--reference-lmc-allocations', default='16:8')

    ap.add_argument('--slice-width', type=int, default=512)
    ap.add_argument('--slice-height', type=int, default=512)
    ap.add_argument('--slice-view', default='xy')
    ap.add_argument('--slice-plane', default='0')
    ap.add_argument('--reference-wpp', type=int, default=16384)
    ap.add_argument('--reference-chunk-wpp', type=int, default=512)
    ap.add_argument('--pure-wpp-list', default='8,16,32,64,128')
    ap.add_argument('--hybrid-wpp-list', default='1,2,4,8,16')
    ap.add_argument('--lmc-allocations', default='16:8,32:16,32:32,64:32')
    ap.add_argument('--threshold-rmse-list', default='0.05,0.075,0.1,0.15,0.2')
    ap.add_argument('--cubql-build-method', default='sah')
    ap.add_argument('--cubql-leaf-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--jit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = parse_list(args.boundaries)
    depths = parse_list_int(args.depths)

    manifest: Dict[str, Any] = {
        'script': 'run_nc_texture_depth_matrix.py',
        'purpose': 'sweep boundary textures and prefix depths while reusing one high-sample Pure WoS reference per boundary',
        'arguments': vars(args),
        'reference_runs': [],
        'cache_sweep_runs': [],
        'limitations': {
            'reference_is_per_boundary_not_per_depth': True,
            'cache_training_is_repeated_per_cache_sweep_run': True,
            'true_online_progressive_runner': False,
        },
    }

    combined_rows: List[Dict[str, Any]] = []
    combined_threshold_rows: List[Dict[str, Any]] = []

    for boundary in boundaries:
        bname = sanitize_name(boundary)
        boundary_dir = out_dir / bname
        ref_dir = boundary_dir / 'reference'
        ref_csv = ref_dir / 'estimates' / 'reference_pure_estimates.csv'

        ref_cmd = [
            'python3', args.reference_runner,
            '--executable', args.executable,
            '--output-dir', str(ref_dir),
        ]
        # add common arguments, replacing placeholder boundary after construction
        saved_placeholder = getattr(args, 'boundary_placeholder', None)
        setattr(args, 'boundary_placeholder', boundary)
        add_common_geometry_args(ref_cmd, args)
        if saved_placeholder is None:
            delattr(args, 'boundary_placeholder')
        else:
            setattr(args, 'boundary_placeholder', saved_placeholder)

        manifest['reference_runs'].append({'boundary': boundary, 'output_dir': str(ref_dir), 'reference_csv': str(ref_csv), 'command': ref_cmd})
        if not (args.skip_existing and ref_csv.exists()):
            run(ref_cmd, args.dry_run)
        elif args.skip_existing:
            print(f'# skip existing reference: {ref_csv}', flush=True)

        # Copy pure curve from the reference runner, if available, into a boundary-level summary.
        ref_points = read_csv(ref_dir / 'time_mse_points_reference.csv')
        for row in ref_points:
            if row.get('method') == 'pure_wos':
                out = dict(row)
                out['boundary'] = boundary
                out['depth_m'] = 'pure'
                out['source'] = str(ref_dir / 'time_mse_points_reference.csv')
                combined_rows.append(out)

        ref_thresholds = read_csv(ref_dir / 'time_to_threshold_reference.csv')
        for row in ref_thresholds:
            if row.get('method') == 'pure_wos':
                out = dict(row)
                out['boundary'] = boundary
                out['depth_m'] = 'pure'
                out['source'] = str(ref_dir / 'time_to_threshold_reference.csv')
                combined_threshold_rows.append(out)

        for depth in depths:
            depth_dir = boundary_dir / f'm{depth}'
            cache_cmd = [
                'python3', args.cache_sweep_runner,
                '--executable', args.executable,
                '--reference-estimates-csv', str(ref_csv),
                '--output-dir', str(depth_dir),
                '--mesh', args.mesh,
                '--boundary', boundary,
                '--label-source', args.label_source,
                '--cache-presets', args.cache_presets,
                '--train-points-list', args.train_points_list,
                '--train-steps-per-refresh-list', args.train_steps_per_refresh_list,
                '--label-refreshes', str(args.label_refreshes),
                '--walks-per-label-refresh', str(args.walks_per_label_refresh),
                '--depth-m', str(depth),
                '--slice-width', str(args.slice_width),
                '--slice-height', str(args.slice_height),
                '--slice-view', args.slice_view,
                '--slice-plane', str(args.slice_plane),
                '--hybrid-wpp-list', args.hybrid_wpp_list,
                '--lmc-allocations', args.lmc_allocations,
                '--threshold-rmse-list', args.threshold_rmse_list,
                '--cubql-build-method', args.cubql_build_method,
                '--cubql-leaf-size', str(args.cubql_leaf_size),
                '--normalize', str(args.normalize),
                '--jit', str(args.jit),
                '--seed', str(args.seed),
            ]
            if args.mesh_path:
                cache_cmd += ['--mesh-path', args.mesh_path]
            if args.mesh == 'procedural_bumpy_sphere':
                cache_cmd += [
                    '--bumpy-stacks', str(args.bumpy_stacks),
                    '--bumpy-slices', str(args.bumpy_slices),
                    '--bumpy-amplitude', str(args.bumpy_amplitude),
                ]
            if args.skip_existing:
                cache_cmd += ['--skip-existing']

            manifest['cache_sweep_runs'].append({'boundary': boundary, 'depth_m': depth, 'output_dir': str(depth_dir), 'command': cache_cmd})
            run(cache_cmd, args.dry_run)

            points_path = depth_dir / 'cache_quality_time_mse_points_reference.csv'
            for row in read_csv(points_path):
                out = dict(row)
                out['boundary'] = boundary
                out['depth_m'] = depth
                out['source'] = str(points_path)
                combined_rows.append(out)

            threshold_path = depth_dir / 'cache_quality_time_to_threshold_reference.csv'
            for row in read_csv(threshold_path):
                out = dict(row)
                out['boundary'] = boundary
                out['depth_m'] = depth
                out['source'] = str(threshold_path)
                combined_threshold_rows.append(out)

    fields = sorted({k for row in combined_rows for k in row.keys()})
    if combined_rows:
        write_csv(out_dir / 'texture_depth_time_mse_points_reference.csv', combined_rows, fields)
    tfields = sorted({k for row in combined_threshold_rows for k in row.keys()})
    if combined_threshold_rows:
        write_csv(out_dir / 'texture_depth_time_to_threshold_reference.csv', combined_threshold_rows, tfields)

    manifest['combined_outputs'] = {
        'time_mse_points': str(out_dir / 'texture_depth_time_mse_points_reference.csv'),
        'time_to_threshold': str(out_dir / 'texture_depth_time_to_threshold_reference.csv'),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

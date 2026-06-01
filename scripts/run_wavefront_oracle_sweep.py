#!/usr/bin/env python3
"""Run a small pure-WoS vs oracle-2LMC wavefront sweep.

This script is orchestration only. The solver timing path remains inside
n2wos_eval_wavefront_wos. It repeatedly invokes the same executable with the
same mesh/backend options, collects the JSON outputs, and writes a compact
summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values


def parse_csv_floats(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(','):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one number')
    return values


def add_common_solver_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd += [
        '--mesh', args.mesh,
        '--engine', args.engine,
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--step-scale', str(args.step_scale),
        '--x0', args.x0,
        '--seed', str(args.seed),
        '--block-size', str(args.block_size),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
    ]
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


def run_command(cmd: List[str], *, dry_run: bool) -> None:
    print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def variance_time_product(estimator: Dict[str, Any]) -> float:
    return finite_float(estimator.get('estimator_variance')) * finite_float(estimator.get('elapsed_ms_total'))


def run_sample_cost_ms(run: Dict[str, Any]) -> float:
    return 1.0e-3 * finite_float(run.get('us_per_sample'))


def compute_oracle_metrics(doc: Dict[str, Any], pure_score: Optional[float]) -> Dict[str, Any]:
    estimator = doc.get('estimator', {})
    runs = doc.get('runs', {})
    coarse = runs.get('coarse', {})
    residual = runs.get('residual', {})

    actual_score = variance_time_product(estimator)
    out: Dict[str, Any] = {
        'actual_variance_time_product_ms': actual_score,
        'actual_speedup_vs_pure': (pure_score / actual_score) if pure_score and actual_score > 0 else math.nan,
    }

    vc = finite_float(coarse.get('sample_variance'))
    vr = finite_float(residual.get('sample_variance'))
    cc = run_sample_cost_ms(coarse)
    cr = run_sample_cost_ms(residual)
    if vc > 0 and vr > 0 and cc > 0 and cr > 0:
        optimal_ratio = math.sqrt((vc * cr) / (vr * cc))
        optimal_score = (math.sqrt(vc * cc) + math.sqrt(vr * cr)) ** 2
        out.update({
            'measured_coarse_to_residual_sample_ratio': finite_float(coarse.get('samples')) / finite_float(residual.get('samples')),
            'optimal_coarse_to_residual_sample_ratio': optimal_ratio,
            'optimal_variance_time_product_ms_predicted': optimal_score,
            'optimal_speedup_vs_pure_predicted': (pure_score / optimal_score) if pure_score and optimal_score > 0 else math.nan,
            'residual_variance_ratio_vs_coarse': vr / vc,
        })
    return out


def flatten_summary_row(label: str, doc: Dict[str, Any], extra: Dict[str, Any], pure_score: Optional[float]) -> Dict[str, Any]:
    estimator = doc.get('estimator', {})
    options = doc.get('options', {})
    score = variance_time_product(estimator)
    row: Dict[str, Any] = {
        'label': label,
        'method': estimator.get('method', options.get('method')),
        'engine': estimator.get('engine', options.get('engine')),
        'depth_m': options.get('depth_m'),
        'samples': options.get('samples'),
        'coarse_samples': options.get('coarse_samples'),
        'residual_samples': options.get('residual_samples'),
        'mean': estimator.get('mean'),
        'exact': estimator.get('exact'),
        'abs_error': estimator.get('abs_error'),
        'stderr': estimator.get('stderr'),
        'elapsed_ms_total': estimator.get('elapsed_ms_total'),
        'estimator_variance': estimator.get('estimator_variance'),
        'variance_time_product_ms': score,
        'score_speedup_vs_pure': (pure_score / score) if pure_score and score > 0 else math.nan,
        'forced_max_steps_total': 0,
        'overflow_count_total': 0,
        'json_path': extra.get('json_path', ''),
    }
    for run in doc.get('runs', {}).values():
        row['forced_max_steps_total'] += int(run.get('forced_max_steps', 0))
        row['overflow_count_total'] += int(run.get('overflow_count', 0))
    row.update(extra)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='./build/cuda-release-cubql/n2wos_eval_wavefront_wos')
    parser.add_argument('--output-dir', default='results/wavefront_oracle_sweep')
    parser.add_argument('--summary-name', default='summary')
    parser.add_argument('--mesh', default='procedural_bumpy_sphere', choices=['procedural_bumpy_sphere', 'obj', 'ply'])
    parser.add_argument('--engine', default='wavefront', choices=['wavefront', 'persistent'],
                        help='Sampling engine passed to n2wos_eval_wavefront_wos')
    parser.add_argument('--mesh-path', default='')
    parser.add_argument('--normalize', type=int, choices=[0, 1], default=1)
    parser.add_argument('--bumpy-stacks', type=int, default=128)
    parser.add_argument('--bumpy-slices', type=int, default=256)
    parser.add_argument('--bumpy-amplitude', type=float, default=0.15)
    parser.add_argument('--pure-samples', type=int, default=262144)
    parser.add_argument('--residual-samples', type=int, default=65536)
    parser.add_argument('--coarse-ratios', type=parse_csv_floats, default=parse_csv_floats('1,2,4,8'))
    parser.add_argument('--depths', type=parse_csv_ints, default=parse_csv_ints('1,2,4,8,16'))
    parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--step-scale', type=float, default=0.999)
    parser.add_argument('--x0', default='0.1,0.05,0')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--cubql-build-method', default='sah')
    parser.add_argument('--cubql-leaf-size', type=int, default=8)
    parser.add_argument('--skip-pure', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.normalize = bool(args.normalize)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    pure_score: Optional[float] = None

    if not args.skip_pure:
        pure_path = out_dir / 'pure_wos.json'
        cmd = [args.executable, '--method', 'pure_wos', '--samples', str(args.pure_samples), '--output', str(pure_path)]
        add_common_solver_args(cmd, args)
        run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            pure_doc = read_json(pure_path)
            pure_score = variance_time_product(pure_doc.get('estimator', {}))
            rows.append(flatten_summary_row('pure_wos', pure_doc, {'json_path': str(pure_path)}, pure_score))
            records.append({'label': 'pure_wos', 'json_path': str(pure_path), 'document': pure_doc})

    for depth in args.depths:
        for ratio in args.coarse_ratios:
            coarse_samples = max(1, int(round(args.residual_samples * ratio)))
            label = f'oracle_2lmc_m{depth}_r{ratio:g}'
            out_path = out_dir / f'{label}.json'
            cmd = [
                args.executable,
                '--method', 'oracle_2lmc',
                '--coarse-samples', str(coarse_samples),
                '--residual-samples', str(args.residual_samples),
                '--depth-m', str(depth),
                '--output', str(out_path),
            ]
            add_common_solver_args(cmd, args)
            run_command(cmd, dry_run=args.dry_run)
            if args.dry_run:
                continue
            doc = read_json(out_path)
            extra = {'json_path': str(out_path), 'coarse_ratio_requested': ratio}
            extra.update(compute_oracle_metrics(doc, pure_score))
            rows.append(flatten_summary_row(label, doc, extra, pure_score))
            records.append({'label': label, 'json_path': str(out_path), 'document': doc, 'metrics': extra})

    if not args.dry_run:
        csv_path = out_dir / f'{args.summary_name}.csv'
        json_path = out_dir / f'{args.summary_name}.json'
        write_csv(csv_path, rows)
        summary = {
            'schema': 'n2wos_wavefront_oracle_sweep_v1',
            'generated_unix_time': time.time(),
            'executable': args.executable,
            'output_dir': str(out_dir),
            'pure_variance_time_product_ms': pure_score,
            'rows': rows,
            'records': records,
        }
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'wrote {csv_path}')
        print(f'wrote {json_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

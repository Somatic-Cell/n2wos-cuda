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
from typing import Any, Dict, Iterable, List, Optional


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values




def parse_csv_uint64s(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(','):
        token = token.strip()
        if token:
            value = int(token, 0)
            if value < 0:
                raise argparse.ArgumentTypeError('seeds must be non-negative')
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError('expected at least one seed')
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


def add_common_solver_args(cmd: List[str], args: argparse.Namespace, seed: Optional[int] = None) -> None:
    cmd += [
        '--mesh', args.mesh,
        '--engine', args.engine,
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--step-scale', str(args.step_scale),
        '--x0', args.x0,
        '--seed', str(args.seed if seed is None else seed),
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


def z_abs(error: float, stderr: float) -> float:
    if not (math.isfinite(error) and math.isfinite(stderr)) or stderr <= 0:
        return math.nan
    return abs(error) / stderr


def run_error_against(run: Dict[str, Any], target: float) -> float:
    return finite_float(run.get('mean')) - target


def compute_oracle_metrics(doc: Dict[str, Any], pure_score: Optional[float]) -> Dict[str, Any]:
    estimator = doc.get('estimator', {})
    runs = doc.get('runs', {})
    coarse = runs.get('coarse', {})
    residual = runs.get('residual', {})

    actual_score = variance_time_product(estimator)
    exact = finite_float(estimator.get('exact'))
    estimator_error = finite_float(estimator.get('mean')) - exact
    estimator_stderr = finite_float(estimator.get('stderr'))
    coarse_error = run_error_against(coarse, exact)
    residual_error = run_error_against(residual, 0.0)
    out: Dict[str, Any] = {
        'actual_variance_time_product_ms': actual_score,
        'actual_speedup_vs_pure': (pure_score / actual_score) if pure_score and actual_score > 0 else math.nan,
        'estimator_z_abs_error': z_abs(estimator_error, estimator_stderr),
        'coarse_mean': coarse.get('mean'),
        'coarse_sample_variance': coarse.get('sample_variance'),
        'coarse_estimator_variance': coarse.get('estimator_variance'),
        'coarse_stderr': coarse.get('stderr'),
        'coarse_elapsed_ms': coarse.get('elapsed_ms'),
        'coarse_us_per_sample': coarse.get('us_per_sample'),
        'coarse_mean_steps': coarse.get('mean_steps'),
        'coarse_z_abs_error': z_abs(coarse_error, finite_float(coarse.get('stderr'))),
        'residual_mean': residual.get('mean'),
        'residual_sample_variance': residual.get('sample_variance'),
        'residual_estimator_variance': residual.get('estimator_variance'),
        'residual_stderr': residual.get('stderr'),
        'residual_elapsed_ms': residual.get('elapsed_ms'),
        'residual_us_per_sample': residual.get('us_per_sample'),
        'residual_mean_steps': residual.get('mean_steps'),
        'residual_z_abs_mean': z_abs(residual_error, finite_float(residual.get('stderr'))),
    }

    # Diagnostic only.  The exact oracle residual has expectation zero; a large
    # z-score is normally a seed/noise warning, but if it persists across seeds
    # it is evidence for a coupling or stopping-rule bug.
    out['diagnostic_flag_large_estimator_z'] = bool(out['estimator_z_abs_error'] > 3.0) if math.isfinite(out['estimator_z_abs_error']) else False
    out['diagnostic_flag_large_residual_z'] = bool(out['residual_z_abs_mean'] > 3.0) if math.isfinite(out['residual_z_abs_mean']) else False

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
    estimator_error = finite_float(estimator.get('mean')) - finite_float(estimator.get('exact'))
    estimator_stderr = finite_float(estimator.get('stderr'))
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
        'z_abs_error': z_abs(estimator_error, estimator_stderr),
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



def numeric_mean(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def numeric_min(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return min(xs) if xs else math.nan


def numeric_max(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else math.nan


def numeric_std(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else math.nan
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) * (x - mu) for x in xs) / (len(xs) - 1))


def aggregate_rows_by_label(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labels: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get('label', ''))
        if label not in grouped:
            grouped[label] = []
            labels.append(label)
        grouped[label].append(row)

    numeric_columns = [
        'score_speedup_vs_pure',
        'optimal_speedup_vs_pure_predicted',
        'variance_time_product_ms',
        'actual_variance_time_product_ms',
        'estimator_variance',
        'elapsed_ms_total',
        'abs_error',
        'stderr',
        'z_abs_error',
        'estimator_z_abs_error',
        'residual_z_abs_mean',
        'residual_variance_ratio_vs_coarse',
        'coarse_us_per_sample',
        'residual_us_per_sample',
        'coarse_mean_steps',
        'residual_mean_steps',
    ]

    out: List[Dict[str, Any]] = []
    for label in labels:
        group = grouped[label]
        first = group[0]
        row: Dict[str, Any] = {
            'label': label,
            'method': first.get('method'),
            'engine': first.get('engine'),
            'depth_m': first.get('depth_m'),
            'coarse_ratio_requested': first.get('coarse_ratio_requested'),
            'seed_count': len(group),
        }
        for col in numeric_columns:
            values = [g.get(col) for g in group]
            row[f'{col}_mean'] = numeric_mean(values)
            row[f'{col}_std'] = numeric_std(values)
            row[f'{col}_min'] = numeric_min(values)
            row[f'{col}_max'] = numeric_max(values)
        row['large_estimator_z_count'] = sum(1 for g in group if bool(g.get('diagnostic_flag_large_estimator_z', False)))
        row['large_residual_z_count'] = sum(1 for g in group if bool(g.get('diagnostic_flag_large_residual_z', False)))
        out.append(row)
    return out


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
    parser.add_argument('--seeds', type=parse_csv_uint64s, default=None,
                        help='Comma-separated seeds. If provided, the full sweep is repeated for each seed.')
    parser.add_argument('--seed-count', type=int, default=1,
                        help='When --seeds is omitted, run this many seeds starting from --seed.')
    parser.add_argument('--seed-stride', type=int, default=1000003,
                        help='Seed increment used with --seed-count.')
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--cubql-build-method', default='sah')
    parser.add_argument('--cubql-leaf-size', type=int, default=8)
    parser.add_argument('--skip-pure', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.normalize = bool(args.normalize)
    if args.seeds is not None:
        seeds = args.seeds
    else:
        if args.seed_count < 1:
            raise ValueError('--seed-count must be positive')
        seeds = [int(args.seed + i * args.seed_stride) for i in range(args.seed_count)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    pure_score: Optional[float] = None

    pure_score_by_seed: Dict[int, float] = {}

    for seed_index, seed in enumerate(seeds):
        seed_suffix = '' if len(seeds) == 1 else f'_seed{seed}'

        if not args.skip_pure:
            pure_path = out_dir / f'pure_wos{seed_suffix}.json'
            cmd = [args.executable, '--method', 'pure_wos', '--samples', str(args.pure_samples), '--output', str(pure_path)]
            add_common_solver_args(cmd, args, seed=seed)
            run_command(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                pure_doc = read_json(pure_path)
                pure_score_by_seed[seed] = variance_time_product(pure_doc.get('estimator', {}))
                pure_score = pure_score_by_seed[seed]
                rows.append(flatten_summary_row(
                    'pure_wos',
                    pure_doc,
                    {'json_path': str(pure_path), 'seed': seed, 'seed_index': seed_index},
                    pure_score,
                ))
                records.append({'label': 'pure_wos', 'seed': seed, 'json_path': str(pure_path), 'document': pure_doc})

        seed_pure_score = pure_score_by_seed.get(seed, pure_score)
        for depth in args.depths:
            for ratio in args.coarse_ratios:
                coarse_samples = max(1, int(round(args.residual_samples * ratio)))
                label = f'oracle_2lmc_m{depth}_r{ratio:g}'
                out_path = out_dir / f'{label}{seed_suffix}.json'
                cmd = [
                    args.executable,
                    '--method', 'oracle_2lmc',
                    '--coarse-samples', str(coarse_samples),
                    '--residual-samples', str(args.residual_samples),
                    '--depth-m', str(depth),
                    '--output', str(out_path),
                ]
                add_common_solver_args(cmd, args, seed=seed)
                run_command(cmd, dry_run=args.dry_run)
                if args.dry_run:
                    continue
                doc = read_json(out_path)
                extra = {
                    'json_path': str(out_path),
                    'coarse_ratio_requested': ratio,
                    'seed': seed,
                    'seed_index': seed_index,
                }
                extra.update(compute_oracle_metrics(doc, seed_pure_score))
                rows.append(flatten_summary_row(label, doc, extra, seed_pure_score))
                records.append({'label': label, 'seed': seed, 'json_path': str(out_path), 'document': doc, 'metrics': extra})

    if not args.dry_run:
        csv_path = out_dir / f'{args.summary_name}.csv'
        json_path = out_dir / f'{args.summary_name}.json'
        aggregate_csv_path = out_dir / f'{args.summary_name}_by_label.csv'
        aggregate_rows = aggregate_rows_by_label(rows)
        write_csv(csv_path, rows)
        write_csv(aggregate_csv_path, aggregate_rows)
        summary = {
            'schema': 'n2wos_wavefront_oracle_sweep_v1',
            'generated_unix_time': time.time(),
            'executable': args.executable,
            'output_dir': str(out_dir),
            'pure_variance_time_product_ms': pure_score,
            'seeds': seeds,
            'rows': rows,
            'rows_by_label': aggregate_rows,
            'records': records,
        }
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'wrote {csv_path}')
        print(f'wrote {aggregate_csv_path}')
        print(f'wrote {json_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

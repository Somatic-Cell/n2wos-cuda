#!/usr/bin/env python3
"""Run wavefront/persistent oracle sweeps over multiple RNG seeds.

This script is orchestration only. The timed solver path remains inside
n2wos_eval_wavefront_wos. It invokes scripts/run_wavefront_oracle_sweep.py once
per seed, then aggregates the per-seed summary CSV files. The purpose is to
separate Monte Carlo seed noise from implementation-level bias/cost issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def finite_int(value: Any, default: int = 0) -> int:
    try:
        x = int(float(value))
    except Exception:
        return default
    return x


def maybe_float_or_text(text: str) -> Any:
    if text == '':
        return ''
    try:
        value = float(text)
    except ValueError:
        return text
    return value


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [{k: maybe_float_or_text(v) for k, v in row.items()} for row in reader]


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


def run_command(cmd: List[str], *, dry_run: bool) -> None:
    print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def script_path() -> Path:
    return Path(__file__).resolve().with_name('run_wavefront_oracle_sweep.py')


def build_sweep_cmd(args: argparse.Namespace, seed: int, out_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path()),
        '--executable', args.executable,
        '--output-dir', str(out_dir),
        '--summary-name', 'summary',
        '--mesh', args.mesh,
        '--engine', args.engine,
        '--pure-samples', str(args.pure_samples),
        '--residual-samples', str(args.residual_samples),
        '--coarse-ratios', args.coarse_ratios,
        '--depths', args.depths,
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--step-scale', str(args.step_scale),
        '--x0', args.x0,
        '--seed', str(seed),
        '--block-size', str(args.block_size),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--normalize', '1' if args.normalize else '0',
    ]
    if args.mesh_path:
        cmd += ['--mesh-path', args.mesh_path]
    if args.mesh == 'procedural_bumpy_sphere':
        cmd += [
            '--bumpy-stacks', str(args.bumpy_stacks),
            '--bumpy-slices', str(args.bumpy_slices),
            '--bumpy-amplitude', str(args.bumpy_amplitude),
        ]
    return cmd


def group_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get('label', ''))
        groups.setdefault(label, []).append(row)
    return groups


def mean(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def stdev_sample(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return math.nan
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) * (x - m) for x in vals) / (len(vals) - 1))


def min_finite(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return min(vals) if vals else math.nan


def max_finite(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max(vals) if vals else math.nan


def aggregate_group(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = rows[0]
    n = len(rows)
    est_means = [finite_float(r.get('mean')) for r in rows]
    exacts = [finite_float(r.get('exact')) for r in rows]
    exact = exacts[0] if exacts else math.nan
    errors = [m - exact for m in est_means if math.isfinite(m) and math.isfinite(exact)]
    estimator_vars = [finite_float(r.get('estimator_variance')) for r in rows]
    # Variance of the average over seed-level independent estimators, using each
    # run's own estimator variance. This is not the across-seed sample variance.
    pooled_var_of_seed_mean = sum(v for v in estimator_vars if math.isfinite(v)) / (n * n) if n > 0 else math.nan
    pooled_stderr = math.sqrt(pooled_var_of_seed_mean) if math.isfinite(pooled_var_of_seed_mean) else math.nan
    mean_est = mean(est_means)
    bias_z = ((mean_est - exact) / pooled_stderr) if pooled_stderr and pooled_stderr > 0 else math.nan

    out: Dict[str, Any] = {
        'label': label,
        'method': first.get('method'),
        'engine': first.get('engine'),
        'depth_m': first.get('depth_m'),
        'coarse_ratio_requested': first.get('coarse_ratio_requested'),
        'num_seeds': n,
        'mean_estimate_across_seeds': mean_est,
        'exact': exact,
        'bias_of_seed_mean': mean_est - exact if math.isfinite(mean_est) and math.isfinite(exact) else math.nan,
        'abs_bias_of_seed_mean': abs(mean_est - exact) if math.isfinite(mean_est) and math.isfinite(exact) else math.nan,
        'pooled_stderr_of_seed_mean': pooled_stderr,
        'bias_z_score': bias_z,
        'rmse_of_seed_estimates': math.sqrt(mean([e * e for e in errors])) if errors else math.nan,
        'elapsed_ms_total_mean': mean([finite_float(r.get('elapsed_ms_total')) for r in rows]),
        'elapsed_ms_total_stdev': stdev_sample([finite_float(r.get('elapsed_ms_total')) for r in rows]),
        'estimator_variance_mean': mean(estimator_vars),
        'variance_time_product_ms_mean': mean([finite_float(r.get('variance_time_product_ms')) for r in rows]),
        'variance_time_product_ms_min': min_finite([finite_float(r.get('variance_time_product_ms')) for r in rows]),
        'variance_time_product_ms_max': max_finite([finite_float(r.get('variance_time_product_ms')) for r in rows]),
        'score_speedup_vs_pure_mean': mean([finite_float(r.get('score_speedup_vs_pure')) for r in rows]),
        'score_speedup_vs_pure_min': min_finite([finite_float(r.get('score_speedup_vs_pure')) for r in rows]),
        'score_speedup_vs_pure_max': max_finite([finite_float(r.get('score_speedup_vs_pure')) for r in rows]),
        'actual_speedup_vs_pure_mean': mean([finite_float(r.get('actual_speedup_vs_pure')) for r in rows]),
        'optimal_speedup_vs_pure_predicted_mean': mean([finite_float(r.get('optimal_speedup_vs_pure_predicted')) for r in rows]),
        'optimal_speedup_vs_pure_predicted_max': max_finite([finite_float(r.get('optimal_speedup_vs_pure_predicted')) for r in rows]),
        'optimal_coarse_to_residual_sample_ratio_mean': mean([finite_float(r.get('optimal_coarse_to_residual_sample_ratio')) for r in rows]),
        'residual_variance_ratio_vs_coarse_mean': mean([finite_float(r.get('residual_variance_ratio_vs_coarse')) for r in rows]),
        'forced_max_steps_total_sum': sum(finite_int(r.get('forced_max_steps_total')) for r in rows),
        'overflow_count_total_sum': sum(finite_int(r.get('overflow_count_total')) for r in rows),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='./build/cuda-release-cubql/n2wos_eval_wavefront_wos')
    parser.add_argument('--output-dir', default='results/persistent_oracle_multiseed')
    parser.add_argument('--seeds', type=parse_csv_ints, default=parse_csv_ints('12345,23456,34567,45678,56789'))
    parser.add_argument('--mesh', default='procedural_bumpy_sphere', choices=['procedural_bumpy_sphere', 'obj', 'ply'])
    parser.add_argument('--engine', default='persistent', choices=['wavefront', 'persistent'])
    parser.add_argument('--mesh-path', default='')
    parser.add_argument('--normalize', type=int, choices=[0, 1], default=1)
    parser.add_argument('--bumpy-stacks', type=int, default=128)
    parser.add_argument('--bumpy-slices', type=int, default=256)
    parser.add_argument('--bumpy-amplitude', type=float, default=0.15)
    parser.add_argument('--pure-samples', type=int, default=262144)
    parser.add_argument('--residual-samples', type=int, default=65536)
    parser.add_argument('--coarse-ratios', default='1,2,4,8')
    parser.add_argument('--depths', default='1,2,4,8,16')
    parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--step-scale', type=float, default=0.999)
    parser.add_argument('--x0', default='0.1,0.05,0')
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--cubql-build-method', default='sah')
    parser.add_argument('--cubql-leaf-size', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.normalize = bool(args.normalize)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_seed_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        'schema': 'n2wos_wavefront_oracle_multiseed_v1',
        'generated_unix_time': time.time(),
        'executable': args.executable,
        'engine': args.engine,
        'seeds': args.seeds,
        'output_dir': str(out_dir),
        'seed_runs': [],
    }

    for seed in args.seeds:
        seed_dir = out_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_sweep_cmd(args, seed, seed_dir)
        run_command(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue
        seed_summary = seed_dir / 'summary.csv'
        rows = read_csv(seed_summary)
        for row in rows:
            row['seed'] = seed
            row['seed_summary_path'] = str(seed_summary)
            by_seed_rows.append(row)
        manifest['seed_runs'].append({'seed': seed, 'summary_csv': str(seed_summary), 'rows': len(rows)})

    if args.dry_run:
        return 0

    by_seed_csv = out_dir / 'summary_by_seed.csv'
    write_csv(by_seed_csv, by_seed_rows)

    aggregate_rows = [aggregate_group(label, rows) for label, rows in sorted(group_rows(by_seed_rows).items())]
    aggregate_csv = out_dir / 'summary_aggregate.csv'
    write_csv(aggregate_csv, aggregate_rows)

    best_actual = max(
        (r for r in aggregate_rows if r.get('method') == 'oracle_2lmc'),
        key=lambda r: finite_float(r.get('score_speedup_vs_pure_mean')),
        default=None,
    )
    best_predicted = max(
        (r for r in aggregate_rows if r.get('method') == 'oracle_2lmc'),
        key=lambda r: finite_float(r.get('optimal_speedup_vs_pure_predicted_mean')),
        default=None,
    )

    summary_json = out_dir / 'summary_aggregate.json'
    manifest.update({
        'summary_by_seed_csv': str(by_seed_csv),
        'summary_aggregate_csv': str(aggregate_csv),
        'best_actual_oracle_2lmc': best_actual,
        'best_predicted_oracle_2lmc': best_predicted,
        'aggregate_rows': aggregate_rows,
    })
    with summary_json.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'wrote {by_seed_csv}')
    print(f'wrote {aggregate_csv}')
    print(f'wrote {summary_json}')
    if best_actual:
        print('best actual oracle_2lmc:', best_actual.get('label'), 'mean speedup=', best_actual.get('score_speedup_vs_pure_mean'))
    if best_predicted:
        print('best predicted oracle_2lmc:', best_predicted.get('label'), 'mean predicted speedup=', best_predicted.get('optimal_speedup_vs_pure_predicted_mean'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

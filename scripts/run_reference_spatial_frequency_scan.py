#!/usr/bin/env python3
"""Scan boundary/slice pairs by spatial frequency of the Pure WoS reference.

This runner does not train a Neural Cache and does not run NC+2LMC.  It is a
pre-flight diagnostic: choose boundary/slice settings whose numerical reference
solution has visible interior structure before spending time on cache training
or 2LMC budget sweeps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(',') if x.strip()]


def parse_float_list(text: str) -> List[str]:
    # Preserve the original token string for CLI reproducibility while also
    # validating that it is a float.
    out: List[str] = []
    for tok in parse_list(text):
        float(tok)
        out.append(tok)
    return out


def sanitize(text: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(text))


def run(cmd: List[str], dry_run: bool) -> None:
    print('+', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_estimates(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_ppm(path: Path, values: np.ndarray, mask: np.ndarray, *, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Write a blue-white-red PPM for signed data, gray outside the mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    finite = m & np.isfinite(arr)
    if vmin is None or vmax is None:
        if np.any(finite):
            max_abs = float(np.nanmax(np.abs(arr[finite])))
            if not math.isfinite(max_abs) or max_abs <= 0.0:
                max_abs = 1.0
            vmin = -max_abs
            vmax = max_abs
        else:
            vmin, vmax = -1.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = (arr - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)

    rgb = np.empty(arr.shape + (3,), dtype=np.uint8)
    rgb[..., :] = 32
    # simple blue-white-red ramp
    neg = t < 0.5
    pos = ~neg
    tt = np.zeros_like(t)
    tt[neg] = t[neg] * 2.0
    tt[pos] = (t[pos] - 0.5) * 2.0
    r = np.where(neg, 255.0 * tt, 255.0)
    g = np.where(neg, 255.0 * tt, 255.0 * (1.0 - tt))
    b = np.where(neg, 255.0, 255.0 * (1.0 - tt))
    rgb[finite, 0] = np.clip(r[finite], 0, 255).astype(np.uint8)
    rgb[finite, 1] = np.clip(g[finite], 0, 255).astype(np.uint8)
    rgb[finite, 2] = np.clip(b[finite], 0, 255).astype(np.uint8)

    h, w = arr.shape
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def write_gray_ppm(path: Path, values: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    finite = m & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(finite):
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
        if hi <= lo:
            hi = lo + 1.0
        out[finite] = np.clip(255.0 * (arr[finite] - lo) / (hi - lo), 0, 255).astype(np.uint8)
    rgb = np.repeat(out[..., None], 3, axis=2)
    rgb[~m, :] = 24
    h, w = arr.shape
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def merge_reference_chunks(chunk_csvs: Sequence[Path], chunk_wpps: Sequence[int], out_csv: Path) -> Tuple[List[Dict[str, Any]], List[float], List[float], int]:
    """Merge per-point chunk means/variances with Chan/Welford statistics."""
    if len(chunk_csvs) != len(chunk_wpps):
        raise RuntimeError('chunk path/wpp count mismatch')
    if not chunk_csvs:
        raise RuntimeError('no reference chunks')

    n_total: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    base_rows: Optional[List[Dict[str, str]]] = None

    for csv_path, wpp in zip(chunk_csvs, chunk_wpps):
        rows = read_estimates(csv_path)
        if not rows:
            raise RuntimeError(f'empty reference chunk: {csv_path}')
        vals = np.array([float(r['pure_mean']) for r in rows], dtype=np.float64)
        vars_ = np.array([float(r.get('pure_sample_variance', 0.0) or 0.0) for r in rows], dtype=np.float64)
        count_b = np.full(vals.shape, float(wpp), dtype=np.float64)
        m2_b = np.maximum(0.0, vars_) * max(0.0, float(wpp - 1))

        if mean is None:
            base_rows = rows
            mean = vals.copy()
            m2 = m2_b.copy()
            n_total = count_b.copy()
        else:
            assert n_total is not None and m2 is not None
            if vals.shape != mean.shape:
                raise RuntimeError(f'row count mismatch in {csv_path}')
            delta = vals - mean
            combined_n = n_total + count_b
            mean = mean + delta * (count_b / combined_n)
            m2 = m2 + m2_b + delta * delta * (n_total * count_b / combined_n)
            n_total = combined_n

    assert base_rows is not None and mean is not None and m2 is not None and n_total is not None
    total_wpp = int(round(float(n_total[0])))
    variance = m2 / np.maximum(1.0, n_total - 1.0)

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(base_rows):
        out_rows.append({
            'point_id': r.get('point_id', str(i)),
            'x': r.get('x', ''),
            'y': r.get('y', ''),
            'z': r.get('z', ''),
            'analytic_value': r.get('analytic_value', ''),
            'pure_mean': float(mean[i]),
            'pure_sample_variance': float(variance[i]),
            'reference_wpp': total_wpp,
        })
    write_csv(out_csv, out_rows, ['point_id', 'x', 'y', 'z', 'analytic_value', 'pure_mean', 'pure_sample_variance', 'reference_wpp'])
    return out_rows, [float(x) for x in mean], [float(x) for x in variance], total_wpp


def uv_from_xyz(row: Dict[str, Any], view: str) -> Tuple[float, float]:
    x = float(row['x']); y = float(row['y']); z = float(row['z'])
    if view == 'xy':
        return x, y
    if view == 'xz':
        return x, z
    if view == 'yz':
        return y, z
    raise RuntimeError(f'unsupported view: {view}')


def image_from_rows(rows: Sequence[Dict[str, Any]], values: Sequence[float], view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_vals = []
    v_vals = []
    for r in rows:
        u, v = uv_from_xyz(r, view)
        u_vals.append(round(u, 9))
        v_vals.append(round(v, 9))
    us = np.array(sorted(set(u_vals)), dtype=np.float64)
    vs = np.array(sorted(set(v_vals)), dtype=np.float64)
    u_index = {float(u): i for i, u in enumerate(us.tolist())}
    v_index = {float(v): i for i, v in enumerate(vs.tolist())}
    img = np.full((len(vs), len(us)), np.nan, dtype=np.float64)
    mask = np.zeros_like(img, dtype=bool)
    for r, val in zip(rows, values):
        u, v = uv_from_xyz(r, view)
        ui = u_index[round(u, 9)]
        vi = v_index[round(v, 9)]
        # image row 0 should be top, so invert v for visual convention
        row = len(vs) - 1 - vi
        img[row, ui] = float(val)
        mask[row, ui] = True
    return img, mask, us, vs


def finite_neighbor_diffs(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_x = mask[:, 1:] & mask[:, :-1] & np.isfinite(img[:, 1:]) & np.isfinite(img[:, :-1])
    valid_y = mask[1:, :] & mask[:-1, :] & np.isfinite(img[1:, :]) & np.isfinite(img[:-1, :])
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]
    return dx[valid_x], dy[valid_y]


def high_frequency_ratio(img: np.ndarray, mask: np.ndarray, radius_fraction: float) -> Tuple[float, np.ndarray]:
    data = np.asarray(img, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0, np.zeros_like(data)
    mean = float(np.nanmean(data[m]))
    data[~m | ~np.isfinite(data)] = mean
    data = data - mean
    data[~m] = 0.0
    fft = np.fft.fftshift(np.fft.fft2(data))
    power = np.abs(fft) ** 2
    h, w = data.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    rr = np.sqrt(xx * xx + yy * yy)
    max_r = float(np.nanmax(rr))
    hp = rr >= float(radius_fraction) * max_r
    total = float(np.sum(power))
    ratio = float(np.sum(power[hp]) / total) if total > 0.0 else 0.0
    high = np.real(np.fft.ifft2(np.fft.ifftshift(fft * hp)))
    high[~m] = np.nan
    return ratio, high


def slice_distance_metrics(mask: np.ndarray) -> Dict[str, float]:
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except Exception:
        return {}
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return {}
    d = distance_transform_edt(m)
    vals = d[m].astype(np.float64)
    return {
        'slice_distance_pixels_mean': float(np.mean(vals)),
        'slice_distance_pixels_q10': float(np.quantile(vals, 0.10)),
        'slice_distance_pixels_q50': float(np.quantile(vals, 0.50)),
        'slice_distance_pixels_q90': float(np.quantile(vals, 0.90)),
    }


def compute_case_metrics(rows: List[Dict[str, Any]], means: List[float], variances: List[float], view: str, highpass_fraction: float, case_dir: Path) -> Dict[str, Any]:
    img, mask, us, vs = image_from_rows(rows, means, view)
    vals = img[mask & np.isfinite(img)]
    if vals.size == 0:
        raise RuntimeError('no valid field values')
    dx, dy = finite_neighbor_diffs(img, mask)
    diffs = np.concatenate([dx, dy]) if dx.size or dy.size else np.array([], dtype=np.float64)
    hfr, high_img = high_frequency_ratio(img, mask, highpass_fraction)

    grad = np.full_like(img, np.nan)
    if dx.size or dy.size:
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        count = np.zeros_like(img)
        valid_x = mask[:, 1:] & mask[:, :-1] & np.isfinite(img[:, 1:]) & np.isfinite(img[:, :-1])
        valid_y = mask[1:, :] & mask[:-1, :] & np.isfinite(img[1:, :]) & np.isfinite(img[:-1, :])
        ddx = np.zeros_like(img[:, 1:])
        ddy = np.zeros_like(img[1:, :])
        ddx[valid_x] = img[:, 1:][valid_x] - img[:, :-1][valid_x]
        ddy[valid_y] = img[1:, :][valid_y] - img[:-1, :][valid_y]
        gx[:, 1:][valid_x] += np.abs(ddx[valid_x]); count[:, 1:][valid_x] += 1
        gx[:, :-1][valid_x] += np.abs(ddx[valid_x]); count[:, :-1][valid_x] += 1
        gy[1:, :][valid_y] += np.abs(ddy[valid_y]); count[1:, :][valid_y] += 1
        gy[:-1, :][valid_y] += np.abs(ddy[valid_y]); count[:-1, :][valid_y] += 1
        good = count > 0
        grad[good] = (gx[good] + gy[good]) / count[good]

    write_ppm(case_dir / 'reference_mean.ppm', img, mask)
    write_gray_ppm(case_dir / 'reference_abs_gradient.ppm', np.abs(grad), mask)
    write_ppm(case_dir / 'reference_highpass.ppm', high_img, mask)

    mean_ref_var = float(np.mean(np.asarray(variances, dtype=np.float64))) if variances else 0.0
    metrics: Dict[str, Any] = {
        'grid_width_reconstructed': int(img.shape[1]),
        'grid_height_reconstructed': int(img.shape[0]),
        'inside_pixels': int(vals.size),
        'field_mean': float(np.mean(vals)),
        'field_rms': float(math.sqrt(float(np.mean(vals * vals)))),
        'field_std': float(np.std(vals)),
        'field_min': float(np.min(vals)),
        'field_max': float(np.max(vals)),
        'gradient_rms': float(math.sqrt(float(np.mean(diffs * diffs)))) if diffs.size else 0.0,
        'total_variation_proxy': float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        'high_frequency_energy_ratio': float(hfr),
        'mean_reference_sample_variance': mean_ref_var,
        'reference_rmse_floor_estimate': float(math.sqrt(max(0.0, mean_ref_var / max(1.0, float(rows[0].get('reference_wpp', 1)))))) if rows else 0.0,
        'u_min': float(np.min(us)) if us.size else 0.0,
        'u_max': float(np.max(us)) if us.size else 0.0,
        'v_min': float(np.min(vs)) if vs.size else 0.0,
        'v_max': float(np.max(vs)) if vs.size else 0.0,
    }
    metrics.update(slice_distance_metrics(mask))
    with (case_dir / 'case_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def common_executable_args(args: argparse.Namespace, boundary: str, view: str, plane: str, chunk_wpp: int, seed: int, output_json: Path, output_prefix: Path, slice_prefix: Path) -> List[str]:
    cmd = [
        args.executable,
        '--mesh', args.mesh,
        '--boundary', boundary,
        '--label-source', 'wos_supervision',
        '--cache-preset', 'nano',
        '--train-points', '128',
        '--train-sampler', 'rejection',
        '--label-refreshes', '1',
        '--walks-per-label-refresh', '1',
        '--train-steps-per-refresh', '0',
        '--skip-training', '1',
        '--eval-mode', 'slice',
        '--slice-width', str(args.slice_width),
        '--slice-height', str(args.slice_height),
        '--slice-view', view,
        '--slice-plane', plane,
        '--slice-preserve-world-aspect', str(args.slice_preserve_world_aspect),
        '--slice-padding-fraction', str(args.slice_padding_fraction),
        '--slice-output-prefix', str(slice_prefix),
        '--pure-walks-per-point', str(chunk_wpp),
        '--hybrid-walks-per-point', '1',
        '--enable-2lmc', '0',
        '--coarse-walks-per-point', '1',
        '--residual-walks-per-point', '1',
        '--depth-m', '0',
        '--max-steps', str(args.max_steps),
        '--epsilon', str(args.epsilon),
        '--seed', str(seed),
        '--cubql-build-method', args.cubql_build_method,
        '--cubql-leaf-size', str(args.cubql_leaf_size),
        '--save-estimates-prefix', str(output_prefix),
        '--output', str(output_json),
        '--normalize', str(args.normalize),
        '--jit', str(args.jit),
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


def main() -> int:
    ap = argparse.ArgumentParser(description='Screen boundary/slice choices by Pure WoS reference spatial frequency.')
    ap.add_argument('--executable', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mesh', default='procedural_bumpy_sphere')
    ap.add_argument('--mesh-path', default='')
    ap.add_argument('--normalize', type=int, default=1)
    ap.add_argument('--bumpy-stacks', type=int, default=128)
    ap.add_argument('--bumpy-slices', type=int, default=256)
    ap.add_argument('--bumpy-amplitude', type=float, default=0.15)
    ap.add_argument('--boundaries', default='boundary_texture_stripes_k8,boundary_texture_stripes_k16,boundary_texture_checker_k8,boundary_texture_checker_k16')
    ap.add_argument('--slice-views', default='xy')
    ap.add_argument('--slice-planes', default='0,0.25,0.5,0.7')
    ap.add_argument('--slice-width', type=int, default=512)
    ap.add_argument('--slice-height', type=int, default=512)
    ap.add_argument('--slice-padding-fraction', default='0.02')
    ap.add_argument('--slice-preserve-world-aspect', type=int, default=1)
    ap.add_argument('--reference-wpp', type=int, default=4096)
    ap.add_argument('--reference-chunk-wpp', type=int, default=512)
    ap.add_argument('--reference-seed', type=int, default=987654321)
    ap.add_argument('--highpass-radius-fraction', type=float, default=0.25)
    ap.add_argument('--max-steps', type=int, default=256)
    ap.add_argument('--epsilon', default='1e-4')
    ap.add_argument('--cubql-build-method', default='sah')
    ap.add_argument('--cubql-leaf-size', type=int, default=8)
    ap.add_argument('--jit', type=int, default=0)
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.reference_wpp <= 0:
        raise RuntimeError('--reference-wpp must be positive')
    if args.reference_chunk_wpp <= 0:
        raise RuntimeError('--reference-chunk-wpp must be positive')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = parse_list(args.boundaries)
    views = parse_list(args.slice_views)
    planes = parse_float_list(args.slice_planes)

    manifest: Dict[str, Any] = {
        'script': 'run_reference_spatial_frequency_scan.py',
        'purpose': 'screen boundary/slice pairs by spatial frequency of high-sample Pure WoS reference field',
        'arguments': vars(args),
        'cases': [],
        'limitations': {
            'not_nc_training': True,
            'not_2lmc': True,
            'screening_reference_not_final_benchmark_by_default': args.reference_wpp < 16384,
            'high_frequency_metric_mask_sensitive': True,
        },
    }

    summary_rows: List[Dict[str, Any]] = []
    case_counter = 0
    for boundary in boundaries:
        for view in views:
            for plane in planes:
                case_name = f'{sanitize(boundary)}_{sanitize(view)}_plane_{sanitize(plane)}'
                case_dir = out_dir / case_name
                runs_dir = case_dir / 'runs'
                estimates_dir = case_dir / 'estimates'
                runs_dir.mkdir(parents=True, exist_ok=True)
                estimates_dir.mkdir(parents=True, exist_ok=True)
                combined_csv = estimates_dir / 'reference_pure_estimates.csv'

                chunk_paths: List[Path] = []
                chunk_wpps: List[int] = []
                remaining = args.reference_wpp
                chunk_index = 0
                while remaining > 0:
                    chunk_wpp = min(args.reference_chunk_wpp, remaining)
                    label = f'reference_chunk{chunk_index:03d}_wpp{chunk_wpp}'
                    out_json = runs_dir / f'{label}.json'
                    out_prefix = estimates_dir / label
                    out_csv = Path(str(out_prefix) + '_estimates.csv')
                    slice_prefix = case_dir / 'slice'
                    seed = args.reference_seed + 1000003 * case_counter + 10007 * chunk_index
                    if not (args.skip_existing and out_json.exists() and out_csv.exists()):
                        cmd = common_executable_args(args, boundary, view, plane, chunk_wpp, seed, out_json, out_prefix, slice_prefix)
                        run(cmd, args.dry_run)
                    chunk_paths.append(out_csv)
                    chunk_wpps.append(chunk_wpp)
                    remaining -= chunk_wpp
                    chunk_index += 1

                if args.dry_run:
                    case_counter += 1
                    continue
                if not (args.skip_existing and combined_csv.exists()):
                    rows, means, variances, total_wpp = merge_reference_chunks(chunk_paths, chunk_wpps, combined_csv)
                else:
                    rows = read_estimates(combined_csv)
                    means = [float(r['pure_mean']) for r in rows]
                    variances = [float(r.get('pure_sample_variance', 0.0) or 0.0) for r in rows]
                    total_wpp = int(float(rows[0].get('reference_wpp', args.reference_wpp))) if rows else args.reference_wpp

                metrics = compute_case_metrics(rows, means, variances, view, args.highpass_radius_fraction, case_dir)
                row: Dict[str, Any] = {
                    'boundary': boundary,
                    'slice_view': view,
                    'slice_plane': plane,
                    'case_dir': str(case_dir),
                    'reference_estimates_csv': str(combined_csv),
                    'reference_wpp': total_wpp,
                    'reference_chunks': len(chunk_paths),
                    **metrics,
                }
                summary_rows.append(row)
                manifest['cases'].append({
                    'boundary': boundary,
                    'slice_view': view,
                    'slice_plane': plane,
                    'case_dir': str(case_dir),
                    'reference_estimates_csv': str(combined_csv),
                    'chunk_estimates_csv': [str(p) for p in chunk_paths],
                    'metrics': metrics,
                })
                case_counter += 1

    fields = [
        'boundary', 'slice_view', 'slice_plane', 'case_dir', 'reference_estimates_csv', 'reference_wpp', 'reference_chunks',
        'inside_pixels', 'field_mean', 'field_rms', 'field_std', 'field_min', 'field_max',
        'gradient_rms', 'total_variation_proxy', 'high_frequency_energy_ratio',
        'mean_reference_sample_variance', 'reference_rmse_floor_estimate',
        'slice_distance_pixels_mean', 'slice_distance_pixels_q10', 'slice_distance_pixels_q50', 'slice_distance_pixels_q90',
        'grid_width_reconstructed', 'grid_height_reconstructed', 'u_min', 'u_max', 'v_min', 'v_max',
    ]
    write_csv(out_dir / 'spatial_frequency_summary.csv', summary_rows, fields)
    with (out_dir / 'spatial_frequency_summary.json').open('w', encoding='utf-8') as f:
        json.dump({'rows': summary_rows}, f, indent=2, sort_keys=True)
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f'wrote {out_dir / "spatial_frequency_summary.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

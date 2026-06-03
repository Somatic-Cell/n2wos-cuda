#!/usr/bin/env python3
"""Train/evaluate a TCNN Neural Cache on a fixed slice and render its signal.

This is a diagnostic/visualization tool.  It is intended to answer whether the
cache field itself is smooth, oscillatory, saturated, or mismatched to a
high-sample Pure-WoS reference before spending time on 2LMC timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render NC/iNGP cache predictions on a slice")
    p.add_argument("--executable", default="./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run", type=int, default=1, help="run n2wos_eval_tcnn_nc_wos before rendering")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--estimates-csv", default="", help="existing *_estimates.csv to render when --run 0")
    p.add_argument("--mask-ppm", default="", help="existing *_mask.ppm to render when --run 0")
    p.add_argument("--reference-estimates-csv", default="", help="optional high-sample Pure-WoS reference estimates CSV")
    p.add_argument("--direct-cache-only", type=int, default=0,
                   help="force deterministic direct C_theta(x): depth_m=0, hybrid_wpp=1, no 2LMC")

    # Forwarded solver options.
    p.add_argument("--mesh", default="procedural_bumpy_sphere")
    p.add_argument("--mesh-path", default="")
    p.add_argument("--normalize", type=int, default=1)
    p.add_argument("--bumpy-stacks", type=int, default=128)
    p.add_argument("--bumpy-slices", type=int, default=256)
    p.add_argument("--bumpy-amplitude", type=float, default=0.15)
    p.add_argument("--boundary", default="boundary_texture_stripes_k16")
    p.add_argument("--label-source", default="wos_supervision")
    p.add_argument("--cache-preset", default="nano")
    p.add_argument("--train-points", type=int, default=5000)
    p.add_argument("--label-refreshes", type=int, default=4)
    p.add_argument("--walks-per-label-refresh", type=int, default=16)
    p.add_argument("--train-steps-per-refresh", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--n-levels", type=int, default=None)
    p.add_argument("--n-features-per-level", type=int, default=None)
    p.add_argument("--log2-hashmap-size", type=int, default=None)
    p.add_argument("--base-resolution", type=int, default=None)
    p.add_argument("--per-level-scale", type=float, default=None)
    p.add_argument("--n-neurons", type=int, default=None)
    p.add_argument("--n-hidden-layers", type=int, default=None)
    p.add_argument("--depth-m", type=int, default=0, help="use 0 for direct cache field C_theta(x)")
    p.add_argument("--slice-width", type=int, default=512)
    p.add_argument("--slice-height", type=int, default=512)
    p.add_argument("--slice-view", default="xy", choices=["xy", "xz", "yz"])
    p.add_argument("--slice-plane", type=float, default=0.0)
    p.add_argument("--slice-padding-fraction", type=float, default=0.02)
    p.add_argument("--slice-preserve-world-aspect", type=int, default=1)
    p.add_argument("--pure-walks-per-point", type=int, default=1)
    p.add_argument("--hybrid-walks-per-point", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--epsilon", default="1e-4")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cubql-build-method", default="sah")
    p.add_argument("--cubql-leaf-size", type=int, default=8)
    p.add_argument("--jit", type=int, default=0)

    # Rendering options.
    p.add_argument("--range", dest="value_range", default="auto-symmetric",
                   help="auto, auto-symmetric, or min,max")
    p.add_argument("--percentile-low", type=float, default=1.0)
    p.add_argument("--percentile-high", type=float, default=99.0)
    return p.parse_args()


def run_command(cmd: Sequence[str]) -> None:
    print("[render_nc_cache_slice] running:")
    print(" ".join(cmd))
    subprocess.run(list(cmd), check=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_values(path: Path, key: str) -> List[float]:
    rows = read_csv_rows(path)
    vals: List[float] = []
    for r in rows:
        if key not in r:
            raise KeyError(f"column {key!r} not found in {path}; available: {list(r.keys())}")
        vals.append(float(r[key]))
    return vals


def read_reference_values(path: Path) -> List[float]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    key = "pure_mean" if "pure_mean" in rows[0] else "reference_mean"
    vals: List[float] = []
    for r in rows:
        vals.append(float(r[key]))
    return vals


def read_ppm_mask(path: Path) -> Tuple[int, int, List[int]]:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"{path} is not a binary P6 PPM")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        if maxval != 255:
            raise ValueError(f"unsupported PPM maxval {maxval}")
        data = f.read()
    expected = width * height * 3
    if len(data) != expected:
        raise ValueError(f"PPM size mismatch: expected {expected}, got {len(data)}")
    mask_display = []
    for i in range(width * height):
        r = data[3*i]
        mask_display.append(1 if r > 128 else 0)
    # The C++ mask writer flips y for display. Convert back to original point order.
    mask_original = [0] * (width * height)
    for dy in range(height):
        oy = height - 1 - dy
        for x in range(width):
            mask_original[oy * width + x] = mask_display[dy * width + x]
    return width, height, mask_original


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    t = pos - lo
    return xs[lo] * (1.0 - t) + xs[hi] * t


def choose_range(values: Sequence[float], mode: str, p_lo: float, p_hi: float) -> Tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return -1.0, 1.0
    if "," in mode:
        a, b = mode.split(",", 1)
        lo, hi = float(a), float(b)
        if not hi > lo:
            raise ValueError("invalid explicit range")
        return lo, hi
    lo = percentile(finite, p_lo)
    hi = percentile(finite, p_hi)
    if mode == "auto-symmetric":
        m = max(abs(lo), abs(hi), 1e-12)
        return -m, m
    if mode == "auto":
        if not hi > lo:
            eps = max(abs(lo), 1.0) * 1e-6
            return lo - eps, hi + eps
        return lo, hi
    raise ValueError(f"unknown range mode: {mode}")


def color_diverging(v: float, lo: float, hi: float) -> Tuple[int, int, int]:
    if not math.isfinite(v):
        return (0, 255, 0)
    t = (v - lo) / (hi - lo) if hi > lo else 0.5
    t = max(0.0, min(1.0, t))
    # blue -> white -> red
    if t < 0.5:
        s = t * 2.0
        r = int(255 * s)
        g = int(255 * s)
        b = 255
    else:
        s = (t - 0.5) * 2.0
        r = 255
        g = int(255 * (1.0 - s))
        b = int(255 * (1.0 - s))
    return (r, g, b)


def render_field_ppm(path: Path, width: int, height: int, mask_original: Sequence[int], values: Sequence[float], lo: float, hi: float) -> None:
    if sum(mask_original) != len(values):
        raise ValueError(f"mask inside count {sum(mask_original)} does not match value count {len(values)}")
    grid: List[Optional[float]] = [None] * (width * height)
    k = 0
    for oy in range(height):
        for x in range(width):
            idx = oy * width + x
            if mask_original[idx]:
                grid[idx] = values[k]
                k += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for dy in range(height):
            oy = height - 1 - dy
            for x in range(width):
                v = grid[oy * width + x]
                if v is None:
                    rgb = (20, 20, 20)
                else:
                    rgb = color_diverging(v, lo, hi)
                f.write(bytes(rgb))



def summarize_pair(pred: Sequence[float], ref: Sequence[float]) -> Dict[str, float]:
    if len(pred) != len(ref):
        raise ValueError(f"point count mismatch: {len(pred)} vs {len(ref)}")
    n = len(pred)
    if n == 0:
        return {"count": 0.0, "rmse": 0.0, "mae": 0.0, "mean_error": 0.0, "corr": 0.0,
                "pred_mean": 0.0, "ref_mean": 0.0, "pred_variance": 0.0, "ref_variance": 0.0,
                "error_variance": 0.0, "residual_variance_ratio": 0.0}
    mean_p = sum(pred) / n
    mean_r = sum(ref) / n
    mse = 0.0
    mae = 0.0
    mean_err = 0.0
    vp = 0.0
    vr = 0.0
    ve = 0.0
    cov = 0.0
    for p, r in zip(pred, ref):
        e = p - r
        mse += e * e
        mae += abs(e)
        mean_err += e
        dp = p - mean_p
        dr = r - mean_r
        de = e - (mean_p - mean_r)
        vp += dp * dp
        vr += dr * dr
        ve += de * de
        cov += dp * dr
    denom = max(1, n - 1)
    var_p = vp / denom
    var_r = vr / denom
    var_e = ve / denom
    corr = cov / math.sqrt(max(vp * vr, 1e-30))
    return {
        "count": float(n),
        "rmse": math.sqrt(mse / n),
        "mae": mae / n,
        "mean_error": mean_err / n,
        "corr": corr,
        "pred_mean": mean_p,
        "ref_mean": mean_r,
        "pred_variance": var_p,
        "ref_variance": var_r,
        "error_variance": var_e,
        "residual_variance_ratio": var_e / var_r if var_r > 0.0 else 0.0,
    }


def write_metrics_json(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_range_json(path: Path, entries: Dict[str, Dict[str, float]]) -> None:
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.direct_cache_only:
        args.depth_m = 0
        args.hybrid_walks_per_point = 1
        args.pure_walks_per_point = 1

    prefix = out_dir / "cache_slice"
    output_json = out_dir / "cache_slice_eval.json"
    estimates_csv = Path(args.estimates_csv) if args.estimates_csv else Path(str(prefix) + "_estimates.csv")
    mask_ppm = Path(args.mask_ppm) if args.mask_ppm else Path(str(prefix) + "_mask.ppm")

    if args.run:
        if args.skip_existing and estimates_csv.exists() and mask_ppm.exists() and output_json.exists():
            print("[render_nc_cache_slice] using existing outputs")
        else:
            cmd: List[str] = [
                args.executable,
                "--mesh", args.mesh,
                "--boundary", args.boundary,
                "--label-source", args.label_source,
                "--cache-preset", args.cache_preset,
                "--train-points", str(args.train_points),
                "--eval-mode", "slice",
                "--slice-width", str(args.slice_width),
                "--slice-height", str(args.slice_height),
                "--slice-view", args.slice_view,
                "--slice-plane", str(args.slice_plane),
                "--slice-preserve-world-aspect", str(args.slice_preserve_world_aspect),
                "--slice-padding-fraction", str(args.slice_padding_fraction),
                "--slice-output-prefix", str(prefix),
                "--label-refreshes", str(args.label_refreshes),
                "--walks-per-label-refresh", str(args.walks_per_label_refresh),
                "--train-steps-per-refresh", str(args.train_steps_per_refresh),
                "--pure-walks-per-point", str(args.pure_walks_per_point),
                "--hybrid-walks-per-point", str(args.hybrid_walks_per_point),
                "--enable-2lmc", "0",
                "--coarse-walks-per-point", "1",
                "--residual-walks-per-point", "1",
                "--depth-m", str(args.depth_m),
                "--max-steps", str(args.max_steps),
                "--epsilon", str(args.epsilon),
                "--seed", str(args.seed),
                "--cubql-build-method", args.cubql_build_method,
                "--cubql-leaf-size", str(args.cubql_leaf_size),
                "--output", str(output_json),
                "--save-estimates-prefix", str(prefix),
                "--normalize", str(args.normalize),
                "--jit", str(args.jit),
            ]
            if args.mesh_path:
                cmd.extend(["--mesh-path", args.mesh_path])
            if args.mesh == "procedural_bumpy_sphere":
                cmd.extend([
                    "--bumpy-stacks", str(args.bumpy_stacks),
                    "--bumpy-slices", str(args.bumpy_slices),
                    "--bumpy-amplitude", str(args.bumpy_amplitude),
                ])
            if args.learning_rate is not None:
                cmd.extend(["--learning-rate", str(args.learning_rate)])
            for opt_name, value in [
                ("--n-levels", args.n_levels),
                ("--n-features-per-level", args.n_features_per_level),
                ("--log2-hashmap-size", args.log2_hashmap_size),
                ("--base-resolution", args.base_resolution),
                ("--per-level-scale", args.per_level_scale),
                ("--n-neurons", args.n_neurons),
                ("--n-hidden-layers", args.n_hidden_layers),
            ]:
                if value is not None:
                    cmd.extend([opt_name, str(value)])
            run_command(cmd)

    if not estimates_csv.exists():
        raise FileNotFoundError(estimates_csv)
    if not mask_ppm.exists():
        raise FileNotFoundError(mask_ppm)

    width, height, mask_original = read_ppm_mask(mask_ppm)
    cache = read_values(estimates_csv, "nc_wos_mean")
    pure = read_values(estimates_csv, "pure_mean")
    analytic = read_values(estimates_csv, "analytic_value")

    fields: Dict[str, List[float]] = {
        "cache_nc_wos_mean": cache,
        "pure_mean_debug": pure,
        "boundary_function_extended": analytic,
    }

    if args.reference_estimates_csv:
        ref = read_reference_values(Path(args.reference_estimates_csv))
        if len(ref) != len(cache):
            raise ValueError(f"reference point count {len(ref)} does not match cache point count {len(cache)}")
        fields["reference_pure_mean"] = ref
        fields["cache_minus_reference"] = [c - r for c, r in zip(cache, ref)]
        fields["abs_cache_minus_reference"] = [abs(c - r) for c, r in zip(cache, ref)]

    metrics: Dict[str, Dict[str, float]] = {
        "cache_vs_analytic_value": summarize_pair(cache, analytic),
    }
    if "reference_pure_mean" in fields:
        metrics["cache_vs_reference"] = summarize_pair(cache, fields["reference_pure_mean"])
        metrics["analytic_value_vs_reference"] = summarize_pair(analytic, fields["reference_pure_mean"])
    write_metrics_json(out_dir / "metrics.json", metrics)

    ranges: Dict[str, Dict[str, float]] = {}
    for name, vals in fields.items():
        mode = args.value_range
        if name.startswith("abs_"):
            mode = "auto"
        lo, hi = choose_range(vals, mode, args.percentile_low, args.percentile_high)
        ranges[name] = {"min": lo, "max": hi}
        render_field_ppm(out_dir / f"{name}.ppm", width, height, mask_original, vals, lo, hi)

    # Use a shared scale for cache and reference when possible.
    if "reference_pure_mean" in fields:
        both = fields["cache_nc_wos_mean"] + fields["reference_pure_mean"]
        lo, hi = choose_range(both, args.value_range, args.percentile_low, args.percentile_high)
        render_field_ppm(out_dir / "cache_nc_wos_mean_shared_scale.ppm", width, height, mask_original, fields["cache_nc_wos_mean"], lo, hi)
        render_field_ppm(out_dir / "reference_pure_mean_shared_scale.ppm", width, height, mask_original, fields["reference_pure_mean"], lo, hi)
        ranges["shared_cache_reference"] = {"min": lo, "max": hi}

    manifest = {
        "purpose": "visualize a trained iNGP/tiny-cuda-nn cache on a fixed slice",
        "estimates_csv": str(estimates_csv),
        "mask_ppm": str(mask_ppm),
        "reference_estimates_csv": args.reference_estimates_csv,
        "run_eval": bool(args.run),
        "depth_m": args.depth_m,
        "direct_cache_only": bool(args.direct_cache_only),
        "note": "Use --direct-cache-only 1 or --depth-m 0 and --hybrid-walks-per-point 1 to visualize deterministic C_theta(x). For m>0, nc_wos_mean visualizes E[C_theta(X_m)] under m-step prefixes.",
        "ranges": ranges,
        "metrics_json": str(out_dir / "metrics.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_range_json(out_dir / "ranges.json", ranges)
    print(f"[render_nc_cache_slice] wrote {out_dir}")


if __name__ == "__main__":
    main()

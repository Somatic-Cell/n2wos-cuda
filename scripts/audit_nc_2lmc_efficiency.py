#!/usr/bin/env python3
"""Audit NC+2LMC result JSON files.

The evaluator already reports most quantities needed to decide whether the
two-level control variate is useful.  This script derives the estimator-level
variance, MSE*time, residual variance ratio, and warning flags.

It is deliberately tolerant of missing fields so it can process old result
JSONs while still flagging incomplete logging.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


NONANALYTIC_TOKENS = (
    "boundary_texture",
    "texture_",
    "checker",
    "stripes",
)

KNOWN_BOUNDARY_MODES = (
    "constant_one",
    "harmonic_x2_minus_y2",
    "harmonic_zebra_k4",
    "harmonic_zebra_k8",
    "harmonic_zebra_k12",
    "external_charges_low",
    "external_charges_medium",
    "external_charges_high",
    "external_charges_shell_k8",
    "external_charges_shell_k16",
    "boundary_texture_checker_k8",
    "boundary_texture_checker_k16",
    "boundary_texture_stripes_k8",
    "boundary_texture_stripes_k16",
)


def _get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or b == 0.0:
        return math.nan
    return a / b


def _first_existing(obj: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        value = _get(obj, path, None)
        if value is not None:
            return value
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_dict(obj: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    for path in paths:
        value = _get(obj, path, None)
        if isinstance(value, dict):
            return value
    return {}


def _parse_filename_metadata(path: Path, mesh: str = "unknown", cache_preset: str = "unknown") -> dict[str, Any]:
    """Recover metadata from filenames like
    procedural_bumpy_sphere_harmonic_zebra_k8_nano_m2_seed12345.json.
    """

    stem = path.stem
    out: dict[str, Any] = {}

    m_seed = re.search(r"_seed(?P<seed>\d+)$", stem)
    if m_seed:
        out["seed"] = int(m_seed.group("seed"))

    m_depth = re.search(r"_m(?P<depth>\d+)_seed\d+$", stem)
    if m_depth:
        out["depth_m"] = int(m_depth.group("depth"))

    prefix = None
    cache = cache_preset if cache_preset and cache_preset != "unknown" else None
    if cache:
        marker = f"_{cache}_m"
        if marker in stem:
            prefix = stem.rsplit(marker, 1)[0]
    if prefix is None:
        m_cache = re.search(r"_(?P<cache>[^_]+)_m\d+_seed\d+$", stem)
        if m_cache:
            cache = m_cache.group("cache")
            prefix = stem[: m_cache.start()]
    if cache and cache_preset == "unknown":
        out["cache_preset"] = cache

    if prefix:
        if mesh and mesh != "unknown" and prefix.startswith(mesh + "_"):
            out["boundary"] = prefix[len(mesh) + 1 :]
        else:
            for boundary in sorted(KNOWN_BOUNDARY_MODES, key=len, reverse=True):
                suffix = "_" + boundary
                if prefix == boundary or prefix.endswith(suffix):
                    out["boundary"] = boundary
                    maybe_mesh = prefix[: -len(suffix)] if prefix.endswith(suffix) else "unknown"
                    if maybe_mesh and mesh == "unknown":
                        out["mesh"] = maybe_mesh
                    break

    return out


def _truth_available(boundary: str) -> bool:
    b = boundary.lower()
    return not any(tok in b for tok in NONANALYTIC_TOKENS)


def _collect_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.exists():
                out.append(p)
    return sorted({p.resolve() for p in out if p.is_file() and p.suffix == ".json"})


def audit_one(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings = _as_dict(data.get("timings_ms", {}))

    boundary = str(_first_existing(data, [
        "options.boundary",
        "options.boundary_mode",
        "options.bc",
        "options.bc_mode",
        "config.boundary",
        "config.boundary_mode",
        "boundary",
        "boundary_mode",
    ], "unknown"))
    cache_preset = str(_first_existing(data, [
        "options.cache_preset",
        "options.cache",
        "config.cache_preset",
        "cache_preset",
    ], "unknown"))
    train_sampler = str(_get(data, "options.train_sampler", "unknown"))
    label_source = str(_get(data, "options.label_source", "unknown"))
    mesh = str(_get(data, "options.mesh", _get(data, "mesh", "unknown")))

    fname_meta = _parse_filename_metadata(path, mesh=mesh, cache_preset=cache_preset)
    if boundary == "unknown" and "boundary" in fname_meta:
        boundary = str(fname_meta["boundary"])
    if cache_preset == "unknown" and "cache_preset" in fname_meta:
        cache_preset = str(fname_meta["cache_preset"])
    if mesh == "unknown" and "mesh" in fname_meta:
        mesh = str(fname_meta["mesh"])

    depth_m = _as_int(_first_existing(data, ["options.depth_m", "config.depth_m", "depth_m"]), 0)
    seed = _as_int(_first_existing(data, ["options.seed", "config.seed", "seed"]), 0)
    if depth_m == 0 and "depth_m" in fname_meta:
        depth_m = int(fname_meta["depth_m"])
    if seed == 0 and "seed" in fname_meta:
        seed = int(fname_meta["seed"])

    pure = _first_dict(data, ["pure", "pure_wos", "runs.pure_wos", "runs.pure"])
    nc_only = _first_dict(data, ["nc_only", "hybrid", "nc_wos", "runs.nc_wos", "runs.nc_only", "runs.hybrid"])
    nc2_paths = []
    if depth_m > 0:
        nc2_paths.append(f"runs.nc_2lmc_m{depth_m}")
    nc2_paths.extend(["runs.nc_2lmc", "nc_2lmc", "two_level", "nc_plus_2lmc"])
    nc2 = _first_dict(data, nc2_paths)

    pure_wpp = _as_int(_first_existing(data, ["options.pure_walks_per_point", "config.pure_walks_per_point"]), 0)
    hybrid_wpp = _as_int(_first_existing(data, ["options.hybrid_walks_per_point", "config.hybrid_walks_per_point"]), 0)
    coarse_wpp = _as_int(_first_existing(data, ["options.coarse_walks_per_point", "config.coarse_walks_per_point"]), 0)
    residual_wpp = _as_int(_first_existing(data, ["options.residual_walks_per_point", "config.residual_walks_per_point"]), 0)
    pure_wpp = pure_wpp or _as_int(_get(pure, "walks_per_point"), 0)
    hybrid_wpp = hybrid_wpp or _as_int(_get(nc_only, "walks_per_point"), 0)
    coarse_wpp = coarse_wpp or _as_int(_get(nc2, "coarse_walks_per_point"), 0)
    residual_wpp = residual_wpp or _as_int(_get(nc2, "residual_walks_per_point"), 0)

    pure_sample_var = _as_float(_get(pure, "mean_sample_variance"))
    nc_sample_var = _as_float(_get(nc_only, "mean_sample_variance"))
    coarse_sample_var = _as_float(_get(nc2, "mean_coarse_sample_variance"))
    residual_sample_var = _as_float(_get(nc2, "mean_residual_sample_variance"))

    pure_estimator_var = _safe_div(pure_sample_var, pure_wpp)
    nc_estimator_var = _safe_div(nc_sample_var, hybrid_wpp)
    nc2_estimator_var = _safe_div(coarse_sample_var, coarse_wpp) + _safe_div(residual_sample_var, residual_wpp)
    if not math.isfinite(nc2_estimator_var):
        nc2_estimator_var = math.nan

    pure_rmse = _as_float(_get(pure, "rmse"))
    nc_rmse = _as_float(_get(nc_only, "rmse"))
    nc2_rmse = _as_float(_get(nc2, "rmse"))

    pure_time = _as_float(_first_existing(data, [
        "timings_ms.pure",
        "timings_ms.pure_wos",
        "pure_elapsed_ms",
        "pure.elapsed_ms",
        "pure_wos.elapsed_ms",
        "runs.pure_wos.elapsed_ms",
        "runs.pure.elapsed_ms",
    ]))
    if not math.isfinite(pure_time):
        pure_time = _as_float(_get(pure, "elapsed_ms"))

    nc_time = _as_float(_first_existing(data, [
        "timings_ms.hybrid_total",
        "timings_ms.nc_only_total",
        "nc_only.elapsed_ms",
        "nc_wos.elapsed_ms",
        "runs.nc_wos.elapsed_ms",
        "runs.nc_only.elapsed_ms",
        "runs.hybrid.elapsed_ms",
        "hybrid_elapsed_ms",
    ]))
    if not math.isfinite(nc_time):
        nc_time = _as_float(_get(nc_only, "elapsed_ms"))

    nc2_time = _as_float(_first_existing(data, [
        "timings_ms.nc_2lmc_inference_total",
        "timings_ms.nc_2lmc_total",
        "nc_2lmc.elapsed_ms",
        "runs.nc_2lmc.elapsed_ms",
        "two_level_elapsed_ms",
    ]))
    if not math.isfinite(nc2_time):
        nc2_time = _as_float(_get(nc2, "elapsed_ms"))

    # Fall back to common phase names if the evaluator reports split phases.
    if not math.isfinite(nc_time):
        parts = [
            _as_float(timings.get("hybrid_prefix")),
            _as_float(timings.get("hybrid_inference")),
            _as_float(timings.get("hybrid_combine")),
        ]
        if any(math.isfinite(x) for x in parts):
            nc_time = sum(x for x in parts if math.isfinite(x))

    if not math.isfinite(nc2_time):
        parts = [
            _as_float(timings.get("coarse_prefix")),
            _as_float(timings.get("coarse_inference")),
            _as_float(timings.get("coarse_combine")),
            _as_float(timings.get("residual_prefix_continue")),
            _as_float(timings.get("residual_inference")),
            _as_float(timings.get("residual_combine")),
        ]
        if any(math.isfinite(x) for x in parts):
            nc2_time = sum(x for x in parts if math.isfinite(x))

    residual_var_ratio = _safe_div(residual_sample_var, pure_sample_var)
    nc2_vs_pure_estimator_var = _safe_div(nc2_estimator_var, pure_estimator_var)

    pure_mse_time = pure_rmse * pure_rmse * pure_time if math.isfinite(pure_rmse) and math.isfinite(pure_time) else math.nan
    nc2_mse_time = nc2_rmse * nc2_rmse * nc2_time if math.isfinite(nc2_rmse) and math.isfinite(nc2_time) else math.nan
    nc2_mse_time_ratio = _safe_div(nc2_mse_time, pure_mse_time)

    coarse_steps = _as_float(_get(nc2, "mean_coarse_steps"))
    residual_steps = _as_float(_get(nc2, "mean_residual_steps"))
    # Approximate optimal allocation using step counts as cost proxies.
    # Nc/Nr = sqrt(Vc * Cr / (Vr * Cc)).
    optimal_coarse_per_residual = math.nan
    if all(math.isfinite(x) and x > 0.0 for x in (coarse_sample_var, residual_sample_var, coarse_steps, residual_steps)):
        optimal_coarse_per_residual = math.sqrt((coarse_sample_var * residual_steps) / (residual_sample_var * coarse_steps))

    analytic_truth = _truth_available(boundary)

    warnings: list[str] = []
    if train_sampler != "rejection":
        warnings.append(f"train_sampler={train_sampler}; expected rejection for paper-like NC labels")
    if label_source != "wos_supervision":
        warnings.append(f"label_source={label_source}; expected wos_supervision")
    if not analytic_truth:
        warnings.append("non-analytic boundary mode: RMSE/mean_bias need numerical Pure WoS reference")
    if math.isfinite(residual_var_ratio) and residual_var_ratio > 0.8 and depth_m >= 4:
        warnings.append(f"residual_var_ratio={residual_var_ratio:.3g} remains high at m={depth_m}")
    if math.isfinite(nc2_mse_time_ratio) and nc2_mse_time_ratio > 1.0:
        warnings.append(f"NC+2LMC MSE*time is worse than Pure WoS by {nc2_mse_time_ratio:.3g}x")
    if depth_m <= 1:
        warnings.append("depth_m<=1: include m=2,4,8 before concluding cache is ineffective")
    if boundary == "unknown":
        warnings.append("boundary=unknown; JSON key missing and filename fallback failed")
    if seed == 0:
        warnings.append("seed=0 or missing; check JSON key or filename fallback")
    if not math.isfinite(pure_rmse):
        warnings.append("missing pure RMSE")
    if not math.isfinite(nc2_rmse):
        warnings.append("missing NC+2LMC RMSE")
    if not math.isfinite(pure_time):
        warnings.append("missing Pure WoS elapsed_ms")
    if not math.isfinite(nc2_time):
        warnings.append("missing NC+2LMC elapsed_ms")

    return {
        "file": path.name,
        "path": str(path),
        "mesh": mesh,
        "boundary": boundary,
        "analytic_truth_available": analytic_truth,
        "cache_preset": cache_preset,
        "seed": seed,
        "depth_m": depth_m,
        "train_sampler": train_sampler,
        "label_source": label_source,
        "pure_wpp": pure_wpp,
        "hybrid_wpp": hybrid_wpp,
        "coarse_wpp": coarse_wpp,
        "residual_wpp": residual_wpp,
        "pure_rmse": pure_rmse,
        "nc_only_rmse": nc_rmse,
        "nc_2lmc_rmse": nc2_rmse,
        "pure_sample_variance": pure_sample_var,
        "nc_only_sample_variance": nc_sample_var,
        "coarse_sample_variance": coarse_sample_var,
        "residual_sample_variance": residual_sample_var,
        "pure_estimator_variance": pure_estimator_var,
        "nc_only_estimator_variance": nc_estimator_var,
        "nc_2lmc_estimator_variance": nc2_estimator_var,
        "residual_variance_ratio_vs_pure": residual_var_ratio,
        "nc_2lmc_estimator_variance_ratio_vs_pure": nc2_vs_pure_estimator_var,
        "pure_elapsed_ms": pure_time,
        "nc_only_elapsed_ms": nc_time,
        "nc_2lmc_elapsed_ms": nc2_time,
        "pure_mse_time": pure_mse_time,
        "nc_2lmc_mse_time": nc2_mse_time,
        "nc_2lmc_mse_time_ratio_vs_pure": nc2_mse_time_ratio,
        "mean_coarse_steps": coarse_steps,
        "mean_residual_steps": residual_steps,
        "optimal_coarse_per_residual_step_proxy": optimal_coarse_per_residual,
        "warning_count": len(warnings),
        "warnings": "; ".join(warnings),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON files or glob patterns")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--strict", action="store_true", help="Return non-zero if any warning is emitted")
    args = ap.parse_args(argv)

    inputs = _collect_inputs(args.inputs)
    if not inputs:
        print("No JSON inputs matched", file=sys.stderr)
        return 2

    rows = [audit_one(p) for p in inputs]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    warning_rows = [r for r in rows if int(r["warning_count"]) > 0]
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Rows with warnings: {len(warning_rows)}")
    for row in warning_rows[:20]:
        print(f"WARN {os.path.relpath(row['path'])}: {row['warnings']}")
    if len(warning_rows) > 20:
        print(f"... {len(warning_rows) - 20} more warning rows")

    return 1 if args.strict and warning_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())

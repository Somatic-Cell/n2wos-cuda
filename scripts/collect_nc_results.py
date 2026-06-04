#!/usr/bin/env python3
"""Collect n2wos neural-cache / 2LMC experiment JSON files into one table.

This script is intentionally schema-tolerant.  It understands the current
`n2wos_tcnn_nc_wos_eval_v4` JSON layout, where solver metrics live under
`runs.{pure_wos,nc_wos,nc_2lmc,nc_2lmc_mX}`, and it also falls back to command
line / filename parsing for fields that older JSONs did not record explicitly.

Typical use:

    python scripts/collect_nc_results.py \
      --inputs "results/**/*.json" \
      --out-csv results/nc_results_summary.csv \
      --out-json results/nc_results_summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CSV_FIELDS: List[str] = [
    "path",
    "valid_result",
    "schema",
    "generated_at_utc",
    "mesh",
    "mesh_path",
    "boundary",
    "analytic_truth_available",
    "label_source",
    "train_sampler",
    "cache_preset",
    "seed",
    "depth_m",
    "eval_mode",
    "eval_use_train_points",
    "slice_view",
    "slice_plane",
    "slice_width",
    "slice_height",
    "slice_inside_pixels",
    "train_points_requested",
    "train_points_padded",
    "train_unique_points",
    "train_acceptance_rate",
    "eval_points",
    "network",
    "n_levels",
    "n_features_per_level",
    "log2_hashmap_size",
    "base_resolution",
    "per_level_scale",
    "n_neurons",
    "n_hidden_layers",
    "learning_rate",
    "label_refreshes",
    "walks_per_label_refresh",
    "train_steps_per_refresh",
    "total_optimizer_steps",
    "total_label_walks_per_point",
    "label_update_ms",
    "tcnn_training_ms",
    "total_training_ms",
    "pure_wpp",
    "hybrid_wpp",
    "coarse_wpp",
    "residual_wpp",
    "pure_rmse",
    "nc_wos_rmse",
    "nc_2lmc_rmse",
    "pure_mae",
    "nc_wos_mae",
    "nc_2lmc_mae",
    "pure_max_abs_error",
    "nc_wos_max_abs_error",
    "nc_2lmc_max_abs_error",
    "pure_mean_bias",
    "nc_wos_mean_bias",
    "nc_2lmc_mean_bias",
    "pure_sample_variance",
    "nc_wos_sample_variance",
    "coarse_sample_variance",
    "residual_sample_variance",
    "pure_estimator_variance",
    "nc_wos_estimator_variance",
    "nc_2lmc_estimator_variance",
    "residual_variance_ratio_vs_pure",
    "nc_2lmc_estimator_variance_ratio_vs_pure",
    "pure_elapsed_ms",
    "nc_wos_elapsed_ms",
    "nc_2lmc_elapsed_ms",
    "nc_2lmc_training_plus_elapsed_ms",
    "pure_us_per_point",
    "nc_wos_us_per_point",
    "nc_2lmc_us_per_point",
    "pure_mse_time",
    "nc_wos_mse_time",
    "nc_2lmc_mse_time",
    "nc_wos_rmse_div_pure_wos_rmse",
    "nc_2lmc_rmse_div_pure_wos_rmse",
    "nc_2lmc_rmse_div_nc_wos_rmse",
    "pure_elapsed_div_nc_2lmc_inference_elapsed",
    "pure_elapsed_div_nc_2lmc_total_with_training",
    "pure_elapsed_div_nc_inference_elapsed",
    "pure_elapsed_div_nc_total_with_training",
    "mean_pure_steps",
    "mean_nc_wos_steps",
    "mean_coarse_steps",
    "mean_residual_steps",
    "pure_forced_max_steps",
    "nc_wos_forced_max_steps",
    "coarse_forced_max_steps",
    "residual_forced_max_steps",
    "pure_overflow_count",
    "nc_wos_overflow_count",
    "coarse_overflow_count",
    "residual_overflow_count",
    "cache_query_fraction_nc_wos",
    "cache_query_fraction_coarse",
    "cache_query_fraction_residual",
    "estimates_csv",
    "slice_points_csv",
    "slice_mask_ppm",
    "warning_count",
    "warnings",
]


KNOWN_BOUNDARY_PREFIXES = [
    "harmonic_zebra_k",
    "harmonic_x2_minus_y2",
    "external_charges_shell_k",
    "external_charges_high",
    "external_charges_medium",
    "external_charges_low",
    "boundary_texture_checker_k",
    "boundary_texture_stripes_k",
    "constant_one",
]


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
        return math.nan
    return num / den


def mse_time(rmse: float, elapsed_ms: float) -> float:
    if not math.isfinite(rmse) or not math.isfinite(elapsed_ms):
        return math.nan
    return rmse * rmse * elapsed_ms


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        print(f"[collect_nc_results] warning: failed to read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def parse_command_line(command_line: str) -> Dict[str, str]:
    if not command_line:
        return {}
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        tokens = command_line.split()
    out: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = "1"
            i += 1
    return out


def filename_seed(path: Path) -> Optional[int]:
    m = re.search(r"(?:^|_)seed(\d+)(?:\.|_|$)", path.name)
    return int(m.group(1)) if m else None


def filename_depth(path: Path) -> Optional[int]:
    m = re.search(r"(?:^|_)m(\d+)(?:_|\.|$)", path.name)
    return int(m.group(1)) if m else None


def infer_boundary_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    for prefix in KNOWN_BOUNDARY_PREFIXES:
        if prefix.endswith("_k"):
            m = re.search(re.escape(prefix) + r"\d+", stem)
            if m:
                return m.group(0)
        elif prefix in stem:
            return prefix
    # Common fallback: mesh_boundary_cache_mX_seedY.  Preserve middle segment
    # when the filename follows this repo's generated naming convention.
    m = re.search(r"procedural_bumpy_sphere_(.+?)_(?:nano|light|baseline|paper_like|custom)_m\d+_seed\d+", stem)
    if m:
        return m.group(1)
    m = re.search(r"bunny.*?_(.+?)_(?:nano|light|baseline|paper_like|custom)_m\d+_seed\d+", stem)
    if m:
        return m.group(1)
    return None


def get_run(runs: Mapping[str, Any], depth_m: int) -> Mapping[str, Any]:
    if not isinstance(runs, Mapping):
        return {}
    preferred = f"nc_2lmc_m{depth_m}"
    value = runs.get(preferred)
    if isinstance(value, Mapping):
        return value
    value = runs.get("nc_2lmc")
    if isinstance(value, Mapping):
        return value
    return {}


def get_mapping(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = obj.get(key, {})
    return value if isinstance(value, Mapping) else {}


def extract_result(path: Path, data: Dict[str, Any], args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    command_opts = parse_command_line(str(data.get("command_line", "")))
    options = get_mapping(data, "options")
    runs = get_mapping(data, "runs")

    if not runs:
        skipped = {"path": str(path), "reason": "missing runs object"}
        return None, skipped

    depth_m = as_int(options.get("depth_m"), as_int(command_opts.get("depth_m"), filename_depth(path) or 0))
    pure = get_mapping(runs, "pure_wos") or get_mapping(runs, "pure")
    nc = get_mapping(runs, "nc_wos") or get_mapping(runs, "nc_only") or get_mapping(runs, "hybrid")
    two = get_run(runs, depth_m)
    comp = get_mapping(data, "comparison")
    training = get_mapping(data, "training")
    sampler = get_mapping(data, "training_point_sampler")
    slice_eval = get_mapping(data, "slice_eval")
    estimate_outputs = get_mapping(data, "estimate_outputs")
    mesh_stats = get_mapping(data, "mesh_stats")
    impl = get_mapping(data, "implementation_mode")

    boundary = (
        options.get("boundary_condition")
        or options.get("boundary")
        or command_opts.get("boundary")
        or command_opts.get("boundary_condition")
        or infer_boundary_from_filename(path)
        or "unknown"
    )
    seed = as_int(options.get("seed"), as_int(command_opts.get("seed"), filename_seed(path) or 0))
    mesh = str(options.get("mesh") or command_opts.get("mesh") or mesh_stats.get("name") or "unknown")
    mesh_path = str(options.get("mesh_path") or command_opts.get("mesh_path") or "")
    cache_preset = str(options.get("cache_preset") or command_opts.get("cache_preset") or "unknown")
    label_source = str(options.get("label_source") or command_opts.get("label_source") or impl.get("training_labels") or "unknown")
    train_sampler = str(options.get("train_sampler") or command_opts.get("train_sampler") or sampler.get("mode") or "unknown")

    pure_wpp = as_int(pure.get("walks_per_point"), as_int(options.get("pure_walks_per_point"), 0))
    hybrid_wpp = as_int(nc.get("walks_per_point"), as_int(options.get("hybrid_walks_per_point"), 0))
    coarse_wpp = as_int(two.get("coarse_walks_per_point"), as_int(options.get("coarse_walks_per_point"), 0))
    residual_wpp = as_int(two.get("residual_walks_per_point"), as_int(options.get("residual_walks_per_point"), 0))

    pure_sample_var = as_float(pure.get("mean_sample_variance"))
    nc_sample_var = as_float(nc.get("mean_sample_variance"))
    coarse_sample_var = as_float(two.get("mean_coarse_sample_variance"))
    residual_sample_var = as_float(two.get("mean_residual_sample_variance"))
    pure_est_var = safe_div(pure_sample_var, pure_wpp)
    nc_est_var = safe_div(nc_sample_var, hybrid_wpp)
    two_est_var = safe_div(coarse_sample_var, coarse_wpp) + safe_div(residual_sample_var, residual_wpp)
    if not math.isfinite(two_est_var):
        two_est_var = math.nan

    pure_rmse = as_float(pure.get("rmse"))
    nc_rmse = as_float(nc.get("rmse"))
    two_rmse = as_float(two.get("rmse"))
    pure_elapsed = as_float(pure.get("elapsed_ms"))
    nc_elapsed = as_float(nc.get("elapsed_ms"))
    two_elapsed = as_float(two.get("elapsed_ms"))

    label_refreshes = as_int(options.get("label_refreshes"), as_int(command_opts.get("label_refreshes"), 0))
    walks_per_label_refresh = as_int(options.get("walks_per_label_refresh"), as_int(command_opts.get("walks_per_label_refresh"), 0))
    train_steps_per_refresh = as_int(options.get("train_steps_per_refresh"), as_int(command_opts.get("train_steps_per_refresh"), 0))

    row: Dict[str, Any] = {
        "path": str(path),
        "valid_result": True,
        "schema": data.get("schema", ""),
        "generated_at_utc": data.get("generated_at_utc", ""),
        "mesh": mesh,
        "mesh_path": mesh_path,
        "boundary": boundary,
        "analytic_truth_available": is_analytic_boundary(boundary),
        "label_source": label_source,
        "train_sampler": train_sampler,
        "cache_preset": cache_preset,
        "seed": seed,
        "depth_m": depth_m,
        "eval_mode": options.get("eval_mode", ""),
        "eval_use_train_points": bool(options.get("eval_use_train_points", False)),
        "slice_view": slice_eval.get("view", ""),
        "slice_plane": as_float(slice_eval.get("plane")),
        "slice_width": as_int(slice_eval.get("width"), 0),
        "slice_height": as_int(slice_eval.get("height"), 0),
        "slice_inside_pixels": as_int(slice_eval.get("inside_pixels"), 0),
        "train_points_requested": as_int(options.get("train_points_requested"), 0),
        "train_points_padded": as_int(options.get("train_points_padded"), 0),
        "train_unique_points": as_int(sampler.get("unique_points"), 0),
        "train_acceptance_rate": as_float(sampler.get("acceptance_rate")),
        "eval_points": as_int(options.get("eval_points"), as_int(pure.get("eval_points"), 0)),
        "network": options.get("network", ""),
        "n_levels": as_int(options.get("n_levels"), 0),
        "n_features_per_level": as_int(options.get("n_features_per_level"), 0),
        "log2_hashmap_size": as_int(options.get("log2_hashmap_size"), 0),
        "base_resolution": as_int(options.get("base_resolution"), 0),
        "per_level_scale": as_float(options.get("per_level_scale")),
        "n_neurons": as_int(options.get("n_neurons"), 0),
        "n_hidden_layers": as_int(options.get("n_hidden_layers"), 0),
        "learning_rate": as_float(options.get("learning_rate"), as_float(command_opts.get("learning_rate"))),
        "label_refreshes": label_refreshes,
        "walks_per_label_refresh": walks_per_label_refresh,
        "train_steps_per_refresh": train_steps_per_refresh,
        "total_optimizer_steps": label_refreshes * train_steps_per_refresh,
        "total_label_walks_per_point": label_refreshes * walks_per_label_refresh,
        "label_update_ms": as_float(training.get("label_update_ms")),
        "tcnn_training_ms": as_float(training.get("tcnn_training_ms")),
        "total_training_ms": as_float(training.get("total_training_ms")),
        "pure_wpp": pure_wpp,
        "hybrid_wpp": hybrid_wpp,
        "coarse_wpp": coarse_wpp,
        "residual_wpp": residual_wpp,
        "pure_rmse": pure_rmse,
        "nc_wos_rmse": nc_rmse,
        "nc_2lmc_rmse": two_rmse,
        "pure_mae": as_float(pure.get("mae")),
        "nc_wos_mae": as_float(nc.get("mae")),
        "nc_2lmc_mae": as_float(two.get("mae")),
        "pure_max_abs_error": as_float(pure.get("max_abs_error")),
        "nc_wos_max_abs_error": as_float(nc.get("max_abs_error")),
        "nc_2lmc_max_abs_error": as_float(two.get("max_abs_error")),
        "pure_mean_bias": as_float(pure.get("mean_bias")),
        "nc_wos_mean_bias": as_float(nc.get("mean_bias"), as_float(comp.get("nc_wos_mean_bias"))),
        "nc_2lmc_mean_bias": as_float(two.get("mean_bias"), as_float(comp.get("nc_2lmc_mean_bias"))),
        "pure_sample_variance": pure_sample_var,
        "nc_wos_sample_variance": nc_sample_var,
        "coarse_sample_variance": coarse_sample_var,
        "residual_sample_variance": residual_sample_var,
        "pure_estimator_variance": pure_est_var,
        "nc_wos_estimator_variance": nc_est_var,
        "nc_2lmc_estimator_variance": two_est_var,
        "residual_variance_ratio_vs_pure": safe_div(residual_sample_var, pure_sample_var),
        "nc_2lmc_estimator_variance_ratio_vs_pure": safe_div(two_est_var, pure_est_var),
        "pure_elapsed_ms": pure_elapsed,
        "nc_wos_elapsed_ms": nc_elapsed,
        "nc_2lmc_elapsed_ms": two_elapsed,
        "nc_2lmc_training_plus_elapsed_ms": as_float(two.get("training_plus_elapsed_ms")),
        "pure_us_per_point": as_float(pure.get("us_per_point")),
        "nc_wos_us_per_point": as_float(nc.get("us_per_point")),
        "nc_2lmc_us_per_point": as_float(two.get("us_per_point")),
        "pure_mse_time": mse_time(pure_rmse, pure_elapsed),
        "nc_wos_mse_time": mse_time(nc_rmse, nc_elapsed),
        "nc_2lmc_mse_time": mse_time(two_rmse, two_elapsed),
        "nc_wos_rmse_div_pure_wos_rmse": as_float(comp.get("nc_wos_rmse_div_pure_wos_rmse"), safe_div(nc_rmse, pure_rmse)),
        "nc_2lmc_rmse_div_pure_wos_rmse": as_float(comp.get("nc_2lmc_rmse_div_pure_wos_rmse"), safe_div(two_rmse, pure_rmse)),
        "nc_2lmc_rmse_div_nc_wos_rmse": as_float(comp.get("nc_2lmc_rmse_div_nc_wos_rmse"), safe_div(two_rmse, nc_rmse)),
        "pure_elapsed_div_nc_2lmc_inference_elapsed": as_float(comp.get("pure_elapsed_div_nc_2lmc_inference_elapsed"), safe_div(pure_elapsed, two_elapsed)),
        "pure_elapsed_div_nc_2lmc_total_with_training": as_float(comp.get("pure_elapsed_div_nc_2lmc_total_with_training")),
        "pure_elapsed_div_nc_inference_elapsed": as_float(comp.get("pure_elapsed_div_nc_inference_elapsed"), safe_div(pure_elapsed, nc_elapsed)),
        "pure_elapsed_div_nc_total_with_training": as_float(comp.get("pure_elapsed_div_nc_total_with_training")),
        "mean_pure_steps": as_float(pure.get("mean_steps")),
        "mean_nc_wos_steps": as_float(nc.get("mean_steps")),
        "mean_coarse_steps": as_float(two.get("mean_coarse_steps")),
        "mean_residual_steps": as_float(two.get("mean_residual_steps")),
        "pure_forced_max_steps": as_int(pure.get("forced_max_steps"), 0),
        "nc_wos_forced_max_steps": as_int(nc.get("forced_max_steps"), 0),
        "coarse_forced_max_steps": as_int(two.get("coarse_forced_max_steps"), 0),
        "residual_forced_max_steps": as_int(two.get("residual_forced_max_steps"), 0),
        "pure_overflow_count": as_int(pure.get("overflow_count"), 0),
        "nc_wos_overflow_count": as_int(nc.get("overflow_count"), 0),
        "coarse_overflow_count": as_int(two.get("coarse_overflow_count"), 0),
        "residual_overflow_count": as_int(two.get("residual_overflow_count"), 0),
        "cache_query_fraction_nc_wos": as_float(nc.get("cache_query_fraction")),
        "cache_query_fraction_coarse": as_float(two.get("coarse_cache_query_fraction")),
        "cache_query_fraction_residual": as_float(two.get("residual_cache_query_fraction")),
        "estimates_csv": estimate_outputs.get("estimates_csv", ""),
        "slice_points_csv": slice_eval.get("points_csv", ""),
        "slice_mask_ppm": slice_eval.get("mask_ppm", ""),
    }

    warnings: List[str] = []
    if boundary == "unknown":
        warnings.append("boundary=unknown")
    if not pure or not nc or not two:
        warnings.append("missing one or more expected runs: pure_wos/nc_wos/nc_2lmc")
    if not math.isfinite(pure_rmse) or not math.isfinite(nc_rmse) or not math.isfinite(two_rmse):
        warnings.append("missing RMSE metric")
    if depth_m == 0:
        warnings.append("depth_m=0: NC+2LMC is a bias-correction sanity check, not a variance-reduction test")
    elif depth_m == 1:
        warnings.append("depth_m=1: include m=2,4 before concluding cache is ineffective")
    if str(row["eval_mode"]) == "slice":
        if "bunny" in (mesh + " " + mesh_path).lower():
            z = row["slice_plane"]
            if math.isfinite(z) and abs(z - float(args.bunny_slice_plane_standard)) > 1.0e-6:
                warnings.append(f"bunny slice plane is {z:g} (current paper convention is {args.bunny_slice_plane_standard:g})")
        if not row["estimates_csv"]:
            warnings.append("slice run has no estimates_csv; cannot render/check neural field image")
    if "procedural_bumpy_sphere" in mesh and not args.allow_bumpy_sphere:
        warnings.append("procedural_bumpy_sphere result (current convention is to use Bunny except explicit diagnostics)")
    if label_source == "wos_supervision" and train_steps_per_refresh >= 5000 and walks_per_label_refresh <= 50:
        warnings.append("many optimizer steps per noisy WoS refresh; check noisy-label overfit")

    row["warning_count"] = len(warnings)
    row["warnings"] = "; ".join(warnings)
    return row, None


def is_analytic_boundary(boundary: str) -> bool:
    b = str(boundary)
    if b.startswith("harmonic_"):
        return True
    if b.startswith("external_charges"):
        return True
    if b == "constant_one":
        return True
    return False


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(p)
    unique = sorted({p.resolve() for p in paths if p.is_file()})
    return unique


def summarize(rows: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_boundary = Counter(str(r.get("boundary", "unknown")) for r in rows)
    by_mesh = Counter(str(r.get("mesh", "unknown")) for r in rows)
    by_label = Counter(str(r.get("label_source", "unknown")) for r in rows)
    by_depth = Counter(str(r.get("depth_m", "unknown")) for r in rows)
    by_cache = Counter(str(r.get("cache_preset", "unknown")) for r in rows)
    warning_counter: Counter[str] = Counter()
    for r in rows:
        for w in str(r.get("warnings", "")).split(";"):
            w = w.strip()
            if w:
                warning_counter[w] += 1
    best_by_group: Dict[str, Dict[str, Any]] = {}
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r.get("mesh", "")), str(r.get("boundary", "")), str(r.get("cache_preset", "")), str(r.get("depth_m", "")))
        groups[key].append(r)
    for key, items in groups.items():
        finite = [r for r in items if math.isfinite(as_float(r.get("nc_2lmc_mse_time")))]
        if finite:
            best = min(finite, key=lambda r: as_float(r.get("nc_2lmc_mse_time")))
            best_by_group["|".join(key)] = {
                "path": best.get("path"),
                "nc_2lmc_rmse": best.get("nc_2lmc_rmse"),
                "nc_2lmc_mse_time": best.get("nc_2lmc_mse_time"),
                "seed": best.get("seed"),
            }
    return {
        "num_results": len(rows),
        "num_skipped": len(skipped),
        "by_boundary": dict(by_boundary),
        "by_mesh": dict(by_mesh),
        "by_label_source": dict(by_label),
        "by_depth_m": dict(by_depth),
        "by_cache_preset": dict(by_cache),
        "warnings": dict(warning_counter.most_common()),
        "best_nc_2lmc_mse_time_by_mesh_boundary_cache_depth": best_by_group,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": list(rows), "skipped": list(skipped)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="JSON files or glob patterns; quote globs in the shell")
    p.add_argument("--out-csv", type=Path, help="write collected table as CSV")
    p.add_argument("--out-json", type=Path, help="write collected table + summary as JSON")
    p.add_argument("--out-jsonl", type=Path, help="write one JSON object per result row")
    p.add_argument("--include-invalid", action="store_true", help="include rows for non-experiment JSON files where possible")
    p.add_argument("--allow-bumpy-sphere", action="store_true", help="do not warn on procedural_bumpy_sphere results")
    p.add_argument("--bunny-slice-plane-standard", type=float, default=0.12, help="warn when Bunny slice runs use a different plane")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("no input JSON files matched")

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if data is None:
            skipped.append({"path": str(path), "reason": "failed to parse JSON"})
            continue
        row, skip = extract_result(path, data, args)
        if row is not None:
            rows.append(row)
        elif skip is not None:
            skipped.append(skip)
            if args.include_invalid:
                invalid = {field: "" for field in CSV_FIELDS}
                invalid.update({"path": str(path), "valid_result": False, "warning_count": 1, "warnings": skip.get("reason", "invalid")})
                rows.append(invalid)

    rows.sort(key=lambda r: (str(r.get("mesh")), str(r.get("boundary")), str(r.get("cache_preset")), as_int(r.get("depth_m")), as_int(r.get("seed")), str(r.get("path"))))
    summary = summarize(rows, skipped)

    if args.out_csv is None and args.out_json is None and args.out_jsonl is None:
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)
    if args.out_json is not None:
        write_json(args.out_json, rows, skipped, summary)
    if args.out_jsonl is not None:
        write_jsonl(args.out_jsonl, rows)

    print(
        f"[collect_nc_results] collected {len(rows)} result rows; skipped {len(skipped)} non-result JSON files",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

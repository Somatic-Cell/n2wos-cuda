#!/usr/bin/env python3
"""Run a fixed-slice NC/NC+2LMC allocation sweep.

This script is intentionally an orchestration layer around the existing
n2wos_eval_tcnn_nc_wos executable. It varies coarse/residual walks-per-point
for a fixed slice, cache preset, boundary condition, and prefix depth m.

It also emits a launch-shape audit based on the sample counts and an assumed
CUDA block size. This is not a replacement for Nsight Compute occupancy data;
it is a first check that sample counts and thread-block dimensions are not
obviously hostile to warp execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_bool01(v: bool) -> str:
    return "1" if v else "0"


def parse_allocations(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a, b = item.split(":", 1)
        elif "/" in item:
            a, b = item.split("/", 1)
        else:
            raise ValueError(f"allocation '{item}' must be formatted as coarse:residual")
        coarse = int(a)
        residual = int(b)
        if coarse <= 0 or residual <= 0:
            raise ValueError(f"allocation '{item}' must contain positive integers")
        out.append((coarse, residual))
    if not out:
        raise ValueError("at least one allocation is required")
    return out


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    if any(v <= 0 for v in values):
        raise ValueError("integer list values must be positive")
    return values


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def get_path(d: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def inum(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def audit_launch_shape(samples: int, assumed_block_size: int) -> Dict[str, Any]:
    if samples <= 0:
        return {
            "samples": samples,
            "assumed_block_size": assumed_block_size,
            "valid": False,
            "reason": "non_positive_sample_count",
        }
    block_size = assumed_block_size
    blocks = (samples + block_size - 1) // block_size
    launched_threads = blocks * block_size
    inactive_tail_threads = launched_threads - samples
    tail_fraction = inactive_tail_threads / launched_threads if launched_threads else 0.0
    warps_per_block = block_size / 32.0
    warnings: List[str] = []
    if block_size % 32 != 0:
        warnings.append("block_size_not_multiple_of_warp_size_32")
    if block_size < 64:
        warnings.append("block_size_below_64_may_underutilize_gpu")
    if block_size > 512:
        warnings.append("block_size_above_512_may_reduce_occupancy_for_register_heavy_kernels")
    if tail_fraction > 0.05:
        warnings.append("tail_threads_exceed_5_percent")
    if blocks < 4 * 46:  # RTX 3070 has 46 SMs; this remains only a rough warning.
        warnings.append("grid_has_few_blocks_for_rtx3070_scale_gpu")
    return {
        "samples": samples,
        "assumed_block_size": block_size,
        "warp_size_assumed": 32,
        "warps_per_block": warps_per_block,
        "blocks": blocks,
        "launched_threads": launched_threads,
        "inactive_tail_threads": inactive_tail_threads,
        "inactive_tail_fraction": tail_fraction,
        "warnings": warnings,
        "valid": len([w for w in warnings if w.startswith("block_size_not")]) == 0,
    }


def summarize_result(path: Path, coarse_wpp: int, residual_wpp: int, train_points: int, assumed_block_size: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    options = data.get("options", {})
    training = data.get("training", {})
    runs = data.get("runs", {})
    pure = runs.get("pure_wos", {})
    nc = runs.get("nc_wos", {})
    two = runs.get("nc_2lmc", runs.get(f"nc_2lmc_m{options.get('depth_m', '')}", {}))
    comparison = data.get("comparison", {})

    eval_points = inum(options.get("eval_points"), inum(two.get("eval_points"), inum(nc.get("eval_points"))))
    inside_pixels = inum(get_path(data, ["slice", "inside_pixels"]), eval_points)
    pure_wpp = inum(options.get("pure_walks_per_point"), inum(pure.get("walks_per_point")))
    hybrid_wpp = inum(options.get("hybrid_walks_per_point"), inum(nc.get("walks_per_point")))

    pure_samples = eval_points * pure_wpp
    nc_samples = eval_points * hybrid_wpp
    coarse_samples = eval_points * coarse_wpp
    residual_samples = eval_points * residual_wpp

    launch_audit = {
        "pure_wos": audit_launch_shape(pure_samples, assumed_block_size),
        "nc_wos": audit_launch_shape(nc_samples, assumed_block_size),
        "nc_2lmc_coarse": audit_launch_shape(coarse_samples, assumed_block_size),
        "nc_2lmc_residual": audit_launch_shape(residual_samples, assumed_block_size),
    }

    pure_rmse = fnum(pure.get("rmse"))
    nc_rmse = fnum(nc.get("rmse"))
    two_rmse = fnum(two.get("rmse"))
    pure_elapsed = fnum(pure.get("elapsed_ms"))
    nc_elapsed = fnum(nc.get("elapsed_ms"))
    two_elapsed = fnum(two.get("elapsed_ms"))
    training_ms = fnum(training.get("total_training_ms"))

    return {
        "json_path": str(path),
        "train_points_requested": train_points,
        "train_points_padded": inum(options.get("train_points_padded")),
        "eval_points": eval_points,
        "inside_pixels": inside_pixels,
        "depth_m": inum(options.get("depth_m")),
        "cache_preset": options.get("cache_preset", ""),
        "boundary_condition": options.get("boundary_condition", ""),
        "label_source": options.get("label_source", ""),
        "slice_width": inum(options.get("slice_width"), inum(get_path(data, ["slice", "width"]))),
        "slice_height": inum(options.get("slice_height"), inum(get_path(data, ["slice", "height"]))),
        "coarse_walks_per_point": coarse_wpp,
        "residual_walks_per_point": residual_wpp,
        "pure_walks_per_point": pure_wpp,
        "hybrid_walks_per_point": hybrid_wpp,
        "label_update_ms": fnum(training.get("label_update_ms")),
        "tcnn_training_ms": fnum(training.get("tcnn_training_ms")),
        "total_training_ms": training_ms,
        "pure_rmse": pure_rmse,
        "pure_elapsed_ms": pure_elapsed,
        "nc_wos_rmse": nc_rmse,
        "nc_wos_mean_bias": fnum(nc.get("mean_bias")),
        "nc_wos_elapsed_ms": nc_elapsed,
        "nc_wos_total_ms": fnum(nc.get("training_plus_elapsed_ms"), training_ms + nc_elapsed),
        "nc_2lmc_rmse": two_rmse,
        "nc_2lmc_mean_bias": fnum(two.get("mean_bias")),
        "nc_2lmc_mean_coarse_sample_variance": fnum(two.get("mean_coarse_sample_variance")),
        "nc_2lmc_mean_residual_sample_variance": fnum(two.get("mean_residual_sample_variance")),
        "nc_2lmc_elapsed_ms": two_elapsed,
        "nc_2lmc_total_ms": fnum(two.get("training_plus_elapsed_ms"), training_ms + two_elapsed),
        "nc_wos_rmse_div_pure_rmse": fnum(comparison.get("nc_wos_rmse_div_pure_wos_rmse"), nc_rmse / pure_rmse if pure_rmse > 0 else float("nan")),
        "nc_2lmc_rmse_div_pure_rmse": fnum(comparison.get("nc_2lmc_rmse_div_pure_wos_rmse"), two_rmse / pure_rmse if pure_rmse > 0 else float("nan")),
        "nc_2lmc_rmse_div_nc_wos_rmse": fnum(comparison.get("nc_2lmc_rmse_div_nc_wos_rmse"), two_rmse / nc_rmse if nc_rmse > 0 else float("nan")),
        "pure_elapsed_div_nc_wos_elapsed": pure_elapsed / nc_elapsed if nc_elapsed > 0 else float("nan"),
        "pure_elapsed_div_nc_2lmc_elapsed": pure_elapsed / two_elapsed if two_elapsed > 0 else float("nan"),
        "pure_elapsed_div_nc_wos_total": pure_elapsed / (training_ms + nc_elapsed) if training_ms + nc_elapsed > 0 else float("nan"),
        "pure_elapsed_div_nc_2lmc_total": pure_elapsed / (training_ms + two_elapsed) if training_ms + two_elapsed > 0 else float("nan"),
        "residual_variance_ratio_vs_pure": fnum(data.get("residual_variance_ratio_vs_pure"), fnum(two.get("mean_residual_sample_variance")) / fnum(pure.get("mean_sample_variance")) if fnum(pure.get("mean_sample_variance")) > 0 else float("nan")),
        "launch_audit": launch_audit,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "json_path",
        "train_points_requested",
        "train_points_padded",
        "eval_points",
        "inside_pixels",
        "depth_m",
        "cache_preset",
        "boundary_condition",
        "label_source",
        "slice_width",
        "slice_height",
        "coarse_walks_per_point",
        "residual_walks_per_point",
        "pure_walks_per_point",
        "hybrid_walks_per_point",
        "label_update_ms",
        "tcnn_training_ms",
        "total_training_ms",
        "pure_rmse",
        "pure_elapsed_ms",
        "nc_wos_rmse",
        "nc_wos_mean_bias",
        "nc_wos_elapsed_ms",
        "nc_wos_total_ms",
        "nc_2lmc_rmse",
        "nc_2lmc_mean_bias",
        "nc_2lmc_mean_coarse_sample_variance",
        "nc_2lmc_mean_residual_sample_variance",
        "residual_variance_ratio_vs_pure",
        "nc_2lmc_elapsed_ms",
        "nc_2lmc_total_ms",
        "nc_wos_rmse_div_pure_rmse",
        "nc_2lmc_rmse_div_pure_rmse",
        "nc_2lmc_rmse_div_nc_wos_rmse",
        "pure_elapsed_div_nc_wos_elapsed",
        "pure_elapsed_div_nc_2lmc_elapsed",
        "pure_elapsed_div_nc_wos_total",
        "pure_elapsed_div_nc_2lmc_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mesh", default="procedural_bumpy_sphere")
    p.add_argument("--mesh-path", default="")
    p.add_argument("--normalize", type=int, default=1)
    p.add_argument("--bumpy-stacks", type=int, default=128)
    p.add_argument("--bumpy-slices", type=int, default=256)
    p.add_argument("--bumpy-amplitude", type=float, default=0.15)
    p.add_argument("--boundary", default="external_charges_medium")
    p.add_argument("--label-source", default="wos_supervision")
    p.add_argument("--cache-preset", default="nano")
    p.add_argument("--train-points-list", default="20000", help="comma-separated train point counts, e.g. 5000,10000,20000")
    p.add_argument("--label-refreshes", type=int, default=4)
    p.add_argument("--walks-per-label-refresh", type=int, default=16)
    p.add_argument("--train-steps-per-refresh", type=int, default=50)
    p.add_argument("--depth-m", type=int, default=4)
    p.add_argument("--allocations", default="32:16,32:32,64:16,64:32", help="comma-separated coarse:residual wpp pairs")
    p.add_argument("--slice-width", type=int, default=512)
    p.add_argument("--slice-height", type=int, default=512)
    p.add_argument("--slice-view", default="xy", choices=["xy", "xz", "yz"])
    p.add_argument("--slice-plane", default="0.0")
    p.add_argument("--slice-frame", default="")
    p.add_argument("--slice-padding-fraction", type=float, default=0.02)
    p.add_argument("--slice-preserve-world-aspect", type=int, default=1)
    p.add_argument("--pure-walks-per-point", type=int, default=64)
    p.add_argument("--hybrid-walks-per-point", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--epsilon", default="1e-4")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--seed-stride", type=int, default=0)
    p.add_argument("--cubql-build-method", default="sah")
    p.add_argument("--cubql-leaf-size", type=int, default=8)
    p.add_argument("--jit", type=int, default=0)
    p.add_argument("--assumed-block-size", type=int, default=128, help="used only for launch-shape audit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allocations = parse_allocations(args.allocations)
    train_points_list = parse_int_list(args.train_points_list)

    commands: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for train_points in train_points_list:
        for coarse_wpp, residual_wpp in allocations:
            label = f"train{train_points}_m{args.depth_m}_c{coarse_wpp}_r{residual_wpp}"
            out_json = output_dir / f"{label}.json"
            slice_prefix = output_dir / f"{label}_slice"
            seed = args.seed + args.seed_stride * len(commands)
            cmd = [
                args.executable,
                "--mesh", args.mesh,
                "--boundary", args.boundary,
                "--label-source", args.label_source,
                "--cache-preset", args.cache_preset,
                "--train-points", str(train_points),
                "--eval-mode", "slice",
                "--slice-width", str(args.slice_width),
                "--slice-height", str(args.slice_height),
                "--slice-view", args.slice_view,
                "--slice-plane", str(args.slice_plane),
                "--slice-preserve-world-aspect", str(args.slice_preserve_world_aspect),
                "--slice-padding-fraction", str(args.slice_padding_fraction),
                "--slice-output-prefix", str(slice_prefix),
                "--label-refreshes", str(args.label_refreshes),
                "--walks-per-label-refresh", str(args.walks_per_label_refresh),
                "--train-steps-per-refresh", str(args.train_steps_per_refresh),
                "--pure-walks-per-point", str(args.pure_walks_per_point),
                "--hybrid-walks-per-point", str(args.hybrid_walks_per_point),
                "--enable-2lmc", "1",
                "--coarse-walks-per-point", str(coarse_wpp),
                "--residual-walks-per-point", str(residual_wpp),
                "--depth-m", str(args.depth_m),
                "--max-steps", str(args.max_steps),
                "--epsilon", str(args.epsilon),
                "--seed", str(seed),
                "--cubql-build-method", args.cubql_build_method,
                "--cubql-leaf-size", str(args.cubql_leaf_size),
                "--output", str(out_json),
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
            if args.slice_frame:
                cmd.extend(["--slice-frame", args.slice_frame])

            record = {
                "label": label,
                "train_points": train_points,
                "coarse_walks_per_point": coarse_wpp,
                "residual_walks_per_point": residual_wpp,
                "seed": seed,
                "output": str(out_json),
                "command": cmd,
            }
            commands.append(record)

            if args.skip_existing and out_json.exists():
                print(f"skip existing {out_json}", flush=True)
            else:
                try:
                    run_command(cmd, dry_run=args.dry_run)
                except subprocess.CalledProcessError as exc:
                    errors.append({"label": label, "returncode": exc.returncode, "command": cmd})
                    if not args.continue_on_error:
                        raise
                    continue

            if not args.dry_run and out_json.exists():
                row = summarize_result(out_json, coarse_wpp, residual_wpp, train_points, args.assumed_block_size)
                row["label"] = label
                rows.append(row)

    manifest: Dict[str, Any] = {
        "script": "run_nc_slice_allocation_sweep.py",
        "purpose": "sweep NC+2LMC coarse/residual allocation on a fixed slice and audit launch shapes",
        "arguments": vars(args),
        "commands": commands,
        "errors": errors,
        "limitations": {
            "launch_audit_is_static": True,
            "launch_audit_uses_assumed_block_size": args.assumed_block_size,
            "actual_occupancy_requires_nsight_compute": True,
            "actual_executable_block_size_may_be_hardcoded": True,
        },
        "rows": rows,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if rows:
        write_csv(output_dir / "summary.csv", rows)
        with (output_dir / "launch_audit.json").open("w", encoding="utf-8") as f:
            json.dump({"rows": [{"label": r["label"], "launch_audit": r["launch_audit"]} for r in rows]}, f, indent=2)

    print(f"wrote {output_dir / 'summary.json'}")
    if rows:
        print(f"wrote {output_dir / 'summary.csv'}")
        print(f"wrote {output_dir / 'launch_audit.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

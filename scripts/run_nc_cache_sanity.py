#!/usr/bin/env python3
"""Sanity checks for the TCNN Neural Cache training/inference path.

This runner is deliberately independent of 2LMC.  It checks whether the cache
can overfit simple deterministic targets and whether larger iNGP presets change
slice/point RMSE.  Use it before interpreting NC/2LMC timing results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_command(cmd: Sequence[str]) -> None:
    print("[run_nc_cache_sanity] running:")
    print(" ".join(cmd))
    subprocess.run(list(cmd), check=True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stats_from_estimates(path: Path, target_col: str = "analytic_value", pred_col: str = "nc_wos_mean") -> Dict[str, float]:
    rows = read_rows(path)
    if not rows:
        return {"points": 0, "rmse": 0.0, "mae": 0.0, "max_abs_error": 0.0, "correlation": 0.0, "target_variance": 0.0, "pred_variance": 0.0}
    pred = [float(r[pred_col]) for r in rows]
    target = [float(r[target_col]) for r in rows]
    n = len(pred)
    err = [p - t for p, t in zip(pred, target)]
    mp = sum(pred) / n
    mt = sum(target) / n
    me = sum(err) / n
    vp = sum((p - mp) ** 2 for p in pred) / (n - 1) if n > 1 else 0.0
    vt = sum((t - mt) ** 2 for t in target) / (n - 1) if n > 1 else 0.0
    ve = sum((e - me) ** 2 for e in err) / (n - 1) if n > 1 else 0.0
    cov = sum((p - mp) * (t - mt) for p, t in zip(pred, target))
    denom = math.sqrt(sum((p - mp) ** 2 for p in pred) * sum((t - mt) ** 2 for t in target))
    corr = cov / denom if denom > 0.0 else 0.0
    rmse = math.sqrt(sum(e * e for e in err) / n)
    mae = sum(abs(e) for e in err) / n
    return {
        "points": n,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max(abs(e) for e in err),
        "mean_error": me,
        "correlation": corr,
        "target_mean": mt,
        "pred_mean": mp,
        "target_variance": vt,
        "pred_variance": vp,
        "error_variance": ve,
        "normalized_rmse_by_target_std": rmse / math.sqrt(vt) if vt > 0.0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run direct-cache sanity checks for TCNN NC training")
    p.add_argument("--executable", default="./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mesh", default="procedural_bumpy_sphere")
    p.add_argument("--mesh-path", default="")
    p.add_argument("--normalize", type=int, default=1)
    p.add_argument("--bumpy-stacks", type=int, default=128)
    p.add_argument("--bumpy-slices", type=int, default=256)
    p.add_argument("--bumpy-amplitude", type=float, default=0.15)
    p.add_argument("--boundaries", default="constant_one,harmonic_x2_minus_y2,boundary_texture_stripes_k16")
    p.add_argument("--cache-presets", default="nano,light,baseline,heavy")
    p.add_argument("--train-points", type=int, default=2048)
    p.add_argument("--train-steps-per-refresh-list", default="0,10,50,250,1000")
    p.add_argument("--label-refreshes", type=int, default=1)
    p.add_argument("--walks-per-label-refresh", type=int, default=1)
    p.add_argument("--learning-rate", default="")
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--epsilon", default="1e-4")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--seed-stride", type=int, default=1009)
    p.add_argument("--cubql-build-method", default="sah")
    p.add_argument("--cubql-leaf-size", type=int, default=8)
    p.add_argument("--jit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "runs"
    est_dir = out_dir / "estimates"
    runs_dir.mkdir(parents=True, exist_ok=True)
    est_dir.mkdir(parents=True, exist_ok=True)

    boundaries = parse_csv_list(args.boundaries)
    presets = parse_csv_list(args.cache_presets)
    steps_list = parse_int_list(args.train_steps_per_refresh_list)
    rows: List[Dict[str, object]] = []
    commands: List[Dict[str, object]] = []

    run_index = 0
    for boundary in boundaries:
        for preset in presets:
            for steps in steps_list:
                label = f"{boundary}_{preset}_steps{steps}".replace("/", "_")
                output_json = runs_dir / f"{label}.json"
                prefix = est_dir / label
                estimates_csv = Path(str(prefix) + "_estimates.csv")
                seed = args.seed + args.seed_stride * run_index
                cmd: List[str] = [
                    args.executable,
                    "--mesh", args.mesh,
                    "--boundary", boundary,
                    "--label-source", "exact_analytic",
                    "--cache-preset", preset,
                    "--train-points", str(args.train_points),
                    "--eval-use-train-points", "1",
                    "--label-refreshes", str(args.label_refreshes),
                    "--walks-per-label-refresh", str(args.walks_per_label_refresh),
                    "--train-steps-per-refresh", str(steps),
                    "--depth-m", "0",
                    "--pure-walks-per-point", "1",
                    "--hybrid-walks-per-point", "1",
                    "--enable-2lmc", "0",
                    "--coarse-walks-per-point", "1",
                    "--residual-walks-per-point", "1",
                    "--max-steps", str(args.max_steps),
                    "--epsilon", args.epsilon,
                    "--seed", str(seed),
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
                    cmd.extend(["--bumpy-stacks", str(args.bumpy_stacks), "--bumpy-slices", str(args.bumpy_slices), "--bumpy-amplitude", str(args.bumpy_amplitude)])
                if args.learning_rate:
                    cmd.extend(["--learning-rate", args.learning_rate])
                commands.append({"label": label, "command": cmd, "output": str(output_json), "estimates_csv": str(estimates_csv)})
                if not args.dry_run and not (args.skip_existing and output_json.exists() and estimates_csv.exists()):
                    run_command(cmd)
                if not args.dry_run:
                    metrics = stats_from_estimates(estimates_csv)
                    run_json = json.loads(output_json.read_text(encoding="utf-8"))
                    row: Dict[str, object] = {
                        "label": label,
                        "boundary": boundary,
                        "cache_preset": preset,
                        "train_steps_per_refresh": steps,
                        "train_points": args.train_points,
                        "json_path": str(output_json),
                        "estimates_csv": str(estimates_csv),
                        **metrics,
                        "label_update_ms": run_json.get("training", {}).get("label_update_ms", 0.0),
                        "tcnn_training_ms": run_json.get("training", {}).get("tcnn_training_ms", 0.0),
                        "total_training_ms": run_json.get("training", {}).get("total_training_ms", 0.0),
                        "nc_wos_elapsed_ms": run_json.get("runs", {}).get("nc_wos", {}).get("elapsed_ms", 0.0),
                    }
                    rows.append(row)
                run_index += 1

    manifest = {"script": "run_nc_cache_sanity.py", "arguments": vars(args), "commands": commands, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        keys = list(rows[0].keys())
        with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    else:
        (out_dir / "summary.csv").write_text("\n", encoding="utf-8")
    print(f"[run_nc_cache_sanity] wrote {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a compact NC+2LMC sweep over boundary signal, cache size, m, and seed."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BOUNDARIES = [
    "external_charges_high",
    "external_charges_shell_k8",
    "external_charges_shell_k16",
    "harmonic_zebra_k8",
    "harmonic_zebra_k12",
]

REFERENCE_ONLY_BOUNDARIES = [
    "boundary_texture_checker_k8",
    "boundary_texture_checker_k16",
    "boundary_texture_stripes_k8",
    "boundary_texture_stripes_k16",
]


def comma_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def comma_ints(text: str) -> list[int]:
    return [int(x) for x in comma_list(text)]


def is_reference_only_boundary(name: str) -> bool:
    b = name.lower()
    return any(tok in b for tok in ("texture", "checker", "stripes"))


def run(cmd: list[str], dry_run: bool) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--executable", default="./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos")
    ap.add_argument("--output-dir", default="results/nc_2lmc_interior_signal_sweep")
    ap.add_argument("--mesh", default="procedural_bumpy_sphere")
    ap.add_argument("--mesh-path", default="")
    ap.add_argument("--cubql-build-method", default="sah")
    ap.add_argument("--boundaries", default=",".join(DEFAULT_BOUNDARIES))
    ap.add_argument("--include-reference-only-boundaries", action="store_true",
                    help="Also run texture/checker boundary modes. Use a numerical Pure WoS reference when interpreting RMSE.")
    ap.add_argument("--cache-presets", default="nano,light")
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--seeds", default="12345,12346,12347")
    ap.add_argument("--train-points", type=int, default=20000)
    ap.add_argument("--eval-points", type=int, default=8192)
    ap.add_argument("--label-refreshes", type=int, default=4)
    ap.add_argument("--walks-per-label-refresh", type=int, default=50)
    ap.add_argument("--train-steps-per-refresh", type=int, default=500)
    ap.add_argument("--pure-walks-per-point", type=int, default=64)
    ap.add_argument("--hybrid-walks-per-point", type=int, default=4)
    ap.add_argument("--coarse-walks-per-point", type=int, default=64)
    ap.add_argument("--residual-walks-per-point", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--epsilon", default="1e-4")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--eval-mode", default="ball", choices=["ball", "slice"])
    ap.add_argument("--slice-width", type=int, default=512)
    ap.add_argument("--slice-height", type=int, default=512)
    ap.add_argument("--slice-view", default="xy")
    ap.add_argument("--slice-plane", default="0.0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument("--audit", action="store_true", help="Run scripts/audit_nc_2lmc_efficiency.py after the sweep")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries = comma_list(args.boundaries)
    if args.include_reference_only_boundaries:
        boundaries += REFERENCE_ONLY_BOUNDARIES
    cache_presets = comma_list(args.cache_presets)
    depths = comma_ints(args.depths)
    seeds = comma_ints(args.seeds)

    manifest = {
        "executable": args.executable,
        "mesh": args.mesh,
        "mesh_path": args.mesh_path,
        "boundaries": boundaries,
        "cache_presets": cache_presets,
        "depths": depths,
        "seeds": seeds,
        "warning": "Texture/checker boundaries require numerical reference-based error metrics.",
        "runs": [],
    }

    failures = 0
    for boundary, preset, depth_m, seed in itertools.product(boundaries, cache_presets, depths, seeds):
        stem = f"{args.mesh}_{boundary}_{preset}_m{depth_m}_seed{seed}"
        stem = stem.replace("/", "_").replace(" ", "_")
        result_path = out_dir / f"{stem}.json"

        cmd = [
            args.executable,
            "--mesh", args.mesh,
            "--boundary", boundary,
            "--label-source", "wos_supervision",
            "--train-sampler", "rejection",
            "--cache-preset", preset,
            "--train-points", str(args.train_points),
            "--eval-points", str(args.eval_points),
            "--eval-mode", args.eval_mode,
            "--label-refreshes", str(args.label_refreshes),
            "--walks-per-label-refresh", str(args.walks_per_label_refresh),
            "--train-steps-per-refresh", str(args.train_steps_per_refresh),
            "--pure-walks-per-point", str(args.pure_walks_per_point),
            "--hybrid-walks-per-point", str(args.hybrid_walks_per_point),
            "--coarse-walks-per-point", str(args.coarse_walks_per_point),
            "--residual-walks-per-point", str(args.residual_walks_per_point),
            "--enable-2lmc", "1",
            "--depth-m", str(depth_m),
            "--max-steps", str(args.max_steps),
            "--epsilon", str(args.epsilon),
            "--block-size", str(args.block_size),
            "--seed", str(seed),
            "--cubql-build-method", args.cubql_build_method,
            "--output", str(result_path),
        ]
        if args.mesh_path:
            cmd += ["--mesh-path", args.mesh_path]
        if args.eval_mode == "slice":
            cmd += [
                "--slice-width", str(args.slice_width),
                "--slice-height", str(args.slice_height),
                "--slice-view", args.slice_view,
                "--slice-plane", str(args.slice_plane),
                "--slice-output-prefix", str(out_dir / stem),
            ]

        code = run(cmd, args.dry_run)
        manifest["runs"].append({
            "result": str(result_path),
            "boundary": boundary,
            "cache_preset": preset,
            "depth_m": depth_m,
            "seed": seed,
            "reference_only_boundary": is_reference_only_boundary(boundary),
            "exit_code": code,
        })
        if code != 0:
            failures += 1
            if args.stop_on_failure:
                break

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if args.audit and not args.dry_run:
        audit_csv = out_dir / "audit.csv"
        audit_script = Path(__file__).with_name("audit_nc_2lmc_efficiency.py")
        code = run([sys.executable, str(audit_script), "--inputs", str(out_dir / "*.json"), "--out-csv", str(audit_csv)], False)
        if code != 0:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

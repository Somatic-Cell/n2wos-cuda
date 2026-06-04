# Neural Cache / 2LMC result collection

Use `scripts/collect_nc_results.py` to gather many `n2wos_eval_tcnn_nc_wos`
JSON files into one CSV/JSON table.  The collector is intended for quick paper
iteration: it extracts solver metrics, training settings, slice metadata, timing,
and warning flags from all matching JSON files.

## Basic usage

```bash
python scripts/collect_nc_results.py \
  --inputs "results/**/*.json" \
  --out-csv results/nc_results_summary.csv \
  --out-json results/nc_results_summary.json
```

Quote the glob pattern in bash/WSL so Python receives it unchanged.

To avoid collecting top-level manifests or old config snapshots, the script skips
JSON files that do not contain a `runs` object by default.  It reports the number
of skipped files on stderr and records them in the JSON output.

## Useful focused collection

For Bunny slice experiments at the current paper convention `z=0.12`:

```bash
python scripts/collect_nc_results.py \
  --inputs "results/bunny_*_z012*.json" \
  --out-csv results/bunny_z012_summary.csv \
  --out-json results/bunny_z012_summary.json
```

For a single diagnostic folder:

```bash
python scripts/collect_nc_results.py \
  --inputs "results/bunny_exact_zebra_k8_z012_train*.json" \
  --out-csv results/bunny_exact_zebra_k8_z012_train_sweep.csv
```

## Important columns

The most useful columns for current debugging are:

```text
mesh, mesh_path, boundary, label_source, train_sampler, cache_preset, seed, depth_m
slice_plane, train_points_requested, total_optimizer_steps, total_label_walks_per_point
pure_rmse, nc_wos_rmse, nc_2lmc_rmse
pure_sample_variance, residual_sample_variance
residual_variance_ratio_vs_pure
nc_2lmc_estimator_variance_ratio_vs_pure
pure_elapsed_ms, nc_wos_elapsed_ms, nc_2lmc_elapsed_ms
pure_mse_time, nc_wos_mse_time, nc_2lmc_mse_time
warning_count, warnings
```

For 2LMC efficiency, prefer estimator-level variance:

```text
Var[C] / N_coarse + Var[W - C] / N_residual
```

This is reported as `nc_2lmc_estimator_variance`.  The raw residual variance
ratio is useful for diagnosing the control variate, but not enough by itself for
MSE/time comparison.

## Warnings

The collector emits warnings for common interpretation mistakes, including:

```text
bunny slice plane is not 0.12
procedural_bumpy_sphere result while Bunny is the current default convention
depth_m=0, which is a bias-correction sanity check, not a variance-reduction test
large optimizer-step count per noisy WoS label refresh
missing estimates_csv for slice runs
```

Warnings are meant to prevent accidental mixing of pilot data and paper-facing
experiments.

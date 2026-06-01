# NC slice allocation sweep and launch-shape audit

This experiment addresses two questions that should be kept separate.

1. **Estimator allocation**: for a fixed cache snapshot, prefix depth, and slice
   point set, how should NC+2LMC split work between the cheap coarse/cache term
   and the expensive residual correction term?
2. **Launch shape sanity**: are the current sample counts obviously hostile to
   CUDA warp execution?

The new script is an orchestration tool around the existing
`n2wos_eval_tcnn_nc_wos` executable. It does not change solver kernels.

## Recommended first sweep

```bash
python3 scripts/run_nc_slice_allocation_sweep.py \
  --executable ./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos \
  --output-dir results/nc_slice_alloc_bumpy_nano_medium_m4 \
  --mesh procedural_bumpy_sphere \
  --boundary external_charges_medium \
  --label-source wos_supervision \
  --cache-preset nano \
  --train-points-list 20000 \
  --label-refreshes 4 \
  --walks-per-label-refresh 16 \
  --train-steps-per-refresh 50 \
  --depth-m 4 \
  --slice-width 512 \
  --slice-height 512 \
  --slice-view xy \
  --slice-plane 0 \
  --pure-walks-per-point 64 \
  --hybrid-walks-per-point 4 \
  --allocations 32:16,32:32,64:16,64:32 \
  --cubql-build-method sah
```

This intentionally reuses the current best diagnostic regime:

- 512 x 512 slice evaluation.
- `external_charges_medium` to avoid making the field too low-frequency while
  not being as severe as the high stress case.
- `cache_preset=nano`, because prior sweeps showed that NC+2LMC does not need a
  large cache to obtain similar residual-variance behavior.
- `m=4`, because earlier slice results showed much stronger residual variance
  reduction than `m=1`.

## Training point count

For NC-paper-like baselines, keep 20,000 training points. For the proposed
low-capacity/unfinished-cache story, it is reasonable to test fewer points:

```bash
--train-points-list 5000,10000,20000
```

Reducing training points should reduce WoS supervision cost approximately
linearly, but it may increase NC-only bias and NC+2LMC residual variance. This
is an ablation, not a replacement for the paper-like 20k baseline.

## Launch-shape audit

The script writes `launch_audit.json`. It computes, for each run:

- sample count,
- assumed block size,
- grid block count,
- launched thread count,
- inactive tail threads,
- inactive tail fraction,
- warnings such as non-warp-multiple block size or small grids.

This is a static check only. It does not measure register pressure, SM
occupancy, memory bandwidth, or divergence. For actual occupancy and warp-level
stall diagnosis, use Nsight Compute on the executable.

Current large-slice runs are expected to have millions of samples, so the final
partial block is negligible. The main warp-safety criterion is that the block
size remains a multiple of 32. The existing code path has historically used 128
threads per block in CUDA geometry probes, which is warp-aligned. If the solver
internally changes block size, the JSON/static audit should be updated to report
that value explicitly.

## Interpretation

A configuration is promising if it improves the variance-time tradeoff rather
than only lowering inference time. Useful columns in `summary.csv` are:

- `nc_2lmc_rmse_div_pure_rmse`,
- `pure_elapsed_div_nc_2lmc_elapsed`,
- `pure_elapsed_div_nc_2lmc_total`,
- `residual_variance_ratio_vs_pure`,
- `nc_2lmc_mean_residual_sample_variance`,
- `label_update_ms`,
- `tcnn_training_ms`.

If lowering the residual walks-per-point makes NC+2LMC faster but the RMSE grows
above pure WoS by too much, the estimator is still not competitive in MSE/time.
If increasing residual walks-per-point reduces RMSE but makes runtime approach
pure WoS, the cache is not yet acting as a strong enough control variate.

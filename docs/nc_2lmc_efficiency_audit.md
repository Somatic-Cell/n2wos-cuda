# NC+2LMC efficiency audit policy

This note defines the minimum checks before using `n2wos_eval_tcnn_nc_wos`
results in a paper figure or table.

The key distinction is:

```text
NC-only:   estimate C_theta(X_m)
NC+2LMC:   estimate E[C_theta(X_m)] + E[W(X_m) - C_theta(X_m)]
```

The second estimator can remain unbiased even when the cache is inaccurate.
That does **not** imply it is more efficient than Pure WoS.  Efficiency requires
enough residual variance reduction to pay for prefix walks, cache inference,
continuations, and the coarse/residual allocation.

## Required result fields

For each run, record or derive the following quantities:

```text
pure_var_estimator   = Var[pure sample] / N_pure
2lmc_var_estimator   = Var[coarse sample] / N_coarse
                     + Var[residual sample] / N_residual
residual_var_ratio   = Var[residual sample] / Var[pure sample]
pure_mse_time        = RMSE_pure^2 * time_pure
2lmc_mse_time        = RMSE_2lmc^2 * time_2lmc_inference
```

Sample variance alone is not enough.  A residual variance ratio of `0.55` is
promising only if the residual continuation cost and TCNN overhead are low
enough, and if the residual allocation is not too small.

## Truth handling

Boundary modes fall into two classes.

### Analytic interior-solution modes

These can be compared directly to an interior value function, assuming their
implementation evaluates the same harmonic field at interior points:

```text
constant_one
harmonic_zebra_k4 / k8 / k12
external_charges_high
external_charges_medium
external_charges_shell_k8 / k16
```

### Boundary-only texture modes

These should be treated as boundary data only:

```text
boundary_texture_checker_k8 / k16
boundary_texture_stripes_k8 / k16
```

For these modes, `nc_boundary_value_host(x)` is not automatically the interior
solution.  Therefore `rmse`, `mae`, and `mean_bias` computed against that value
should be read as a proxy unless a high-sample Pure WoS reference is supplied.

## Interpreting a failed MSE/time comparison

If NC+2LMC is unbiased but loses to Pure WoS, check in this order:

1. **Training distribution**: training points should be sampled in the domain,
   not from an ad-hoc local ball, unless this is an explicit ablation.
2. **Residual variance**: if `residual_var_ratio > 0.7`, the cache is not yet a
   strong control variate at the chosen depth.
3. **Depth**: if `m=1`, the prefix may be too shallow for `C_theta(X_m)` and the
   true continuation `W(X_m)` to correlate strongly.
4. **Allocation**: if residual variance is small, reduce residual samples; if it
   is large, increasing coarse samples will not fix MSE.
5. **Boundary signal**: if the interior solution is nearly constant over the
   evaluation region, there is little room for a cache to help.
6. **Timing path**: native TCNN batch inference, prefix kernels, combine kernels,
   and readbacks can dominate small batch or low-WPP tests.

## Recommended screening matrix

Start with analytic interior-solution boundary conditions:

```text
boundary:     external_charges_high, external_charges_shell_k8,
              external_charges_shell_k16, harmonic_zebra_k8, harmonic_zebra_k12
cache:        nano, light
depth_m:      1, 2, 4, 8
seeds:        3 or more
```

Use texture/checker boundary modes only after switching to a numerical reference
workflow.

## Fast rejection criteria

Do not promote a run to a main figure if any of these are true:

```text
train_sampler != rejection
label_source  != wos_supervision
residual_var_ratio > 0.8 at m >= 4
analytic_truth_available == false but RMSE is labeled analytic
2lmc_mse_time / pure_mse_time > 1.0 without explaining the purpose as bias safety
```

## Paper-safe claims

The following claims are safe if supported by the audit table:

```text
Safe:
  NC+2LMC removes the bias floor of an unfinished neural cache.
  Low-capacity caches can still be useful as control variates.
  Increasing m trades extra prefix cost for lower residual variance.

Not safe unless directly measured:
  NC+2LMC is faster than Pure WoS for a fixed MSE.
  Texture/checker boundary RMSE is analytic RMSE.
  More optimizer steps will automatically improve residual variance.
```

## Usage

Run a screening sweep:

```bash
python3 scripts/run_nc_interior_signal_sweep.py \
  --executable ./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos \
  --output-dir results/nc_2lmc_interior_signal_sweep
```

Audit existing JSON results:

```bash
python3 scripts/audit_nc_2lmc_efficiency.py \
  --inputs "results/**/*.json" \
  --out-csv results/nc_2lmc_audit.csv
```

Inspect rows with `warning_count > 0` before making figures.

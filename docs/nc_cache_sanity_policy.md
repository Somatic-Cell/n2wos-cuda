# Neural Cache sanity checks

Before interpreting Neural Cache + WoS or NC+2LMC results, verify that the cache
backend can learn simple deterministic targets.

The main diagnostic is `scripts/run_nc_cache_sanity.py`. It runs
`n2wos_eval_tcnn_nc_wos` with:

- `--label-source exact_analytic`, so no WoS label noise is present;
- `--eval-use-train-points 1`, so evaluation reuses the training points;
- `--depth-m 0`, so the reported `nc_wos_mean` is direct `C_theta(x)` rather than
  a stochastic prefix estimate.

Use this to distinguish implementation bugs from capacity/training issues.
A constant-one field and a low-frequency harmonic field should be easy. If these
fail to overfit, do not interpret 2LMC results.

Cache presets added for diagnosis:

- `heavy`: larger HashGrid + MLP than `baseline`.
- `xlarge`: larger feature count and MLP depth; intended only for capacity checks.

This diagnostic is not a production benchmark. It is a correctness and capacity
screening step for the TCNN training/inference path.

# Neural Cache + WoS screening policy

Patch 0005 adds a biased Neural Cache early-termination diagnostic. It is not the
unbiased two-level residual-correction estimator.

The training schedule is fixed rather than early-stopped by validation MSE:

1. sample a fixed set of screening points;
2. refresh labels with either WoS supervision or exact analytic labels;
3. maintain labels as running averages across refreshes;
4. run a fixed number of tiny-cuda-nn optimizer steps after each refresh;
5. evaluate pure WoS and biased NC+WoS on the same evaluation points.

The main path keeps prefix outputs and TCNN inputs/outputs on CUDA device memory.
No CSV export, Python binding, or host transfer is used between prefix WoS and
cache inference.

`external_charges_high` is the first higher-frequency boundary signal. The charge
sources are outside the normalized diagnostic domain, so the exact harmonic value
is known and can be used for pointwise RMSE screening.

This tool uses an inscribed-ball point sampler to keep 0005 small and robust for
the bumpy-sphere screening case. Later dense-field patches should replace this
with slice/interior-mask sampling and time-to-MSE curves.

## Patch 0006: fixed-m NC+2LMC diagnostic

Patch 0006 extends the same executable with an m=1 NC+2LMC diagnostic. The
comparison deliberately keeps NC-only and NC+2LMC on the same trained cache,
input points, boundary condition, geometry backend, and prefix depth.

For each evaluation point, the tool now reports three estimates:

```text
pure_wos:
  full WoS samples to the boundary

nc_wos:
  x -> one WoS prefix step -> C_theta(X_1)
  boundary hits before m return the boundary value directly

nc_2lmc_m1:
  coarse:   mean C_theta(X_1) with many cheap prefix/cache samples
  residual: mean [W(X_1) - C_theta(X_1)] with fewer continuation samples
  estimate: mean_coarse + mean_residual
```

The m=1 constraint is intentional for the first comparison. It prevents NC-only
from using a shallower prefix than 2LMC, and it prevents 2LMC from winning merely
because it used a different early-termination depth. Later sweeps can add m=2,4,8
once the m=1 estimator has been checked.

The TCNN output is consumed by CUDA kernels to form NC-only samples and residual
samples. The path still avoids CSV export, Python bindings, and host transfer
between TCNN inference and the residual-combination kernel. Final readback is
for statistics and JSON output only.

Two cache presets are meant for the first experiments:

```text
baseline:
  n_levels=12, log2_hashmap_size=18, n_neurons=32, n_hidden_layers=2

light:
  n_levels=8, log2_hashmap_size=15, n_neurons=16, n_hidden_layers=1
```

The intended research diagnostic is not simply whether NC-only is fastest. It is
whether a lightweight cache creates a visible NC-only bias floor, while NC+2LMC
with the same lightweight cache removes or reduces that bias at a runtime between
NC-only and pure WoS.

# NC / NC+2LMC prefix-depth sweep

This diagnostic is the follow-up to the unfinished-cache m=1 check. It allows
`depth_m > 1` while keeping the comparison fair:

```text
NC-WoS(m):     x -> m WoS prefix steps -> C_theta(X_m)
NC+2LMC(m):    E[C_theta(X_m)] + E[W(X_m) - C_theta(X_m)]
```

The main rule is that NC-WoS and NC+2LMC must share the same m in each row.
Increasing m should not be used only for 2LMC.

What to look for:

```text
NC-only mean bias:
  should generally decrease as m increases, but speed also worsens.

NC+2LMC mean bias:
  should remain small if the snapshot discipline and residual correction are
  implemented correctly.

Residual variance ratio vs pure WoS:
  should decrease if C_theta(X_m) becomes a better control variate for the
  continuation from X_m.

MSE/time:
  may or may not beat pure WoS. The first target is a clean bias-correction and
  residual-variance tradeoff story, not a universal speedup claim.
```

Recommended first sweep:

```bash
python3 scripts/run_nc_depth_sweep.py \
  --executable ./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos \
  --output-dir results/nc_depth_sweep_light_high \
  --mesh procedural_bumpy_sphere \
  --boundary external_charges_high \
  --label-source wos_supervision \
  --cache-preset light \
  --train-points 20000 \
  --eval-points 8192 \
  --label-refreshes 4 \
  --walks-per-label-refresh 16 \
  --train-steps-per-refresh 50 \
  --depths 1,2,4 \
  --pure-walks-per-point 64 \
  --hybrid-walks-per-point 4 \
  --coarse-walks-per-point 64 \
  --residual-walks-per-point 32 \
  --cubql-build-method sah
```

If `external_charges_high` remains too hard and the residual variance ratio does
not move much with m, repeat the same sweep with `external_charges_medium` before
changing network architecture or geometry backend.

# Manufactured Poisson WoS modes

This patch adds an experimental constant-coefficient Poisson diagnostic for the NC/WoS evaluator.  The implemented equation is

```text
-Delta u = f,  u|boundary = g
```

There is no screening/absorption term.  Therefore the precise name is **unscreened Poisson** or simply **Poisson**, not "screened Poisson without absorption".

The new boundary/problem modes are:

```text
poisson_multiscale
poisson_figlike_hf
```

The same analytic manufactured function is used for interior truth and boundary data.  The source term is analytic.  For the current experimental patch, the source terms are chosen to be harmonic, so the local Green contribution for a WoS ball in 3D is exact with a center evaluation:

```text
source_contribution = radius^2 / 6 * f(center)
```

This is not the most general Poisson WoS estimator.  It is a deliberately controlled first step that avoids the extra random volume-source sample while still injecting an interior source term.

After applying the git patch, run:

```bash
python scripts/apply_manufactured_poisson_patch.py --check
python scripts/apply_manufactured_poisson_patch.py --apply
cmake --build ./build/cuda-release-cubql-tcnn --target n2wos_eval_tcnn_nc_wos -j
```

Then test with `poisson_multiscale` first.  Use `poisson_figlike_hf` only if the multiscale field is still too low-frequency.

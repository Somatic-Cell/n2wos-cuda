# Harmonic mixture boundary modes

This patch adds two manufactured analytic harmonic fields for Neural Cache / 2LMC diagnostics on Bunny slices:

```text
harmonic_mixture_smooth
harmonic_mixture_figlike
```

They are not texture-only boundary modes.  The same function is used as Dirichlet boundary data and as interior analytic truth, so RMSE and signed error images remain meaningful.

Each nonlinear term has the form

```text
exp(k a·x) cos(k b·x + phi)
```

or the corresponding sine variant, with `a` and `b` orthonormal.  Therefore the Laplacian cancels between the exponential and oscillatory directions.  Linear terms are also harmonic.

After applying the git patch, run the source patcher:

```bash
python scripts/apply_harmonic_mixture_boundary_patch.py --check
python scripts/apply_harmonic_mixture_boundary_patch.py --apply
cmake --build ./build/cuda-release-cubql-tcnn -j
```

The patcher is intentionally text-based because the NC evaluator has been evolving through local patches.  It adds enum values, CLI aliases, name parsing, and boundary-value switch cases where it finds the existing NC boundary implementation.

# Multiscale harmonic mixture boundary modes

This patch adds two more analytic harmonic boundary modes:

```text
harmonic_mixture_multiscale
harmonic_mixture_figlike_hf
```

These are manufactured harmonic functions, not texture-only boundaries.  Each
nonlinear term has the form

```text
exp(k a·x) cos(k b·x + phi)
```

or the sine variant, with `a·b = 0`.  Each term is harmonic because the second
derivative in the exponential direction cancels the second derivative in the
oscillatory direction.  The functions are evaluated in solver-normalized
coordinates and can therefore be used both as Dirichlet boundary data and as
analytic interior truth.

The modes are deliberately only medium-high frequency.  Very high-frequency
boundary data for a Laplace-type problem is strongly attenuated in the interior.
These modes aim to keep visible interior structure without immediately making
the neural cache impossible to train.

Apply after 0007/0007a:

```bash
python scripts/apply_harmonic_mixture_multiscale_patch.py --check
python scripts/apply_harmonic_mixture_multiscale_patch.py --apply
cmake --build ./build/cuda-release-cubql-tcnn --target n2wos_eval_tcnn_nc_wos -j
```

Smoke test:

```bash
bash scripts/run_bunny_harmonic_mixture_multiscale_smoke.sh
```

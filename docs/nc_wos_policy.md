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

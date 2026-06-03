# Reference spatial-frequency scan policy

This diagnostic exists to avoid running NC/2LMC on boundary/slice configurations whose interior solution is too smooth to stress Neural Cache, or too oscillatory for the cache to become correlated with the WoS continuation.

The scan evaluates only high-sample Pure WoS references. It does **not** train a Neural Cache and does **not** run NC+2LMC. The output is a ranked list of boundary/slice pairs and reference images.

## What it varies

This script-only fallback uses boundary modes that are already available in the current code after the previous boundary-texture patches:

- `boundary_texture_stripes_k8`
- `boundary_texture_stripes_k16`
- `boundary_texture_checker_k8`
- `boundary_texture_checker_k16`

It also varies slice view and plane, for example `xy` with `z = 0, 0.25, 0.5, 0.7`.

The previous 0020 patch attempted to add `k4` and `k32` modes in C++ as well. That failed on some repository states because the relevant source files had diverged. This fallback deliberately avoids touching C++ source files and therefore does not introduce new boundary modes. Add finer/coarser texture modes only after the current source layout is inspected directly.

## Metrics

For each case the script writes:

- `field_rms`
- `field_std`
- `gradient_rms`
- `total_variation_proxy`
- `high_frequency_energy_ratio`
- `slice_distance_pixels_mean`
- `slice_distance_pixels_q10/q50/q90` when a 2-D mask distance transform is available
- `inside_pixels`

The most useful first filter is usually:

1. reference image visually has interior structure, not just boundary artifacts;
2. `high_frequency_energy_ratio` is not near zero;
3. many pixels are not extremely close to the slice boundary unless the intended problem is a near-boundary stress case;
4. the field amplitude is not nearly zero.

## Reference budget

The default `reference_wpp=4096` is for screening many cases. This is deliberately not the final benchmark budget. Once a promising boundary/slice pair is selected, rebuild its reference with a richer budget such as `reference_wpp=16384` before NC/2LMC time-to-MSE plots are reported.

## Output

Each case directory contains:

- `reference_pure_estimates.csv`
- `reference_mean.ppm`
- `reference_abs_gradient.ppm`
- `reference_highpass.ppm`
- `case_metrics.json`

The combined files are:

- `spatial_frequency_summary.csv`
- `spatial_frequency_summary.json`
- `manifest.json`

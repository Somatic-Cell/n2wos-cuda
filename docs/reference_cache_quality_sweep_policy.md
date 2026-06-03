# Reference-based cache quality sweep

This diagnostic reuses an existing high-sample Pure-WoS numerical reference and
varies only the Neural Cache quality parameters.  It is meant to answer whether
NC+2LMC fails because the cache is too weak, rather than because the reference
or boundary condition is too easy.

The reference must be generated once with `run_nc_slice_reference_time_to_mse.py`
and passed through `--reference-estimates-csv`.  This runner does not rebuild the
reference.  It assumes that the mesh, boundary condition, normalization, slice
view, slice plane, slice resolution, and inside-pixel ordering match the original
reference run.

The runner varies:

- `--cache-presets`, for example `nano,light,baseline`
- `--train-points-list`, for example `5000,20000`
- `--train-steps-per-refresh-list`, for example `50,250,1000`
- NC-only hybrid walks per point
- NC+2LMC coarse/residual allocations

It writes:

- `cache_quality_time_mse_points_reference.csv`
- `cache_quality_time_to_threshold_reference.csv`
- `summary_reference_cache_quality.json`

`solve_only_ms` measures the fixed-snapshot evaluation cost.  `training_plus_solve_ms`
adds the independently measured cache training cost for that particular run.  A
future true online/progressive runner should replace this independent retraining
with a snapshot pipeline; this script is not that pipeline.

Because the executable still computes a minimal Pure-WoS run internally, the
runner sets `--pure-walks-per-point` to a small value and ignores the resulting
pure estimate.  The only reference for reported MSE is the supplied reference CSV.

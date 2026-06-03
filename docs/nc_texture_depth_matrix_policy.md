# Boundary texture / depth matrix policy

This runner is for the stage where analytic boundary conditions are no longer the main benchmark.
For each arbitrary Dirichlet boundary texture, it builds one high-sample Pure WoS numerical reference and reuses that reference for several prefix depths.

The intended first matrix is small:

- `boundary_texture_stripes_k8`
- `boundary_texture_checker_k8`
- `m = 4, 8`
- `cache = nano, light`
- `train_points = 5000`
- `train_steps_per_refresh = 50`

The reference is per boundary condition, mesh, slice, and point ordering.  It must not be reused across a different mesh, boundary, normalization, slice view, slice plane, slice resolution, or inside-pixel ordering.

The script wraps two existing runners:

1. `run_nc_slice_reference_time_to_mse.py` to generate the reference and the low-budget Pure WoS curve.
2. `run_nc_slice_reference_cache_sweep.py` to reuse the reference for NC-only and NC+2LMC at each requested depth.

It writes combined CSV files:

- `texture_depth_time_mse_points_reference.csv`
- `texture_depth_time_to_threshold_reference.csv`

This is still a fixed-snapshot benchmark.  It is not a true online progressive snapshot runner, and it does not reuse residual samples for training.  It is meant to find a boundary/depth operating point where NC-only has a floor and NC+2LMC lowers that floor enough to compete with low-budget Pure WoS.

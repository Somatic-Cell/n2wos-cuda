# Reference-based slice benchmark policy

This benchmark evaluates WoS, NC-WoS, and NC+2LMC against a high-sample Pure WoS numerical reference rather than an analytic exact solution.

Use this mode for complex meshes or arbitrary Dirichlet boundary values where an analytic interior solution is unavailable or where an analytic construction would unduly constrain the boundary signal.

Rules:

1. The reference is a high-sample Pure WoS estimate, not an exact ground truth.
2. Reference and method seeds must be independent.
3. The reference CSV must store per-point mean and sample variance.
4. Method MSE is computed against the reference mean.
5. Report the estimated reference RMSE floor so readers can judge whether reference noise is negligible.
6. Do not reuse reference samples for Neural Cache training labels.

This is still an offline fixed-snapshot benchmark. It does not implement true online progressive training/evaluation overlap.


## Boundary textures and large references

For the main NC-style benchmark, prefer arbitrary Dirichlet boundary textures such as `boundary_texture_stripes_k16` or `boundary_texture_checker_k16`. These are boundary value generators only; they are not analytic interior solutions. Report MSE against the high-sample Pure WoS numerical reference.

Use a large total reference budget, e.g. `--reference-wpp 16384`, but keep each executable call below sample-count limits with `--reference-chunk-wpp 512`. The runner combines per-point reference means and sample variances across independent chunks.

## Streaming reference statistics

Reference chunks must be combined sequentially with a Chan/Welford merge of
per-point means and variances. Do not hold all chunks in memory and do not
reconstruct a large reference from raw `sum` and `sum_sq`. The C++ estimate CSV
writer also uses Welford updates for per-point means and sample variances.

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

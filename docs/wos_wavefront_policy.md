# CUDA-resident wavefront WoS policy

Patch 0004 introduces the first solver-level GPU-resident sampling engine. It is
not yet the final time-to-MSE implementation, but it establishes the common data
path that later pure WoS, Neural Cache, and 2LMC evaluations should share.

Production constraints kept by this patch:

- walker positions remain in CUDA device memory;
- RNG state is per sample and stored in CUDA device memory;
- closest-point query input and output remain in CUDA device memory;
- boundary evaluation for the harmonic debug problem runs in CUDA kernels;
- reductions are computed on the GPU before a final summary readback;
- FCPW public host-vector APIs are not used;
- there is no CSV or Python postprocess in the timing path.

The current geometry backend is cuBQL. Because cuBQL is used as a batched query
stage rather than as a device-callable function inside the update kernel, the
engine uses a wavefront schedule:

1. query closest point for the current position array;
2. update active walkers and values in a CUDA kernel;
3. repeat for a fixed global step budget.

This is not one launch per walk. It is one batched query stage plus one update
stage per global step. The host controls the global step loop, but it does not
read walker data back inside the loop.

Patch 0004 has no queue compaction. Instead, it adds a masked cuBQL query entry
point. Inactive slots are still covered by the kernel launch, but they skip BVH
traversal. This avoids a per-step active-count readback while preventing the
worst waste from completed walks.

The debug cache is analytic: `u(x,y,z)=x^2-y^2`. The `oracle_coarse` and
`oracle_residual` modes are not claims about neural-cache performance. They are
scaffolding for the later TCNN-backed two-level estimator.


## 0004b note: coarse-level launch cap

The oracle coarse estimator is intentionally capped at `depth_m + 1` scheduled
query rounds.  The original 0004 scaffold ran all methods for `max_steps + 1`
rounds, even after all coarse samples had stopped at depth `m`.  That made the
cheap level look artificially expensive and should not be used for 2LMC cost
reasoning.

Pure WoS and residual continuation still use `max_steps + 1` scheduled rounds
until a later active-queue compaction or device-side termination mechanism is
implemented.  The JSON field `scheduled_query_rounds` records this explicitly.

## 0004c: oracle depth/allocation sweep

`n2wos_eval_wavefront_wos` is the common solver executable. The helper script
`scripts/run_wavefront_oracle_sweep.py` only orchestrates repeated runs of that
executable and summarizes the resulting JSON files. It must not be counted as a
separate timing path.

The sweep is intended to answer two questions before connecting tiny-cuda-nn to
the solver:

1. Which prefix depth `m` produces useful residual variance reduction under the
   current GPU-resident geometry backend?
2. Given measured per-sample costs and sample variances, what coarse/residual
   allocation would be predicted by the standard two-level allocation rule?

For independent coarse and residual estimators,

```text
mu_hat = mean(C) + mean(W - C)
```

with sample variances `V_c`, `V_r` and per-sample costs `c_c`, `c_r`, the
predicted optimal coarse-to-residual sample ratio is

```text
n_c / n_r = sqrt((V_c * c_r) / (V_r * c_c)).
```

The predicted optimal variance-time product is

```text
(sqrt(V_c * c_c) + sqrt(V_r * c_r))^2.
```

The script writes both the actual score for the requested allocation and this
predicted optimum score. These scores are diagnostic only; the current engine
still uses a global-step wavefront without active compaction and is not a final
wall-clock implementation.

## 0004d persistent per-sample diagnostic engine

The initial `wavefront` engine keeps all walker state on the GPU and avoids
per-walk kernel launches, but the host still controls the global step loop.  It
therefore launches one cuBQL query kernel and one update kernel per scheduled
step.  This is useful for matching a future batched TCNN inference design, but
it can overstate launch overhead in single-point diagnostics.

`--engine persistent` adds a diagnostic engine in which one CUDA thread owns one
walk and performs the cuBQL closest-point traversal loop directly on device.
This path has no host-controlled step loop and no CPU-GPU transfer inside the
sampling loop.  It is intended to answer whether the poor oracle 2LMC score is
caused mainly by the global-step wavefront scaffold or by the estimator/cost
trade-off itself.

The persistent engine is still not the final NC+2LMC implementation: TCNN cache
inference is not called inside the persistent kernel.  The expected production
layout remains a hybrid design:

```text
prefix persistent kernel to X_m
batched tiny-cuda-nn inference on X_m
persistent continuation/residual kernel
GPU reduction
```

Use the same executable and method flags, adding `--engine persistent`.

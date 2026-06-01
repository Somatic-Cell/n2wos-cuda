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

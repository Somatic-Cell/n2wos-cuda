# tiny-cuda-nn cache policy

Patch 0003 introduces tiny-cuda-nn only as an optional native C++/CUDA backend. The production timing path must not use the Python binding, CSV dumps, or host-side postprocess for cache inference.

## Required interface shape

The cache path used by NC / NC+2LMC must satisfy the following constraints:

- Cache inputs are CUDA device buffers, e.g. the m-step continuation points X_m.
- Cache outputs are CUDA device buffers, e.g. C_theta(X_m).
- A downstream CUDA kernel can consume C_theta(X_m) without an intermediate CPU copy.
- Training batches are eventually generated from GPU-resident WoS samples or copied only at block boundaries for diagnostics; per-sample CSV export is non-production.
- Snapshot discipline is enforced outside the probe: inference model theta_k is read-only during block k, and training updates produce theta_{k+1} after block k contributions have been added to the estimator.

## Patch 0003 scope

`n2wos_probe_tcnn_cache` is a connectivity probe, not a WoS solver. It generates synthetic 3D inputs on the GPU, trains a small HashGrid + MLP cache on the harmonic target x^2 - y^2, runs tiny-cuda-nn native C++ inference into a GPU output matrix, then launches a separate CUDA consumer kernel that computes validation statistics directly from those GPU outputs.

This verifies the key interface requirement before the common WoS / NC / 2LMC sampling engine is implemented.

## JIT fusion

Manual or automatic JIT fusion is not required for patch 0003. The first production path should use native batched inference. JIT fusion may be enabled later and benchmarked separately because WoS traversal is divergent, whereas tiny-cuda-nn fused kernels have warp-level assumptions that may not combine well with a branchy random-walk kernel.

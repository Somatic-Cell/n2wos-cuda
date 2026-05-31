# Geometry backend policy after patch 0002a

Patch 0001 proved only that the in-tree median-split CUDA BVH can produce plausible closest-point answers on the procedural bumpy sphere. It did not prove equivalence to FCPW, cuBQL, OptiX, or any production-grade geometry library.

Patch 0002 introduced NVIDIA/cuBQL as a device-resident candidate. Patch 0002a changes the decision process from "try one cuBQL builder" to "sweep the plausible cuBQL builders under WoS-like query distributions."

| Backend | Intended role | Production timing status |
|---|---|---|
| `custom_cuda_bvh` | Comparison / fallback / regression target | Not trusted as the main backend yet |
| `cubql_cuda/<method>` | Primary production candidate family | Candidate only when compiled, validated, and benchmarked |
| FCPW public `GPUScene` | Reference / non-production baseline if added later | Not allowed in production timing if host-vector I/O remains |
| OptiX | Not planned for closest-point WoS | Not applicable unless the solver is reformulated |

The production path must expose a device-pointer query interface:

```cpp
query_device(d_points, query_count, d_distance2, d_closest, d_triangle_id, d_overflow, block_size, stream)
```

This interface is present for both `custom_cuda_bvh` and, when cuBQL is enabled, `cubql_cuda/<method>`. Host-vector `query(...)` wrappers remain only for probes and validation. They are not the intended path for WoS kernels or wavefront stages.

## cuBQL builders to test

cuBQL's `BuildConfig` exposes `SPATIAL_MEDIAN`, `SAH`, and experimental `ELH`. The CUDA builder header also exposes explicit `radixBuilder` and `rebinRadixBuilder` entry points. Patch 0002a supports these names:

```text
spatial_median
sah
elh
radix
```

Use:

```bash
--cubql-build-methods sweep
```

as shorthand for all currently enabled methods.

Suggested interpretation:

```text
sah:
  Highest-quality candidate for a static mesh if it wins query time.
  Build can be more expensive, but WoS performs many steady-state queries per mesh.

rebin_radix:
  Disabled for current cuBQL main. `cuBQL::cuda::rebinRadixBuilder` is declared in `builder/cuda.h`, but the included implementation headers do not provide a linkable definition. Re-enable only after cuBQL exposes an implemented entry point.

radix:
  Fast-builder candidate. Keep only if it validates on near-surface and OBJ stress tests.

spatial_median:
  Simple default/reference candidate. It should remain in every sweep.

elh:
  Experimental cuBQL heuristic. Keep in sweeps until measured; do not make it the default without evidence.
```

The production default is not hard-coded by name. Select the validated cuBQL method with the lowest median device query time on `wos_like_prefix` or `near_boundary_shell`. Report build time separately. For static meshes, it is acceptable for `sah` to have a larger one-time build cost if it materially improves query time.

## Query distributions

Patch 0002a adds query modes:

```text
uniform_box:
  Original synthetic box distribution.

interior_slice:
  z=0 slice distribution, closer to dense field evaluation.

near_boundary_shell:
  Area-weighted surface samples offset by a small distance. This stresses WoS terminal-region closest-point queries.

wos_like_prefix:
  Synthetic mixture of near-surface shell and slice points. It avoids CPU brute-force WoS generation while approximating the fact that production WoS will spend many queries near the boundary.
```

The synthetic `wos_like_prefix` distribution is not a production WoS sampler. It is only a geometry stress test before implementing the common WoS / NC / 2LMC engine.


## Mesh import policy

cuBQL consumes triangle arrays and builds/query BVHs; it does not import mesh files. Patch 0002b adds an in-tree PLY loader for Stanford-style meshes so geometry backend tests can use the same mesh loading path as future WoS / NC / 2LMC executables.

Supported PLY subset:

```text
ascii 1.0
binary_little_endian 1.0
vertex element with scalar x, y, z properties
face element with a list property such as vertex_indices or vertex_index
polygonal faces triangulated by a fan
extra scalar/list properties skipped
```

Unsupported PLY subset:

```text
binary_big_endian
non-face mesh primitives
per-face holes or non-simple polygons
```

Assimp remains an option for a later converter tool, but it is intentionally not introduced into the core timing build.

## Minimum acceptance criteria before WoS kernels

Run at least:

```bash
./build/cuda-release-cubql/n2wos_probe_geometry_backends \
  --mesh procedural_bumpy_sphere \
  --bumpy-stacks 128 \
  --bumpy-slices 256 \
  --query-mode wos_like_prefix \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --cubql-build-methods sweep \
  --output results/probe_geometry_backends_bumpy_woslike_sweep.json
```

and, for at least one file mesh. PLY is preferred for Stanford-style data:

```bash
./build/cuda-release-cubql/n2wos_probe_geometry_backends \
  --mesh ply \
  --mesh-path meshes/processed/<mesh>.ply \
  --normalize 1 \
  --query-mode near_boundary_shell \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --cubql-build-methods sweep \
  --output results/probe_geometry_backends_ply_near_boundary_sweep.json
```

Accept only a backend with:

```text
bad_distance_count_threshold_2e_4 == 0
finite distance values
no overflow or query failure
no host transfer in timed query path
```

Triangle id agreement is diagnostic only. Distinct but geometrically coincident closest triangles can occur near edges, vertices, and symmetric regions.

## Relation to Neural Cache and 2LMC

The geometry backend must remain exact mesh-based for the main claim. A neural distance field would change the WoS path distribution and can introduce bias not corrected by the Neural Cache residual. Neural Cache approximation error can be corrected by fixed-snapshot 2LMC residuals; distance-field geometry error is a different error source.

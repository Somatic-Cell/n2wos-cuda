# Geometry backend policy after patch 0002

Patch 0001 proved only that the in-tree median-split CUDA BVH can produce plausible closest-point answers on the procedural bumpy sphere. It did not prove equivalence to FCPW, cuBQL, OptiX, or any production-grade geometry library.

Patch 0002 changes the status of geometry backends:

| Backend | Intended role | Production timing status |
|---|---|---|
| `custom_cuda_bvh` | Comparison / fallback / regression target | Not trusted as the main backend yet |
| `cubql_cuda` | Primary production candidate | Candidate only when compiled, validated, and benchmarked |
| FCPW public `GPUScene` | Reference / non-production baseline if added later | Not allowed in production timing if host-vector I/O remains |
| OptiX | Not planned for closest-point WoS | Not applicable unless the solver is reformulated |

The production path must expose a device-pointer query interface:

```cpp
query_device(d_points, query_count, d_distance2, d_closest, d_triangle_id, d_overflow, block_size, stream)
```

This interface is now present for both `custom_cuda_bvh` and, when cuBQL is enabled, `cubql_cuda`. The host-vector `query(...)` wrappers remain only for probes and validation. They are not the intended path for WoS kernels or wavefront stages.

## Why cuBQL

NVIDIA/cuBQL is a CUDA BVH build-and-query library. Its README describes GPU-side BVH construction, device-memory BVH output, traversal templates, and closest-surface-point triangle examples. This matches the current requirement better than FCPW public `GPUScene`, whose public C++ API is host-vector oriented.

Patch 0002 uses the cuBQL triangle closest-point sample path:

```cpp
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_TRIANGLE_CPAT_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include <cuBQL/queries/triangleData/math/pointToTriangleDistance.h>
#include <cuBQL/queries/triangleData/closestPointOnAnyTriangle.h>

cuBQL::triangles::CPAT cpat;
cpat.runQuery(d_triangles, bvh, query_point);
```

## Acceptance criteria before WoS kernels

Before implementing pure WoS on top of a backend, run at least:

```bash
./build/cuda-release/n2wos_probe_geometry_backends \
  --mesh procedural_bumpy_sphere \
  --bumpy-stacks 128 \
  --bumpy-slices 256 \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --output results/probe_geometry_backends_bumpy.json
```

and, for at least one OBJ mesh:

```bash
./build/cuda-release/n2wos_probe_geometry_backends \
  --mesh obj \
  --mesh-path meshes/processed/<mesh>.obj \
  --normalize 1 \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --output results/probe_geometry_backends_obj.json
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

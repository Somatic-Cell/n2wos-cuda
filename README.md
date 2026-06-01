# n2wos-cuda

CUDA-resident geometry and Walk-on-Spheres infrastructure for Neural Cache + two-level Monte Carlo WoS.

This repository is intentionally separate from the pilot `n2wos` repository. The production timing path is expected to keep walker state, random number generation, closest-point queries, boundary evaluation, accumulation, and reductions on the GPU. Python is reserved for orchestration, plotting, and packaging results.

## Patch 0001 scope

`0001-bootstrap-cuda-resident-geometry.patch` adds only the geometry bootstrap:

- CMake/CUDA project scaffold.
- Procedural watertight bumpy sphere mesh.
- Minimal OBJ loader with triangle/quad face support.
- Mesh normalization to centered unit radius.
- CPU brute-force closest-point reference.
- Host-built BVH copied to compact device arrays.
- Custom CUDA BVH closest-point traversal.
- `n2wos_probe_cuda_bvh` validation/throughput probe.
- JSON result output with implementation-mode labels.
- Result bundler script.

It deliberately does not add WoS, 2LMC, tiny-cuda-nn, or FCPW integration. Those come after geometry correctness is stable.

## Build

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release -j
```

The default preset targets `sm_86`, matching RTX 3070-class GPUs. Override if needed:

```bash
cmake -S . -B build/cuda-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda-release -j
```

## Run the CUDA BVH probe

Procedural bumpy sphere:

```bash
./build/cuda-release/n2wos_probe_cuda_bvh \
  --mesh procedural_bumpy_sphere \
  --bumpy-stacks 128 \
  --bumpy-slices 256 \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --output results/probe_cuda_bvh.json
```

OBJ mesh:

```bash
./build/cuda-release/n2wos_probe_cuda_bvh \
  --mesh obj \
  --mesh-path meshes/processed/bunny.obj \
  --normalize 1 \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --output results/probe_cuda_bvh_bunny.json
```

Package results:

```bash
python3 scripts/collect_experiment_bundle.py --results-dir results
```

The default bundle is written to `results/share_latest.tar.gz`.

## JSON implementation labels

The probe writes labels intended to prevent accidental promotion of pilot/non-production paths:

```json
{
  "implementation_mode": {
    "gpu_resident_geometry_query": true,
    "geometry_query": "custom_cuda_bvh",
    "fcpw_gpu_public_api": false,
    "host_driven_walker_loop": false,
    "cache_inference": "none",
    "rng": "host_query_generation_only",
    "accumulation": "not_applicable_patch_0001"
  }
}
```

Patch 0003 should add a tiny-cuda-nn device-resident cache probe after the geometry backend decision.

## Patch 0002 / 0002a scope

`0002-audit-and-integrate-cubql-geometry.patch` and `0002a-geometry-backend-stress-and-cubql-builder-sweep.patch` change the geometry plan:

- The in-tree median-split CUDA BVH from 0001 remains available, but only as `custom_cuda_bvh` comparison/fallback.
- A device-pointer query API is added to the in-tree backend so future WoS/wavefront code can avoid host transfers.
- Optional NVIDIA cuBQL integration is added behind `-DN2WOS_ENABLE_CUBQL=ON`.
- A cuBQL-backed `CuBqlBvh` exposes the same device-pointer closest-point query API.
- `n2wos_probe_geometry_backends` compares enabled geometry backends under the same query set and timing scope.
- 0002a adds cuBQL builder sweeps: `spatial_median`, `sah`, `elh`, and `radix`. `rebin_radix` is disabled for current cuBQL main because `rebinRadixBuilder` is declared but not defined by the included implementation headers.
- 0002a adds query distributions: `uniform_box`, `interior_slice`, `near_boundary_shell`, and `wos_like_prefix`.
- `docs/geometry_backend_policy.md` records the backend policy and acceptance criteria.

The production rule remains: no per-step CPU-GPU transfers, no FCPW public host-vector loop in timing, and no Python/CSV path for geometry or cache inference.

### Fetch cuBQL

cuBQL is not vendored by this patch. Fetch it explicitly:

```bash
python3 scripts/fetch_cubql.py --dest external/cuBQL
```

Then configure with cuBQL enabled:

```bash
rm -rf build/cuda-release
cmake -S . -B build/cuda-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DN2WOS_ENABLE_CUBQL=ON \
  -DN2WOS_CUBQL_DIR=$PWD/external/cuBQL
cmake --build build/cuda-release -j
```

Without cuBQL, the project still builds and the new probe reports `cubql_cuda` as disabled.

### Run geometry backend comparison

```bash
./build/cuda-release/n2wos_probe_geometry_backends \
  --mesh procedural_bumpy_sphere \
  --bumpy-stacks 128 \
  --bumpy-slices 256 \
  --query-mode wos_like_prefix \
  --queries 262144 \
  --validate 2048 \
  --repeat 10 \
  --cubql-build-methods sweep \
  --output results/probe_geometry_backends.json
```

The timing scope is device query kernel-only. Query points are uploaded once before timing; output is copied back after timing for validation. cuBQL build time is reported separately and is not included in steady-state query timings. This is intentionally different from the host-vector probe wrapper and is the interface shape expected for future WoS stages.


## Mesh input

Patch 0002b adds a lightweight PLY loader for Stanford-style triangle meshes. Supported formats are ASCII and binary little-endian PLY files with `vertex` x/y/z properties and polygonal `face` vertex index lists. Faces with more than three vertices are triangulated by a fan. Binary big-endian PLY is rejected; convert it before use. OBJ loading remains available for small tests. cuBQL builds and queries BVHs from the loaded triangle mesh, but it does not import mesh files.


## Patch 0003 scope

`0003-add-tcnn-device-resident-cache-probe.patch` adds optional native tiny-cuda-nn integration. It does not add WoS yet. The new executable verifies the cache interface needed by NC and NC+2LMC:

- GPU-side generation of training and inference point batches.
- tiny-cuda-nn C++/CUDA training on a small HashGrid + MLP cache.
- tiny-cuda-nn C++/CUDA inference from a GPU input matrix to a GPU output matrix.
- A downstream CUDA kernel consumes the output without a CPU transfer between TCNN and the consumer.
- JSON labels explicitly reject Python bindings and CSV postprocess as production cache paths.

Fetch tiny-cuda-nn recursively:

```bash
python3 scripts/fetch_tcnn.py --dest external/tiny-cuda-nn
```

Build the TCNN probe:

```bash
rm -rf build/cuda-release-tcnn
cmake --preset cuda-release-tcnn
cmake --build --preset cuda-release-tcnn -j
```

Run it:

```bash
./build/cuda-release-tcnn/n2wos_probe_tcnn_cache \
  --samples 262144 \
  --batch-size 262144 \
  --train-steps 200 \
  --repeat 20 \
  --output results/probe_tcnn_cache.json
```

Use `cuda-release-cubql-tcnn` when geometry and cache probes should be built in the same tree.

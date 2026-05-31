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

Patch 0002 should add GPU-resident pure WoS using the same geometry query path.

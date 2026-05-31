# Mesh data

Place downloaded source meshes in `meshes/raw/` and normalized/watertight meshes in `meshes/processed/`.

For paper experiments, each mesh should have a small metadata file recording:

- source URL,
- license,
- original filename,
- processed filename,
- normalization transform,
- vertex count,
- triangle count,
- watertightness status,
- notes on repairs or simplification.

Patch 0001 can run without external mesh data by using `--mesh procedural_bumpy_sphere`.


## PLY meshes

Stanford-style `.ply` triangle meshes can be placed under `meshes/raw/` or `meshes/processed/` and loaded with `--mesh ply --mesh-path <path>`. The in-tree loader supports ASCII and binary little-endian PLY files with `vertex` x/y/z properties and a list face property such as `vertex_indices`. It ignores normals, colors, and other scalar properties.

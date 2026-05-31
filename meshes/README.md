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

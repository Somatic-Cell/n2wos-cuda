# Slice figure workflow

This document describes a lightweight figure path for short-paper figures.  It
keeps solver output, texture generation, 3D rasterization, and final figure layout
separate.

## Dependencies

```bash
python -m pip install numpy pillow matplotlib pyvista trimesh
```

On a headless WSL or remote Linux session, VTK/PyVista may need Xvfb:

```bash
sudo apt-get install -y xvfb
xvfb-run -a python scripts/render_slice_figure.py --help
```

## 1. Run a slice evaluation with per-point estimates

The renderer expects the evaluator to write a slice mask, slice point CSV, and
per-point estimates CSV.  Use `--eval-mode slice` and `--save-estimates-prefix`.

```bash
./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos \
  --mesh procedural_bumpy_sphere \
  --boundary harmonic_zebra_k8 \
  --label-source wos_supervision \
  --train-sampler rejection \
  --cache-preset nano \
  --train-points 20000 \
  --eval-mode slice \
  --slice-view xy \
  --slice-plane 0.0 \
  --slice-width 512 \
  --slice-height 512 \
  --label-refreshes 4 \
  --walks-per-label-refresh 50 \
  --train-steps-per-refresh 5000 \
  --pure-walks-per-point 64 \
  --hybrid-walks-per-point 4 \
  --coarse-walks-per-point 64 \
  --residual-walks-per-point 32 \
  --enable-2lmc 1 \
  --depth-m 2 \
  --save-estimates-prefix figures/zebra_k8_m2 \
  --output figures/zebra_k8_m2.json
```

## 2. Render the neural-cache slice on the clipped mesh

```bash
python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --value-column nc_wos_mean \
  --texture-output figures/zebra_k8_nc_texture.png \
  --texture-meta figures/zebra_k8_nc_texture.json \
  --render-output figures/zebra_k8_nc_render.png \
  --colorbar-output figures/zebra_k8_nc_colorbar.png \
  --colorbar-meta figures/zebra_k8_nc_colorbar.json \
  --cmap coolwarm \
  --show-cut
```

## 3. Render a signed error slice

```bash
python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --value-column nc_wos_mean \
  --subtract-column analytic_value \
  --symmetric-range \
  --texture-output figures/zebra_k8_nc_error_texture.png \
  --texture-meta figures/zebra_k8_nc_error_texture.json \
  --render-output figures/zebra_k8_nc_error_render.png \
  --colorbar-output figures/zebra_k8_error_colorbar.png \
  --colorbar-meta figures/zebra_k8_error_colorbar.json \
  --cmap coolwarm \
  --show-cut
```

For absolute error, add `--abs-value` and normally use a sequential colormap such
as `magma` or `viridis`.

## 4. Render mesh only

```bash
python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --mesh-only \
  --render-output figures/mesh_only.png \
  --show-cut
```

## Notes

- The script writes colorbar range metadata to JSON.  Put tick labels in the PDF
  or vector-graphics editor using those numeric `vmin` and `vmax` values.
- If the texture appears flipped, flip the generated texture PNG for the final
  figure or add a one-line image transpose before rendering.  The data texture is
  intentionally kept as a normal image file so this edit is simple.
- For exact mesh geometry, export/pass the same normalized OBJ/PLY used by the
  solver.  The built-in `procedural_bumpy_sphere` is adequate for figure layout
  but may not exactly match the C++ procedural phase.

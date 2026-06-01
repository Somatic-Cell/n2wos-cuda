# Field visualization diagnostics

`n2wos_render_harmonic_slice` is a visual diagnostic tool for the current
single-boundary-condition implementation. It renders a z-slice for the analytic
harmonic target

```text
u(x,y,z) = x^2 - y^2
```

using the same cuBQL geometry backend and persistent per-sample WoS kernel style
used by the pointwise diagnostic executable.

The tool writes:

```text
<prefix>.json
<prefix>.csv
<prefix>_exact.ppm
<prefix>_estimate.ppm
<prefix>_error.ppm
<prefix>_stderr.ppm
```

The CSV contains per-pixel coordinates, the Monte Carlo estimate, the exact
value, error, stderr, mean step count, and forced/overflow counts. The PPM files
are intentionally dependency-free and can be opened by common image viewers or
converted with ImageMagick, Python, or other plotting tools.

This tool is not a final time-to-MSE benchmark. It is for checking whether the
solver is qualitatively solving the right field and whether geometry masking,
normalization, and stopping behavior look plausible before harder boundary
conditions and Neural Cache integration are added.

For the current low-frequency harmonic polynomial, Neural Cache and two-level
methods may have little room to beat a persistent pure WoS baseline. Harder tests
should add external-charge boundary values and more varied meshes before making
any broad speed claim.


## Slice geometry and apparent distortion

The renderer shows a planar cross-section of the solid, not a camera projection of
the surface.  For example, the default `--view xy --plane-z 0` means the set of
points `(x, y, 0)` that are inside the closed mesh.  A bunny may therefore look
unexpected if the selected plane is not the view direction you had in mind.

Use the mask image first when diagnosing geometry:

```text
<prefix>_mask.ppm
```

The exact and estimate images are scalar heatmaps over the same inside mask; they
are not intended to be shape silhouettes.  As of patch 0004h, exact and estimate
use a shared scalar range, the renderer preserves world aspect by default, and
`--view xy|xz|yz` can be used to inspect all three principal slices.

Examples:

```bash
./build/cuda-release-cubql/n2wos_render_harmonic_slice \
  --mesh ply --mesh-path meshes/bun_zipper_wataertight.ply \
  --view xy --plane-z 0 --output-prefix results/bunny_xy

./build/cuda-release-cubql/n2wos_render_harmonic_slice \
  --mesh ply --mesh-path meshes/bun_zipper_wataertight.ply \
  --view xz --plane-z 0 --output-prefix results/bunny_xz

./build/cuda-release-cubql/n2wos_render_harmonic_slice \
  --mesh ply --mesh-path meshes/bun_zipper_wataertight.ply \
  --view yz --plane-z 0 --output-prefix results/bunny_yz
```

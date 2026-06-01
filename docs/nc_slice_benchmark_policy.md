# Slice benchmark policy

The random 8192-point evaluator is a development diagnostic. Main comparisons
against Neural Cache-style work should use a fixed two-dimensional slice through
the domain.

`n2wos_eval_tcnn_nc_wos` now supports:

```text
--eval-mode slice
--slice-width 512
--slice-height 512
--slice-view xy|xz|yz
--slice-plane 0
--slice-frame u_min,u_max,v_min,v_max
--slice-output-prefix results/<name>
```

The evaluator places a regular grid on the selected plane, keeps only pixels
whose centers are inside the closed mesh according to a CPU ray-parity mask, and
uses these interior pixels as evaluation points. It writes:

```text
<prefix>_mask.ppm
<prefix>_points.csv
```

Training points are unchanged: the current implementation still uses the
screening sampler. A later patch should add full-domain rejection sampling and a
train-point-count sweep.

Interpretation notes:

- `candidate_pixels = width * height` is not the actual number of evaluated
  points.
- `inside_pixels` is the actual evaluation point count.
- The mask is a diagnostic image, not a solver output.
- For arbitrary boundary values on complex meshes, prefer a high-sample pure-WoS
  numerical reference over an analytic reference unless the analytic harmonic
  condition has been verified for the mesh.

# Neural Cache slice visualization

This diagnostic renders the trained tiny-cuda-nn/iNGP cache field on a fixed
slice.  It is intended to answer whether the cache has learned a plausible
solution-like field before interpreting NC-WoS or NC+2LMC timing.

Use `--depth-m 0` for direct cache visualization, i.e. `C_theta(x)` at the slice
points.  For `m > 0`, `nc_wos_mean` is the average of `C_theta(X_m)` after an
m-step WoS prefix, so it is no longer a direct image of the neural field at the
slice coordinates.

When a high-sample Pure-WoS reference CSV is available, pass it with
`--reference-estimates-csv`.  The script then writes cache/reference images on a
shared color scale and cache-minus-reference error images.

The generated images are PPM files to avoid optional plotting dependencies.
Convert them with ImageMagick if needed:

```bash
magick results/cache_slice_visualization_stripes16/cache_nc_wos_mean.ppm cache.png
```

This visualization is not part of the timing path.  It may run training and a
minimal evaluation pass, but the generated PPM/CSV/JSON I/O should not be mixed
with solver wall-clock claims.

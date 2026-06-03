# Neural Cache visualization policy

Neural Cache experiments must generate plane images, not only CSV / JSON summaries.
The required diagnostic images are:

1. reference field, shared scale
2. cache field, shared scale
3. signed cache error
4. absolute cache error
5. slice mask

The intent is to catch failures such as untrained caches, collapsed amplitude,
boundary-region errors, noisy label artifacts, and training-distribution mismatch
before launching expensive sweeps.

## Single case

```bash
python3 scripts/generate_nc_visualizations.py \
  --estimates-csv results/run/run_estimates.csv \
  --mask-ppm results/run/run_slice_mask.ppm \
  --reference-estimates-csv results/reference/reference_pure_estimates.csv \
  --print-summary
```

## Results tree

```bash
python3 scripts/run_nc_visualization_postprocess.py \
  --results-root results/bunny_onecheck \
  --reference-estimates-csv results/references/bunny/reference_pure_estimates.csv \
  --print-summary
```

Outputs are written under `figures/` next to each processed estimates CSV.
Both PPM and PNG files are produced.

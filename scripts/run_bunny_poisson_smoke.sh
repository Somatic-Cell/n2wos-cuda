#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos}
MESH=${MESH:-meshes/bunny_zipper_wataertight.ply}
OUT=${OUT:-results/poisson_bunny_z012}
mkdir -p "$OUT"

for BOUNDARY in poisson_multiscale poisson_figlike_hf; do
  "$EXE" \
    --mesh ply \
    --mesh-path "$MESH" \
    --boundary "$BOUNDARY" \
    --label-source exact_analytic \
    --train-sampler rejection \
    --cache-preset custom \
    --n-levels 12 \
    --n-features-per-level 2 \
    --log2-hashmap-size 18 \
    --base-resolution 16 \
    --per-level-scale 1.5 \
    --n-neurons 32 \
    --n-hidden-layers 2 \
    --learning-rate 1e-2 \
    --train-points 20000 \
    --eval-mode slice \
    --slice-view xy \
    --slice-plane 0.12 \
    --slice-width 512 \
    --slice-height 512 \
    --label-refreshes 1 \
    --walks-per-label-refresh 1 \
    --train-steps-per-refresh 5000 \
    --pure-walks-per-point 16 \
    --hybrid-walks-per-point 1 \
    --coarse-walks-per-point 16 \
    --residual-walks-per-point 8 \
    --enable-2lmc 1 \
    --depth-m 0 \
    --save-estimates-prefix "$OUT/${BOUNDARY}_exact_m0" \
    --output "$OUT/${BOUNDARY}_exact_m0.json"
done

python scripts/collect_nc_results.py \
  --inputs "$OUT/*.json" \
  --out-csv "$OUT/summary.csv" \
  --out-json "$OUT/summary.json"

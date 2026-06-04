#!/usr/bin/env bash
set -euo pipefail

EXE="${EXE:-./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos}"
MESH="${MESH:-meshes/bunny_zipper_wataertight.ply}"
OUT="${OUT:-results/hmix_multiscale_bunny_z012}"
mkdir -p "$OUT"

COMMON=(
  --mesh ply
  --mesh-path "$MESH"
  --train-sampler rejection
  --cache-preset custom
  --n-levels 12
  --n-features-per-level 2
  --log2-hashmap-size 18
  --base-resolution 16
  --per-level-scale 1.5
  --n-neurons 32
  --n-hidden-layers 2
  --learning-rate 1e-2
  --train-points 20000
  --eval-mode slice
  --slice-view xy
  --slice-plane 0.12
  --slice-width 512
  --slice-height 512
  --max-steps 256
  --epsilon 1e-4
  --cubql-build-method sah
  --enable-2lmc 1
)

run_exact () {
  local B="$1"
  "$EXE" "${COMMON[@]}" \
    --boundary "$B" \
    --label-source exact_analytic \
    --label-refreshes 1 \
    --walks-per-label-refresh 1 \
    --train-steps-per-refresh 5000 \
    --pure-walks-per-point 16 \
    --hybrid-walks-per-point 1 \
    --coarse-walks-per-point 16 \
    --residual-walks-per-point 8 \
    --depth-m 0 \
    --save-estimates-prefix "$OUT/${B}_exact_m0" \
    --output "$OUT/${B}_exact_m0.json"
}

run_wos () {
  local B="$1"
  "$EXE" "${COMMON[@]}" \
    --boundary "$B" \
    --label-source wos_supervision \
    --label-refreshes 4 \
    --walks-per-label-refresh 50 \
    --train-steps-per-refresh 500 \
    --pure-walks-per-point 64 \
    --hybrid-walks-per-point 4 \
    --coarse-walks-per-point 64 \
    --residual-walks-per-point 48 \
    --depth-m 4 \
    --save-estimates-prefix "$OUT/${B}_wos_m4_c64_r48" \
    --output "$OUT/${B}_wos_m4_c64_r48.json"
}

for B in harmonic_mixture_multiscale harmonic_mixture_figlike_hf; do
  run_exact "$B"
  run_wos "$B"
done

python scripts/collect_nc_results.py \
  --inputs "$OUT/*.json" \
  --out-csv "$OUT/summary.csv" \
  --out-json "$OUT/summary.json"

#!/usr/bin/env bash
set -euo pipefail

EXE="${EXE:-./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos}"
MESH="${MESH:-meshes/bunny_zipper_wataertight.ply}"
OUT="${OUT:-results/poisson_figlike_hf_x5_bunny_z012}"

mkdir -p "$OUT"

COMMON=(
  --mesh ply
  --mesh-path "$MESH"
  --boundary poisson_figlike_hf_x5
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
)

echo "[exact] poisson_figlike_hf_x5 m0"
"$EXE" \
  "${COMMON[@]}" \
  --label-source exact_analytic \
  --label-refreshes 1 \
  --walks-per-label-refresh 1 \
  --train-steps-per-refresh 5000 \
  --pure-walks-per-point 16 \
  --hybrid-walks-per-point 1 \
  --coarse-walks-per-point 16 \
  --residual-walks-per-point 8 \
  --enable-2lmc 1 \
  --depth-m 0 \
  --save-estimates-prefix "$OUT/poisson_figlike_hf_x5_exact_m0" \
  --output "$OUT/poisson_figlike_hf_x5_exact_m0.json"

for CR in "32 32" "64 48"; do
  read -r C R <<< "$CR"
  NAME="poisson_figlike_hf_x5_wos_m4_c${C}_r${R}"
  echo "[wos] $NAME"
  "$EXE" \
    "${COMMON[@]}" \
    --label-source wos_supervision \
    --label-refreshes 4 \
    --walks-per-label-refresh 50 \
    --train-steps-per-refresh 500 \
    --pure-walks-per-point 64 \
    --hybrid-walks-per-point 4 \
    --coarse-walks-per-point "$C" \
    --residual-walks-per-point "$R" \
    --enable-2lmc 1 \
    --depth-m 4 \
    --save-estimates-prefix "$OUT/$NAME" \
    --output "$OUT/$NAME.json"
done

python scripts/collect_nc_results.py \
  --inputs "$OUT/*.json" \
  --out-csv "$OUT/summary.csv" \
  --out-json "$OUT/summary.json"

echo
echo "[done] wrote:"
echo "  $OUT/summary.csv"
echo "  $OUT/summary.json"

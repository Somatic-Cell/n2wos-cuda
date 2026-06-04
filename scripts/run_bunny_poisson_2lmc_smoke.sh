#!/usr/bin/env bash
set -euo pipefail

EXE="${EXE:-./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos}"
MESH="${MESH:-meshes/bunny_zipper_wataertight.ply}"
OUT="${OUT:-results/poisson_bunny_z012_2lmc}"

mkdir -p "$OUT"

run_one () {
  local boundary="$1"
  local name="$2"

  echo
  echo "[run] ${name}"

  "$EXE" \
    --mesh ply \
    --mesh-path "$MESH" \
    --boundary "$boundary" \
    --label-source wos_supervision \
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
    --label-refreshes 4 \
    --walks-per-label-refresh 50 \
    --train-steps-per-refresh 500 \
    --pure-walks-per-point 64 \
    --hybrid-walks-per-point 4 \
    --coarse-walks-per-point 64 \
    --residual-walks-per-point 48 \
    --enable-2lmc 1 \
    --depth-m 4 \
    --save-estimates-prefix "$OUT/${name}" \
    --output "$OUT/${name}.json"
}

run_one poisson_multiscale poisson_multiscale_wos_m4_c64_r48
run_one poisson_figlike_hf poisson_figlike_hf_wos_m4_c64_r48

python scripts/collect_nc_results.py \
  --inputs "$OUT/*.json" \
  --out-csv "$OUT/summary.csv" \
  --out-json "$OUT/summary.json"

echo
echo "[done]"
echo "  $OUT/summary.csv"
echo "  $OUT/summary.json"

#!/usr/bin/env bash
set -euo pipefail

EXE="${EXE:-./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos}"
MESH="${MESH:-meshes/bunny_zipper_wataertight.ply}"
OUT="${OUT:-results/overnight_bunny_z012}"
SAVE_ESTIMATES="${SAVE_ESTIMATES:-0}"

mkdir -p "$OUT"

# Edit this to SEEDS=(0) if you want a shorter sweep.
SEEDS=(0 1 2)

COMMON_ARGS=(
  --mesh ply
  --mesh-path "$MESH"
  --boundary harmonic_zebra_k8
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

run_one () {
  local name="$1"
  shift
  local json="$OUT/${name}.json"

  if [[ -f "$json" ]]; then
    echo "[skip] $json"
    return
  fi

  local estimate_args=()
  if [[ "$SAVE_ESTIMATES" == "1" ]]; then
    estimate_args=(--save-estimates-prefix "$OUT/${name}")
  fi

  echo
  echo "[run] $json"
  "$EXE" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    "${estimate_args[@]}" \
    --output "$json"
}

for SEED in "${SEEDS[@]}"; do
  # --------------------------------------------------------------------------
  # A. exact analytic label: check whether spatial coverage is the bottleneck.
  #    depth_m=0; use NC-WoS RMSE only. 2LMC is not meaningful here.
  # --------------------------------------------------------------------------
  for N in 20000 50000 100000; do
    run_one "exact_medium_train${N}_seed${SEED}_m0" \
      --seed "$SEED" \
      --label-source exact_analytic \
      --train-points "$N" \
      --label-refreshes 1 \
      --walks-per-label-refresh 1 \
      --train-steps-per-refresh 5000 \
      --pure-walks-per-point 16 \
      --hybrid-walks-per-point 1 \
      --coarse-walks-per-point 16 \
      --residual-walks-per-point 8 \
      --depth-m 0
  done

  # --------------------------------------------------------------------------
  # B. depth sweep: test whether larger m actually buys residual variance.
  # --------------------------------------------------------------------------
  for M in 2 4 6 8; do
    run_one "wos_medium_train20000_steps500_seed${SEED}_m${M}_c64_r32" \
      --seed "$SEED" \
      --label-source wos_supervision \
      --train-points 20000 \
      --label-refreshes 4 \
      --walks-per-label-refresh 50 \
      --train-steps-per-refresh 500 \
      --pure-walks-per-point 64 \
      --hybrid-walks-per-point 4 \
      --coarse-walks-per-point 64 \
      --residual-walks-per-point 32 \
      --depth-m "$M"
  done

  # --------------------------------------------------------------------------
  # C. train-point coverage under noisy WoS supervision.
  #    Run only for m=4; if this improves NC/2LMC, coverage is a real bottleneck.
  # --------------------------------------------------------------------------
  for N in 50000 100000; do
    run_one "wos_medium_train${N}_steps500_seed${SEED}_m4_c64_r32" \
      --seed "$SEED" \
      --label-source wos_supervision \
      --train-points "$N" \
      --label-refreshes 4 \
      --walks-per-label-refresh 50 \
      --train-steps-per-refresh 500 \
      --pure-walks-per-point 64 \
      --hybrid-walks-per-point 4 \
      --coarse-walks-per-point 64 \
      --residual-walks-per-point 32 \
      --depth-m 4
  done

  # --------------------------------------------------------------------------
  # D. allocation sweep around m=4.
  #    This checks whether coarse/residual sampling is hiding a win.
  # --------------------------------------------------------------------------
  for CR in "32 32" "32 48" "64 48" "16 32"; do
    read -r C R <<< "$CR"
    run_one "wos_medium_train20000_steps500_seed${SEED}_m4_c${C}_r${R}" \
      --seed "$SEED" \
      --label-source wos_supervision \
      --train-points 20000 \
      --label-refreshes 4 \
      --walks-per-label-refresh 50 \
      --train-steps-per-refresh 500 \
      --pure-walks-per-point 64 \
      --hybrid-walks-per-point 4 \
      --coarse-walks-per-point "$C" \
      --residual-walks-per-point "$R" \
      --depth-m 4
  done
done

python scripts/collect_nc_results.py \
  --inputs "$OUT/*.json" \
  --out-csv "$OUT/summary.csv" \
  --out-json "$OUT/summary.json"

echo
echo "[done] wrote:"
echo "  $OUT/summary.csv"
echo "  $OUT/summary.json"

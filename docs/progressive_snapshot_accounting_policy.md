# Progressive snapshot accounting policy

The current `n2wos_eval_tcnn_nc_wos` executable is an offline fixed-snapshot
solver: it first creates a Neural Cache, freezes the final snapshot, and then
runs Pure WoS, NC-WoS, and NC+2LMC. Therefore, `runs.nc_wos.elapsed_ms` and
`runs.nc_2lmc.elapsed_ms` are solve-only times. Training is reported separately
as `training.total_training_ms`, and `training_plus_elapsed_ms` is the offline
sum.

`scripts/run_progressive_snapshot_accounting.py` is a first step toward the
online/progressive experiment. It measures several fixed snapshots and then
computes two accounting models:

- **offline serial:** train snapshot k and then evaluate snapshot k.
- **progressive overlap model:** evaluate snapshot k while training snapshot
  k+1, with block time approximated by `max(eval_time(k), train_time(k+1))`.

The runner deliberately does **not** reuse residual samples for training. Training
is assumed to use the global training-point distribution from the solver, while
NC+2LMC evaluates the chosen point or slice distribution.

This runner is not a final production online implementation. It does not reuse
model state between snapshots and does not launch the training and estimator in
two CUDA streams inside one process. Its purpose is to test whether the
progressive accounting story can plausibly hide the cache-training cost behind
2LMC estimation before adding a larger C++/CUDA online runner.

A future production implementation should use two cache instances:

1. `theta_infer`: a read-only snapshot used by estimator block k.
2. `theta_train`: a mutable model being trained toward snapshot k+1.

The snapshot discipline is:

1. Freeze `theta_infer = theta_k`.
2. Evaluate the NC+2LMC block using `theta_k`.
3. Train `theta_train` independently toward `theta_{k+1}`.
4. Swap snapshots only at a block boundary.

This preserves the estimator-level interpretation that each block uses a fixed
cache snapshot. If residual samples are later reused for training, they must be
added only after the estimator contribution has been recorded.

## Suggested command

```bash
python3 scripts/run_progressive_snapshot_accounting.py \
  --executable ./build/cuda-release-cubql-tcnn/n2wos_eval_tcnn_nc_wos \
  --output-dir results/progressive_snapshot_bumpy_nano_medium_m4 \
  --mesh procedural_bumpy_sphere \
  --boundary external_charges_medium \
  --label-source wos_supervision \
  --cache-preset nano \
  --train-points 20000 \
  --eval-mode slice \
  --slice-width 512 \
  --slice-height 512 \
  --slice-view xy \
  --slice-plane 0 \
  --snapshot-train-steps-per-refresh-list 0,50,100,250 \
  --depth-m 4 \
  --pure-walks-per-point 64 \
  --hybrid-walks-per-point 4 \
  --coarse-walks-per-point 64 \
  --residual-walks-per-point 32 \
  --cubql-build-method sah
```

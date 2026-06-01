# NC slice time-to-MSE policy

This note documents `scripts/run_nc_slice_time_to_mse.py`.

The purpose is to move away from a single fixed-wpp comparison. The runner
collects MSE-vs-time points for:

- low-budget pure WoS with multiple walks-per-pixel values,
- NC-only with multiple hybrid walks-per-pixel values,
- NC+2LMC with multiple coarse/residual allocations.

The default configuration intentionally uses:

```text
cache_preset = nano
train_points = 5000
boundary = external_charges_medium
depth_m = 4
slice = 512 x 512 xy plane
```

The `nano` cache and 5000 training points are not meant to reproduce the Neural
Cache paper setting. They are a low-capacity / low-supervision setting intended
to make the direct NC-only bias floor visible. The NC-paper-like setting should
remain a separate baseline with a larger cache and about 20k training points.

## Timing modes

The underlying executable reports the method evaluation time after a cache
snapshot has been trained. The runner writes three timing modes:

```text
solve_only_ms:
  method evaluation after a snapshot exists.

single_snapshot_total_ms:
  one representative training cost plus solve_only_ms. This approximates
  training a single cache snapshot and reusing it for the whole curve.

end_to_end_ms:
  training + solve time from each independent executable run.
```

The current runner does not implement true online/progressive snapshot training.
It also does not construct a high-sample pure-WoS numerical reference. It uses
the RMSE reported by `n2wos_eval_tcnn_nc_wos`.

## Interpretation

A useful outcome is not necessarily that every NC+2LMC point beats pure WoS.
The expected curve shape is:

```text
NC-only:
  low solve time, but visible MSE floor when the cache is nano/under-trained.

Pure WoS:
  unbiased reference curve, but MSE decreases only by more walks.

NC+2LMC:
  slower than NC-only, but should lower the NC-only floor. The operating point
  of interest is an intermediate threshold below the NC-only floor where
  NC+2LMC may reach the threshold faster than pure WoS.
```

For final wall-clock claims, this script is only an orchestration/accounting
step. A later implementation should run true progressive snapshot blocks and, if
needed, overlap training and estimation in a single process.

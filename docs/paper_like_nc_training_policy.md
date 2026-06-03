# Paper-like Neural Cache training policy

Use this policy for Neural Cache experiments that are intended to match the
Neural Cache paper rather than for quick debugging.

## Training point distribution

The default training point sampler is now `--train-sampler rejection`.
It draws proposal points uniformly in the normalized mesh bounding box and keeps
only points classified as inside the closed mesh by a ray-parity inside test.
This replaces the earlier center-ball sampler, which is retained only as
`--train-sampler ball` for debugging.

The executable still pads the training set to tiny-cuda-nn's batch-size
granularity.  The padded points are random repeats of accepted interior points;
JSON reports both the requested unique point count and the padded count.

Use `--save-train-points-prefix <prefix>` to write:

```text
<prefix>_train_points.csv
```

The CSV marks padding rows and their source training point.

## Default paper-like schedule

The executable defaults are:

```text
--train-points 20000
--train-sampler rejection
--walks-per-label-refresh 50
--label-refreshes 4
--train-steps-per-refresh 5000
```

This corresponds to an initial 50-WoS-walk label batch followed by 20,000 Adam
steps in four 5,000-step blocks.  At the start of each block, another batch of
50 WoS walks per training point is accumulated into the running-average labels.

The older `ball` sampler and short schedules are only for smoke tests and must
not be used for paper-like claims.

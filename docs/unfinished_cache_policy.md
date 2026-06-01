# Unfinished neural cache diagnostics

This diagnostic isolates one estimator-level claim: a neural cache need not be
fully converged before it can be used in a two-level estimator, provided the
cache snapshot used by the estimator is fixed.

For a fixed snapshot theta and the same prefix depth m, the biased Neural Cache
WoS estimator is

```text
NC-WoS(m): x -> m WoS steps -> X_m -> C_theta(X_m).
```

The two-level diagnostic uses

```text
NC+2LMC(m): E[C_theta(X_m)] + E[W(X_m) - C_theta(X_m)].
```

The intended first check is not that every unfinished cache is faster than pure
WoS. The narrower claim is:

```text
NC-only bias can be large for an unfinished cache.
For the same snapshot and same m, NC+2LMC should reduce mean bias.
Cache quality should primarily affect residual variance rather than estimator
bias.
```

The main diagnostic keeps `m = 1` fixed. If residual variance remains too high or
the result is otherwise inconclusive, increase m only after this fixed-m
unfinished-cache sweep has been examined.

The current script launches independent offline runs at different training step
counts. It is not yet the final online snapshot pipeline. A later online version
should use this order:

```text
1. freeze theta_k
2. evaluate the current estimator block using theta_k
3. add that block to the estimator
4. only then add full-WoS residual samples to the training/replay buffer
5. update to theta_{k+1}
```

Residual-derived samples can be useful training data, but they are biased toward
the evaluation distribution. They should be mixed with global training points
rather than replacing the global cache-training distribution.

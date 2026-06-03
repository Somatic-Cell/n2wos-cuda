# Implementation integrity policy

Do not replace a requested experimental condition with an easier implementation
unless the deviation is explicit in the command-line option name, JSON output,
documentation, and user-facing description.

A debug scaffold is acceptable only when it is clearly labeled as such and cannot
be mistaken for the paper-like path.  Silent substitutions can invalidate long
experiment chains: they may make results appear to support or refute an idea
while actually measuring a different algorithm or data distribution.

For Neural Cache experiments, the paper-like training path is:

- interior training points from domain-wide rejection sampling;
- 20,000 requested unique training points;
- 50 WoS walks per label refresh;
- 20,000 Adam iterations in four 5,000-iteration blocks;
- accumulated running-average labels over refreshes.

The center-ball sampler remains only as `--train-sampler ball` for smoke tests.
It must not be used for paper-like claims.

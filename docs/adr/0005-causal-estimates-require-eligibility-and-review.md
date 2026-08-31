# Causal estimates require eligibility and review

> Superseded by [ADR-0006](0006-retire-causal-analysis.md): the causal-estimate
> workflow described here is retired.

The agent returns causal estimates only after an eligible design passes its
diagnostics and receives the required review. Registered randomized experiments
may use a pre-approved estimator; observational and quasi-experimental methods
produce a reviewed analysis plan first, and all-user pre/post comparisons stay
descriptive.

# Opportunity Sizing is governed, never inferred

An Opportunity Estimate is available only for a Candidate Causal Factor whose
Factor Vocabulary category has an explicit, reviewed dbt/MetricFlow
event-and-audience mapping — a small, hardcoded table, not a live taxonomy.
In this delivery only `provisioning_or_entitlement` is mapped, and only for
the New Paid Enabled User driver metrics (`jira_new_peu`,
`confluence_new_peu`). Every other category remains a Hypothesis and is
never Sizing Eligible.

The Eligible Population behind an estimate is always a fresh governed
dbt/MetricFlow count — Product Users entitled to the product who have not
already qualified through Paid Enablement for it — computed at sizing time
via a new `ValidatedMetricFlowGateway.eligible_population(...)` call that
re-authorizes the Agent User's Access Profile independently, the same
discipline as every other semantic-layer query (ADR-0003). It is never
approximated from the factor's cited documents or the evidence graph: those
sources can name a documented change, but only the semantic layer may name
the population it affects.

The formula is exactly `eligible_population × scenario_percentage_point_change
÷ 100`, with the analyst's percentage-point assumption always echoed back
alongside the current baseline rate — an Opportunity Estimate is a
conditional projection, never a causal effect, forecast, or observed uplift.

A factor with no governed mapping is never silently sized against a
document/graph-derived approximation. It remains a Hypothesis and the
response offers a data-team mapping request instead — extending ADR-0007's
"Provisional Factor Record is never a source or permission authority"
principle to the audience side of a sizing scenario.

## Sizing trigger is asymmetric by design

An explicit `opportunity_scenario_percentage_points` value is the only signal
that a turn is attempting sizing. A turn that selects or reasserts a factor
without one is not treated as a sizing attempt at all — it is ADR-0009's
Active Investigation reference confirmation, a complete and valid action on
its own that must keep returning `HYPOTHESIS`, not `LIMITATION`. Only the
reverse gap — a scenario supplied with no resolved selection — produces the
explicit `LIMITATION` "missing input" response. A confirmed single selection
that is Sizing Eligible still gets a caveat naming the scenario field it can
supply, so a forgotten scenario is surfaced as guidance rather than either a
silent no-op or a spurious error on an otherwise-ordinary selection turn.

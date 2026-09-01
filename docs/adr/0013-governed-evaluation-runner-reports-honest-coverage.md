# Governed evaluation runner reports honest coverage, not full-dataset execution

The runner (`src/growth_data_agent/evaluation_runner.py`) only executes a
Governed Evaluation Case when none of its turns declares a `setup_note` —
the free-text field from ADR-0012 documenting bespoke harness state (a
custom evidence store, a monkeypatched Access Profile between turns, an
injected local model) that isn't yet machine-actionable. Every other case is
counted as `not_yet_automated_cases` on the scorecard: a real, visible
status, never a silent skip and never scored as passing. Building generic
harness support for those cases is a separate, comparably-sized follow-up;
it would first need `setup_note` restructured into something the runner can
act on rather than a human-readable description.

Evaluators check invariants on fields the governed response already
computes — `driver_decomposition.residual == 0`, not a re-derived sum of
baseline/comparison values — so the evaluator and the service can never
silently disagree about the same formula. Two evaluator assumptions were
corrected after running against the real dataset and finding false
positives, not real bugs: (1) an individual segment's `percentage_of_decline`
can legitimately exceed 100% when other segments move in the opposite
direction (residual staying at 0 is the actual reconciliation guarantee, not
contributions summing to ~100%); (2) `source_freshness.is_current=False` is
a placeholder, not a staleness signal, on `safe_refusal` responses that
never ran a semantic check at all (see
`AnswerQuestionService._answer_direct_identifier_request`'s unentitled-profile
branch) — the "stale artifact blocks canonical response" invariant is scoped
to canonical/driver-decomposition paths, not universal.

The Evaluation Scorecard is a separate, non-composite MLflow record
(`MlflowTraceSink.record_scorecard`) — distinct from the existing
`record_evaluation`, which still links one turn's pass/fail to its own
`trace_id`. Token/cost is reported as zero with an explicit note: this
runner only exercises the deterministic path, which never calls the
optional local-model adapters, so there is no usage to report.

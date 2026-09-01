# Governed Evaluation Dataset is versioned and never runtime evidence

The Governed Evaluation Dataset (`evaluations/dataset/v1/cases.json`,
`src/growth_data_agent/evaluation_dataset.py`) is a separate, offline artifact
for measuring the system, never a source the system reads to answer a
request. No file under `src/growth_data_agent/` other than
`evaluation_dataset.py` itself may import it — enforced by a static AST check
in `tests/test_evaluation_dataset.py`, mirroring the "never" boundary already
proven for MLflow trace content (ADR-0011) but for code coupling rather than
data leakage, since this ticket adds no service-layer integration to leak
content through.

Each Evaluation Case's expected behaviour and reviewer labels are grounded in
an already-passing test, named in `provenance.source_reference`. This first
version's reviewer labels are gold/expected labels — what a compliant
response *is* for a scenario already proven correct — not judgements of a
live evaluation run. No evaluator has executed against this dataset yet;
running cases and publishing a scorecard is a later child of the epic
(#85+). Fabricating "imperfect" scores to look like a real review would be
dishonest and would give #88's later calibration work a false starting
point.

A published version directory (`v1/`) is never edited in place. A dataset
content change is a new version: a new `v{n}/` directory, a bumped
`dataset_version`, and a fresh `published_at`. `dataset_version` and the
version path segment are cross-checked by a test, so a version bump without
a matching directory (or vice versa) fails before it can ship. Git history
remaining an accurate record of what shipped when is a process invariant
this ADR states rather than something a single checkout can mechanically
enforce.

Many cases describe scenarios that need more than a literal request replay
against the default test fixtures — a bespoke evidence store, a
monkeypatched Access Profile between turns, an injected local model. Each
such turn carries a `setup_note` naming the exact harness state needed and
the test that proves it, generalizing the existing
`evaluations/fixtures.json` `"requires": "stale_artifact"` convention.
Building that harness to actually execute these cases is out of scope here.

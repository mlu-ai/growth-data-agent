# Candidate Causal Factor is a derived projection

A Candidate Causal Factor card is computed synchronously per request from the
currently authorized Evidence Revision set, using deterministic rule-based
extraction and a separate deterministic ranking-eligibility validator. No
Provisional Factor Record is persisted across requests or turns.

Extraction only runs on documents already returned by the bounded evidence
tools, after both the store-side and context-boundary evidence authorization
checks (ADR-0003). A revised, deleted, superseded, or inaccessible source is
therefore excluded from the document set extraction operates on, so a
Candidate Causal Factor citing it cannot recur in the next answer without a
dedicated invalidation mechanism.

The initial Factor Vocabulary is a small hardcoded list, matching the
existing pattern for other reviewed canonical lists in this codebase. It and
Canonical Metric qualifying prerequisites guide evidence-retrieval query
construction only — never metric or causal authority.

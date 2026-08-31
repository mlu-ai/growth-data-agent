# Active Investigation is a revalidated reference, never persisted content

A selected Candidate Causal Factor is carried forward across turns only as
its `factor_id` string — an opaque lookup key — never as the card's content,
citations, or evidence. This extends ADR-0007's "no Provisional Factor
Record persisted across requests or turns" principle to the cross-turn
selection case explicitly.

Every turn re-authorizes the Agent User and re-runs the full ranking
pipeline from scratch (ADR-0003, ADR-0007); the stored reference is used
only to filter the freshly computed candidate list down to one match. If no
current candidate's `factor_id` matches — because the source evidence was
revoked, the Agent User's entitlements narrowed, the candidate is no longer
ranking-eligible, or it has since become contradicted — the response is an
explicit limitation with an empty candidate list, never a silent
substitution of a different candidate.

# Widen candidate evidence retrieval to top three

The bounded evidence-investigation tool anchors its authorized candidate
document set to LightRAG's top-ranked *chunk* references, up to three, not
one. This is a deliberate widening of a prior hardening decision — earlier
code anchored to exactly the single top-ranked reference — made to let a
Hypothesis Investigation return more than one ranked Candidate Causal Factor
card. It sources strictly from LightRAG's chunk-kind records, never the
interleaved chunk/entity/relation reference list, so graph-context entities
and relations can never be treated as additional authorized documents.

Nothing else changes: a document outside the authorized revision set still
fails closed, and the required cross-encoder reranker still only reorders,
never adds to, the authorized candidate set.

# RAG evaluation separates retrieval from generation

The RAG Evaluation Dataset (`evaluations/rag_dataset/v1/cases.json`,
`src/growth_data_agent/rag_evaluation_dataset.py`) is a separate, small
artifact from the #84 Governed Evaluation Dataset: a RAG Evaluation Case is
a retrieval query plus gold-relevant Evidence Revisions, not a request/
response turn, and doesn't fit #84's per-category split-validation contract.

Gold relevance is keyed by `(source_document_id, source_revision)` — this
codebase's existing Evidence Revision identity (`lightrag.py`'s
`_revision_key`, `evidence_sync.py`'s `ConfluenceEvidenceRevision`), never by
`chunk_id` alone. A gold label may record `chunk_id` as optional retrieval
detail, but it is never the correctness key.

Retrieval quality (Recall@K, Precision@K, MRR, nDCG@K —
`src/growth_data_agent/rag_evaluation.py::evaluate_rag_retrieval`) and
generation quality (RAGAS's context relevance, faithfulness, and answer
relevance) are two independent `ScorecardCategory` fields on
`RagEvaluationScorecard`, never merged into one score — a retrieval
regression and a generation regression are always distinguishable signals.
Retrieval always goes through the same authorized
`evidence_store.retrieve(...)` seam production and `scripts/run_evaluations.py`
already use — never a bypassed or unauthenticated query — so entitlement and
runtime evidence rules are never altered by evaluating them.

## The RAGAS judge is real, optional, and never fabricates a result

`RagJudge.from_environment()` mirrors `OllamaLocalModel.from_environment()`
exactly: it returns `None` when `RAGAS_JUDGE_MODEL_NAME` isn't set, and
`evaluate_rag_generation` then honestly records every case as
`not_configured` — not scored, not silently passing. This matches #61's own
principle that "LLM-as-a-judge" is used "only after calibration against
held-out human labels" (issue #88, not yet done); this ticket makes the real
integration point exist without pretending it's a blocking gate yet. When a
judge *is* configured and its transport fails, that raises
`RagJudgeUnavailable` and is recorded as `"unavailable"` — a real failure,
distinct from "not configured" — never a fabricated score either way. The
judge talks to a local Ollama server through its OpenAI-compatible endpoint
via `ragas.llms.llm_factory(..., provider="openai", client=AsyncOpenAI(...))`,
so no real OpenAI API call or key is ever involved. The real-judge path is
unverified end-to-end in this environment (no Ollama server here) — the same
disclosed caveat every other Ollama-dependent script in this repo already
carries.

**Dependency note**: `ragas` (any current version) imports
`langchain_community.chat_models.vertexai` unconditionally at import time,
which no longer exists in current `langchain-community` releases (that
integration was split into a standalone package and removed). `pyproject.toml`
pins `langchain-community==0.3.30` specifically to keep this import path
working; bumping `ragas` or removing that pin without checking this import
still resolves will break at import time, not at call time — a `uv run
python -c "import ragas"` check is the fast way to catch it.

## Chunking-version comparability is an honest placeholder

`CHUNKING_STRATEGY_VERSION = "fixed-chunk-v1"` records that there is no
automated chunking pipeline in this codebase — every Evidence Revision is
already split into explicit, hand-assigned chunks. This is a single
placeholder string, not a fabricated versioning scheme, existing so results
stay comparable once a real chunking configuration is introduced. Embedding
and reranker versions (`EmbeddingProvider`/`EvidenceReranker`'s existing
`model_name`/`model_version` fields) are genuinely first-class and are
recorded on the scorecard as-is.

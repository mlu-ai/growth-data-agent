# Governed Confluence-to-Qdrant Evidence Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator backfill and incrementally synchronize governed Confluence Evidence Revisions into external Qdrant with durable provenance, access metadata, lifecycle state, and embedding metadata.

**Architecture:** A strict normalized Confluence revision contract separates source normalization from indexing. A Qdrant synchronizer owns page-level replacement: it skips an unchanged revision, removes the prior revision before writing an updated active revision, and stores deleted or inaccessible revisions only as non-retrievable tombstones. Qdrant remains the external vector boundary for synchronization; the deterministic embedding provider is explicit metadata-bearing POC infrastructure, and synthetic data uses the same normalized source contract as a live-shaped adapter.

**Tech Stack:** Python 3.11+, Pydantic, LlamaIndex QdrantVectorStore, external `qdrant-client`, pytest, FastAPI readiness endpoint, `uv` and Ruff.

**Spec:** GitHub issue #56, “Governed Confluence-to-Qdrant evidence sync”.

## Global Constraints

- The synchronization path must index source URL, page ID, revision, chunk provenance, lifecycle state, source access metadata, and embedding model/version metadata.
- An unchanged revision is idempotent; an updated revision replaces retrievable evidence; a deleted or inaccessible revision becomes non-retrievable.
- The initial synthetic Confluence corpus must be backfilled through the same normalized contract used by the live-shaped source adapter.
- Missing required source provenance or access metadata prevents indexing, and Qdrant/embedding readiness is observable.
- Keep external Qdrant as the production path; in-memory Qdrant clients are test doubles/local fixtures only.
- Do not implement answering/retrieval (#57), LightRAG (#58), or later tickets.

---

### Task 1: Define the normalized Confluence revision seam and validation

**Files:**
- Create: `src/growth_data_agent/evidence_sync.py`
- Modify: `src/growth_data_agent/evidence.py` to share lifecycle and persisted metadata vocabulary where needed
- Test: `tests/test_evidence_sync.py`

**Interfaces:**
- Produces `EvidenceLifecycleState`, `SourceAccessMetadata`, `ConfluenceEvidenceChunk`, `ConfluenceEvidenceRevision`, `ConfluenceEvidenceSource`, `EmbeddingProvider`, and validation errors for later tasks.
- The source contract exposes `iter_revisions() -> Iterable[ConfluenceEvidenceRevision]`; active revisions require non-empty page ID, URL, revision, chunks, source access metadata, and embedding metadata.

- [ ] **Step 1: Write the failing tests** for accepted active revisions, rejection of missing URL/page ID/revision/chunk provenance/access metadata/embedding metadata, and deleted or inaccessible revisions being valid tombstones without content.
- [ ] **Step 2: Run the focused tests** with `UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest tests/test_evidence_sync.py -v`; confirm failures are caused by missing contracts or validation.
- [ ] **Step 3: Implement the minimal Pydantic models and protocols** with explicit lifecycle values and no fallback provenance or access defaults on the normalized contract.
- [ ] **Step 4: Run the focused tests again** and keep the implementation limited to the tested validation boundary.

### Task 2: Implement Qdrant revision synchronization and durable metadata

**Files:**
- Modify: `src/growth_data_agent/evidence_sync.py`
- Modify: `src/growth_data_agent/evidence.py`
- Create: `tests/test_evidence_sync_qdrant.py`

**Interfaces:**
- Consumes `ConfluenceEvidenceSource`, `ConfluenceEvidenceRevision`, and `EmbeddingProvider` from Task 1.
- Produces `QdrantEvidenceSynchronizer.sync()` returning counts for indexed, skipped, and removed revisions; `QdrantEvidenceSynchronizer.readiness()` reports Qdrant/embedding status without silently synthesizing provenance.

- [ ] **Step 1: Write the failing tests** for first backfill, unchanged-revision idempotency, updated-revision replacement with stale chunks removed, deleted/inaccessible non-retrievability, and persisted payload metadata.
- [ ] **Step 2: Run the focused tests** and confirm the expected red failures against the current store.
- [ ] **Step 3: Implement page-scoped Qdrant replacement** using stable point IDs, payload filters, explicit active lifecycle filtering, and the supplied embedding provider; use an in-memory Qdrant client only in tests.
- [ ] **Step 4: Run the focused tests** and verify the red-green cycles for each behavior.
- [ ] **Step 5: Refactor only after green** to share node/payload construction with existing LlamaIndex retrieval behavior without adding #57 retrieval features.

### Task 3: Wire synthetic/live-shaped sources, runtime configuration, and readiness

**Files:**
- Modify: `src/growth_data_agent/synthetic.py`
- Modify: `src/growth_data_agent/main.py`
- Modify: `src/growth_data_agent/service.py`
- Create: `scripts/sync_confluence_evidence.py`
- Create: `tests/test_confluence_evidence_sync_runtime.py`

**Interfaces:**
- Consumes the normalized source and synchronizer from Tasks 1–2.
- Produces `SyntheticConfluenceEvidenceSource`, external-Qdrant environment construction, synchronization CLI behavior, and readiness fields for Qdrant and embedding dependencies.

- [ ] **Step 1: Write the failing tests** for synthetic corpus normalization, external `QDRANT_URL` construction, missing production configuration, and readiness reporting without exposing credentials.
- [ ] **Step 2: Run the focused tests** and confirm the new runtime behavior is absent.
- [ ] **Step 3: Implement the synthetic adapter, environment factories, readiness aggregation, and operator script**; keep the script on the same normalized contract and require external Qdrant for operator sync.
- [ ] **Step 4: Run focused runtime tests** and verify readiness and configuration failures are explicit.

### Task 4: Verify the issue boundary and handoff

**Files:**
- Modify: only files required by focused verification findings.

- [ ] **Step 1: Run focused sync tests, Ruff, and Python bytecode compilation; this repository has no configured static type checker.**
- [ ] **Step 2: Run the full pytest suite and inspect the complete diff against `origin/main`.**
- [ ] **Step 3: Run the standards/spec code review against `origin/main`; fix Critical and Important findings and rerun affected checks.**
- [ ] **Step 4: The user delegation explicitly authorizes repository delivery as an exception to the repository's default safety guidance: commit, push the issue #56 branch, open a PR closing #56, and wait for all PR checks without merging.**

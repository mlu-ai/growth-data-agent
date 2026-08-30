from __future__ import annotations

import pytest

from growth_data_agent.evidence import QdrantEvidenceStore
from growth_data_agent.evidence_sync import (
    ConfluenceEvidenceSourceAdapter,
    EvidenceLifecycleState,
    EvidenceRevisionValidationError,
    HashEmbeddingProvider,
    QdrantConfigurationError,
    QdrantEvidenceSynchronizer,
    qdrant_client_from_environment,
    qdrant_collection_from_environment,
)
from growth_data_agent.synthetic import SyntheticConfluenceEvidenceSource, evidence_corpus


def test_synthetic_corpus_uses_the_normalized_live_source_contract() -> None:
    revisions = list(SyntheticConfluenceEvidenceSource(evidence_corpus()).iter_revisions())

    assert len(revisions) == 8
    assert {revision.product for revision in revisions} == {"Confluence"}
    assert all(revision.source_page_id for revision in revisions)
    assert all(revision.source_url.startswith("https://") for revision in revisions)
    assert all(revision.source_revision == "synthetic-v1" for revision in revisions)
    assert all(revision.lifecycle_state is EvidenceLifecycleState.ACTIVE for revision in revisions)
    assert all(revision.source_access.classification == "internal" for revision in revisions[:3])


def test_synthetic_adapter_rejects_a_fixture_without_explicit_provenance() -> None:
    incomplete_document = next(
        document for document in evidence_corpus() if document.product == "Confluence"
    ).model_copy(update={"source_url": None})

    with pytest.raises(EvidenceRevisionValidationError, match="missing provenance"):
        list(SyntheticConfluenceEvidenceSource([incomplete_document]).iter_revisions())


def test_synthetic_adapter_rejects_a_fixture_without_explicit_embedding_metadata() -> None:
    incomplete_document = next(
        document for document in evidence_corpus() if document.product == "Confluence"
    ).model_copy(update={"embedding_model": None, "embedding_version": None})

    with pytest.raises(EvidenceRevisionValidationError, match="embedding_model"):
        list(SyntheticConfluenceEvidenceSource([incomplete_document]).iter_revisions())


def test_synthetic_adapter_rejects_a_fixture_without_explicit_chunk_provenance() -> None:
    incomplete_document = next(
        document for document in evidence_corpus() if document.product == "Confluence"
    ).model_copy(update={"chunk_id": None})

    with pytest.raises(EvidenceRevisionValidationError, match="chunk_id"):
        list(SyntheticConfluenceEvidenceSource([incomplete_document]).iter_revisions())


def test_live_shaped_adapter_validates_raw_source_records() -> None:
    adapter = ConfluenceEvidenceSourceAdapter(
        lambda: [
            {
                "source_page_id": "page-123",
                "source_url": "https://confluence.example/pages/page-123",
                "source_revision": "42",
                "title": "Onboarding regression",
                "product": "Confluence",
                "region": "EMEA",
                "tenant_ids": ["tenant-0001"],
                "tenant_scope": "EMEA 51-200 Seat Tier Tenants",
                "relevant_date": "2026-06-20",
                "freshness": "2026-06-21T00:00:00Z",
                "support_status": "supports",
                "support_explanation": "The source overlaps the affected period.",
                "chunks": [
                    {
                        "chunk_id": "page-123:chunk:0",
                        "chunk_index": 0,
                        "text": "The finding.",
                    }
                ],
                "source_access": {
                    "classification": "internal",
                    "identifier_entitlement": "none",
                    "policy_expires_at": "2099-12-31T00:00:00Z",
                },
                "embedding_model": "deterministic-hash",
                "embedding_version": "1",
            }
        ]
    )

    revision = next(adapter.iter_revisions())

    assert revision.source_page_id == "page-123"
    assert revision.chunks[0].text == "The finding."


def test_external_qdrant_configuration_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)

    with pytest.raises(QdrantConfigurationError, match="QDRANT_URL"):
        qdrant_client_from_environment()


def test_external_qdrant_configuration_is_observable_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "must-not-appear-in-readiness")
    monkeypatch.setenv("QDRANT_COLLECTION", "governed-evidence")

    client = qdrant_client_from_environment()
    monkeypatch.setattr(client, "get_collections", lambda: object())
    status = QdrantEvidenceSynchronizer(
        client=client,
        collection_name=qdrant_collection_from_environment(),
    ).readiness()

    assert status["status"] == "ready"
    assert status["qdrant"]["collection"] == "governed-evidence"
    assert "must-not-appear" not in str(status)
    assert status["embedding"] == {
        "status": "ready",
        "model": "deterministic-hash",
        "version": "1",
    }


def test_hash_embedding_readiness_is_explicit() -> None:
    assert HashEmbeddingProvider().readiness() == {
        "status": "ready",
        "model": "deterministic-hash",
        "version": "1",
    }


def test_synthetic_confluence_corpus_backfills_through_the_sync_contract() -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    result = QdrantEvidenceSynchronizer(
        client=client,
        collection_name="synthetic-sync",
    ).sync(SyntheticConfluenceEvidenceSource(evidence_corpus()))

    assert result.indexed_revision_count == 8
    assert client.count("synthetic-sync").count == 8


def test_external_qdrant_unavailability_is_observable_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    client = qdrant_client_from_environment()

    monkeypatch.setattr(
        client,
        "get_collections",
        lambda: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )
    status = QdrantEvidenceSynchronizer(client=client).readiness()

    assert status["status"] == "unavailable"
    assert status["embedding"] == {
        "status": "ready",
        "model": "deterministic-hash",
        "version": "1",
    }


def test_configured_embedding_failure_is_observable() -> None:
    from qdrant_client import QdrantClient

    class UnavailableEmbeddingProvider:
        model_name = "configured-embedder"
        model_version = "2"

        def embed(self, text: str) -> list[float]:
            raise RuntimeError("embedding unavailable")

        def readiness(self) -> dict[str, object]:
            return {
                "status": "unavailable",
                "model": self.model_name,
                "version": self.model_version,
            }

    status = QdrantEvidenceSynchronizer(
        client=QdrantClient(location=":memory:"),
        embedding_provider=UnavailableEmbeddingProvider(),
    ).readiness()

    assert status["status"] == "unavailable"
    assert status["embedding"]["status"] == "unavailable"


def test_service_readiness_exposes_qdrant_and_embedding_dependencies(tmp_path) -> None:
    from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
    from qdrant_client import QdrantClient

    from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
    from growth_data_agent.service import AnswerQuestionService

    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
    )
    evidence_store = QdrantEvidenceStore(
        (), client=QdrantClient(location=":memory:"), collection_name="readiness-test"
    )

    status = AnswerQuestionService(gateway, evidence_store=evidence_store).readiness()

    assert status["qdrant"]["status"] == "ready"
    assert status["embedding"] == {
        "status": "ready",
        "model": "deterministic-hash",
        "version": "1",
    }

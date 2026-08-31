from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    EvidenceLifecycleState,
)
from growth_data_agent.lightrag import (
    InMemoryLightRAGStore,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
)
from growth_data_agent.main import create_app
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.synthetic import evidence_corpus

_APAC_QUESTION = "What evidence may explain the APAC 51–200-seat Tenant decline?"


class _MutableEvidenceStore:
    """A store whose held documents can change between requests, like a live corpus."""

    def __init__(self, documents: list[EvidenceDocument]):
        self.documents = documents

    def retrieve_scoped(
        self,
        query: str,
        access_filter: EvidenceAccessFilter,
        authorized_document_ids,
        *,
        limit: int,
        authorized_revision_keys=(),
    ) -> list[EvidenceDocument]:
        del query, authorized_document_ids, authorized_revision_keys
        return [document for document in self.documents if access_filter.allows(document)][:limit]

    def retrieve(
        self, query: str, access_filter: EvidenceAccessFilter, *, limit: int
    ) -> list[EvidenceDocument]:
        del query
        return [document for document in self.documents if access_filter.allows(document)][:limit]


def _client_for(tmp_path: Path, store: _MutableEvidenceStore) -> TestClient:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    lightrag_adapter = LightRAGEvidenceAdapter(
        LightRAGBackend(
            InMemoryLightRAGStore(
                chunks=[
                    LightRAGChunkRecord(
                        reference=LightRAGEvidenceReference.from_document(document),
                        text=document.text,
                    )
                    for document in store.documents
                ]
            )
        )
    )
    return TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                evidence_store=store,
                evidence_reranker=DeterministicCrossEncoderReranker(),
                lightrag_adapter=lightrag_adapter,
            )
        )
    )


def test_inaccessible_source_revision_stops_supporting_the_next_answer(tmp_path: Path) -> None:
    documents = list(evidence_corpus())
    store = _MutableEvidenceStore(documents)
    client = _client_for(tmp_path, store)

    first = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    assert first.status_code == 200
    first_factor = first.json()["candidate_causal_factor"]
    assert first_factor is not None
    assert first_factor["citation"]["source_document_id"] == "jira-apac-paid-provisioning-incident"

    store.documents = [
        document.model_copy(update={"lifecycle_state": EvidenceLifecycleState.INACCESSIBLE})
        if document.document_id == "jira-apac-paid-provisioning-incident"
        else document
        for document in store.documents
    ]

    second = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["candidate_causal_factor"] is None
    assert "jira-apac-paid-provisioning-incident" not in second.text


def test_occurrence_time_outside_window_yields_no_rank_eligible_card(tmp_path: Path) -> None:
    documents = [
        document.model_copy(update={"relevant_date": document.relevant_date.replace(month=3)})
        if document.document_id == "jira-apac-paid-provisioning-incident"
        else document
        for document in evidence_corpus()
    ]
    store = _MutableEvidenceStore(documents)
    client = _client_for(tmp_path, store)

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_causal_factor"] is None
    assert body["result_classification"] == "inconclusive"
    assert "occurrence_time_exceeds_initial_lookback" in body["answer"]

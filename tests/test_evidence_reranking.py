from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from growth_data_agent.contracts import EvidenceSupportStatus
from growth_data_agent.evidence import (
    EvidenceDocument,
    EvidenceLifecycleState,
    QdrantEvidenceStore,
    _evidence_revision_key,
)
from growth_data_agent.evidence_sync import (
    ConfluenceEvidenceChunk,
    ConfluenceEvidenceRevision,
    QdrantEvidenceSynchronizer,
    SourceAccessMetadata,
)
from growth_data_agent.lightrag import (
    InMemoryLightRAGStore,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
)
from growth_data_agent.main import create_app
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.reranking import (
    RERANKER_MODEL_NAME,
    OllamaCrossEncoderReranker,
)
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.synthetic import evidence_corpus


class RecordingEvidenceStore:
    def __init__(self, documents: list[EvidenceDocument]) -> None:
        self.documents = documents
        self.filters = []

    def retrieve(self, query, access_filter, *, limit):
        self.filters.append(access_filter)
        # A real Qdrant query applies the policy filter before its result limit;
        # return the complete query result here so the service-side defensive
        # filter is exercised against every candidate.
        return self.documents

    def retrieve_scoped(
        self,
        query,
        access_filter,
        authorized_document_ids,
        *,
        limit,
        authorized_revision_keys,
    ):
        del query, authorized_document_ids
        self.filters.append(access_filter)
        return [
            document
            for document in self.documents
            if access_filter.allows(document)
            and _evidence_revision_key(document) in authorized_revision_keys
        ][:limit]


class RevokingEvidenceStore(RecordingEvidenceStore):
    """Revoke the validated source scope after cited retrieval completes."""

    def __init__(self, documents: list[EvidenceDocument]) -> None:
        super().__init__(documents)
        self.revoked = False

    def authorized_revisions(self, access_filter):
        if self.revoked:
            return []
        return [
            document.model_copy(deep=True)
            for document in self.documents
            if access_filter.allows(document)
        ]

    def retrieve_scoped(
        self,
        query,
        access_filter,
        authorized_document_ids,
        *,
        limit,
        authorized_revision_keys,
    ):
        result = super().retrieve_scoped(
            query,
            access_filter,
            authorized_document_ids,
            limit=limit,
            authorized_revision_keys=authorized_revision_keys,
        )
        self.revoked = True
        return result


class RecordingGraphStore:
    def __init__(self) -> None:
        self.calls = 0

    def traverse(self, query, access_filter, *, limit):
        del query, access_filter, limit
        self.calls += 1
        return []


class RecordingReranker:
    model_name = "test-cross-encoder"
    model_version = "1"

    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, candidates, *, limit):
        self.calls.append((query, list(candidates), limit))
        return list(reversed(candidates))[:limit]

    def readiness(self):
        return {
            "provider": "test",
            "status": "ready",
            "model": self.model_name,
            "version": self.model_version,
        }


class InjectingReranker(RecordingReranker):
    def __init__(self, injected: EvidenceDocument) -> None:
        super().__init__()
        self.injected = injected

    def rerank(self, query, candidates, *, limit):
        self.calls.append((query, list(candidates), limit))
        return [*candidates, self.injected][:limit]


class FailingReranker(RecordingReranker):
    def rerank(self, query, candidates, *, limit):
        raise RuntimeError("cross-encoder is unavailable")


class StaticRevisionSource:
    def __init__(self, revisions: list[ConfluenceEvidenceRevision]) -> None:
        self.revisions = revisions

    def iter_revisions(self):
        return iter(self.revisions)


class EmptyDriverRowsExecutor(RecordingPostgresExecutor):
    def execute_rows(self, plan):
        self.plans.append(plan)
        return []


def _client(
    tmp_path: Path,
    evidence_store: RecordingEvidenceStore,
    reranker=None,
    graph_store=None,
) -> TestClient:
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
                    for document in evidence_store.documents
                ]
            )
        )
    )
    return TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                evidence_store=evidence_store,
                evidence_reranker=reranker,
                graph_store=graph_store,
                lightrag_adapter=lightrag_adapter,
            )
        )
    )


def _evidence_question() -> str:
    return "What evidence may explain the APAC 51–200-seat Tenant decline?"


def test_only_active_driver_candidates_reach_the_cross_encoder_and_response(
    tmp_path: Path,
) -> None:
    supporting = evidence_corpus()[0]
    second_supporting = supporting.model_copy(
        update={
            "document_id": "jira-apac-paid-provisioning-incident-follow-up",
            "source_document_id": "jira-apac-paid-provisioning-incident-follow-up",
            "source_url": "https://evidence.example/jira/apac-follow-up",
            "source_revision": "43",
            "chunk_id": "jira-apac-paid-provisioning-incident-follow-up:chunk:0",
        }
    )
    wrong_segment = evidence_corpus()[1]
    restricted = evidence_corpus()[3]
    expired = supporting.model_copy(
        update={
            "document_id": "expired-driver-evidence",
            "policy_expires_at": datetime(2026, 8, 24, tzinfo=UTC),
        }
    )
    deleted = supporting.model_copy(
        update={
            "document_id": "deleted-driver-evidence",
            "lifecycle_state": EvidenceLifecycleState.DELETED,
        }
    )
    wrong_metric = supporting.model_copy(
        update={"document_id": "wrong-metric-evidence", "metric_name": "jira_new_mau"}
    )
    reranker = RecordingReranker()
    store = RecordingEvidenceStore(
        [supporting, second_supporting, wrong_segment, restricted, expired, deleted, wrong_metric]
    )

    response = _client(tmp_path, store, reranker).post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 200
    assert len(reranker.calls) == 1
    _, candidates, limit = reranker.calls[0]
    assert limit == 3
    # Both `supporting` and `second_supporting` are equally active, correctly-scoped
    # candidates now that retrieval is widened to top-3 (#75) — only the expired,
    # deleted, wrong-segment, wrong-metric, and restricted documents are excluded.
    assert {document.document_id for document in candidates} == {
        supporting.document_id,
        second_supporting.document_id,
    }
    assert {
        citation["document_id"] for citation in response.json()["evidence"]["citations"]
    } == {supporting.document_id, second_supporting.document_id}
    assert "restricted" not in response.text
    assert "expired-driver-evidence" not in response.text
    assert "deleted-driver-evidence" not in response.text
    assert "wrong-metric-evidence" not in response.text
    assert response.json()["evidence"]["citations"][0]["source_url"] == second_supporting.source_url
    assert response.json()["evidence"]["citations"][0]["source_revision"] == "43"


def test_revoked_authoritative_revision_blocks_graph_traversal_after_cited_retrieval(
    tmp_path: Path,
) -> None:
    evidence_store = RevokingEvidenceStore(list(evidence_corpus()))
    graph_store = RecordingGraphStore()
    client = _client(
        tmp_path,
        evidence_store,
        reranker=RecordingReranker(),
        graph_store=graph_store,
    )

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": _evidence_question()},
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"
    assert graph_store.calls == 0
    metadata = response.json()["lead_agent_metadata"]
    assert metadata["last_replan_reason"] == "invariant_blocked"


def test_reranker_cannot_introduce_a_candidate_outside_the_authorized_set(
    tmp_path: Path,
) -> None:
    supporting = evidence_corpus()[0]
    injected = evidence_corpus()[1]
    reranker = InjectingReranker(injected)
    client = _client(tmp_path, RecordingEvidenceStore([supporting]), reranker)

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 503
    assert "Evidence reranker is unavailable" in response.json()["detail"]
    assert injected.document_id not in response.text


def test_missing_reranker_fails_closed_without_a_weaker_evidence_answer(
    tmp_path: Path,
) -> None:
    supporting = evidence_corpus()[0]
    client = _client(tmp_path, RecordingEvidenceStore([supporting]))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 503
    assert "Evidence reranker is unavailable" in response.json()["detail"]
    assert supporting.document_id not in response.text


def test_reranker_failure_fails_closed_without_returning_unranked_evidence(
    tmp_path: Path,
) -> None:
    supporting = evidence_corpus()[0]
    client = _client(tmp_path, RecordingEvidenceStore([supporting]), FailingReranker())

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 503
    assert "Evidence reranker is unavailable" in response.json()["detail"]
    assert supporting.document_id not in response.text


def test_unresolved_driver_does_not_retrieve_or_rank_evidence(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=EmptyDriverRowsExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    evidence_store = RecordingEvidenceStore([evidence_corpus()[0]])
    reranker = RecordingReranker()
    client = TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                evidence_store=evidence_store,
                evidence_reranker=reranker,
            )
        )
    )

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "inconclusive"
    assert response.json()["evidence"]["citations"] == []
    assert evidence_store.filters == []
    assert reranker.calls == []


def test_readiness_marks_an_unconfigured_reranker_as_unavailable(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
    )

    status = AnswerQuestionService(gateway).readiness()

    assert status["reranker"] == {
        "provider": "none",
        "status": "unconfigured",
        "model": None,
        "version": None,
    }
    assert status["status"] == "unavailable"


def test_ollama_reranker_uses_the_agreed_model_and_returns_only_scored_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = list(evidence_corpus()[:2])
    documents[0] = documents[0].model_copy(
        update={"text": "Review references tenant-0099 but remains internal."}
    )
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "response": json.dumps(
                        {
                            "scores": [
                                {"candidate_id": "candidate-1", "score": 0.9},
                                {"candidate_id": "candidate-0", "score": 0.1},
                            ]
                        }
                    )
                }
            ).encode()

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    reranker = OllamaCrossEncoderReranker()

    ranked = reranker.rerank("what changed", documents, limit=1)

    assert [document.document_id for document in ranked] == [documents[1].document_id]
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert payload["model"] == RERANKER_MODEL_NAME
    assert payload["stream"] is False
    assert timeout == 60.0
    assert "tenant-0099" not in payload["prompt"]


def test_ollama_reranker_rejects_a_model_name_outside_the_governed_bundle() -> None:
    with pytest.raises(ValueError, match=RERANKER_MODEL_NAME):
        OllamaCrossEncoderReranker(model_name="another-model")


def test_external_qdrant_store_reconstructs_current_provenance_from_payload() -> None:
    client = QdrantClient(location=":memory:")
    document = evidence_corpus()[0]
    revision = ConfluenceEvidenceRevision(
        source_page_id="jira-page-123",
        source_url="https://jira.example/pages/jira-page-123",
        source_revision="43",
        metric_name=document.metric_name,
        title=document.title,
        product=document.product,
        region=document.region,
        tenant_ids=document.tenant_ids,
        tenant_scope=document.tenant_scope,
        relevant_date=document.relevant_date,
        freshness=document.freshness,
        support_status=EvidenceSupportStatus.SUPPORTS,
        support_explanation=document.support_explanation,
        chunks=[
            ConfluenceEvidenceChunk(
                chunk_id="jira-page-123:chunk:0",
                chunk_index=0,
                text=document.text,
            )
        ],
        source_access=SourceAccessMetadata(
            classification=document.classification,
            identifier_entitlement=document.identifier_entitlement,
            access_groups=document.access_groups,
            direct_principal_grants=document.direct_principal_grants,
            policy_expires_at=document.policy_expires_at,
        ),
        embedding_model="deterministic-hash",
        embedding_version="1",
    )
    QdrantEvidenceSynchronizer(client=client, collection_name="external-test").sync(
        StaticRevisionSource([revision])
    )
    external_store = QdrantEvidenceStore(
        client=client,
        collection_name="external-test",
        external=True,
    )
    profile = resolve_access_profile("apac_regional_manager")
    access_filter = profile.evidence_filter(
        "Jira",
        "APAC",
        seat_tier="51-200",
        metric_name="jira_new_peu",
        agent_user_id="apac_regional_manager",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    documents = external_store.retrieve("Jira APAC provisioning incident", access_filter, limit=3)

    assert [item.document_id for item in documents] == ["jira-page-123:chunk:0"]
    assert documents[0].source_url == revision.source_url
    assert documents[0].source_revision == revision.source_revision
    assert documents[0].chunk_id == revision.chunks[0].chunk_id


def test_external_qdrant_store_rejects_unversioned_payload_defaults() -> None:
    client = QdrantClient(location=":memory:")
    document = evidence_corpus()[0]
    QdrantEvidenceStore([document], client=client, collection_name="unversioned-test")
    external_store = QdrantEvidenceStore(
        client=client,
        collection_name="unversioned-test",
        external=True,
    )
    profile = resolve_access_profile("apac_regional_manager")
    access_filter = profile.evidence_filter(
        "Jira",
        "APAC",
        seat_tier="51-200",
        metric_name="jira_new_peu",
        agent_user_id="apac_regional_manager",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert external_store.retrieve("Jira provisioning incident", access_filter, limit=3) == []

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.contracts import EvidenceSupportStatus
from growth_data_agent.evidence import EvidenceAccessFilter, EvidenceDocument
from growth_data_agent.main import create_app
from growth_data_agent.policy import AccessDeniedError, AccessProfile, tenant_ids_for_region
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.synthetic import evidence_corpus


class RecordingEvidenceStore:
    def __init__(self, documents: list[EvidenceDocument]):
        self.documents = documents
        self.filters: list[EvidenceAccessFilter] = []

    def retrieve(
        self,
        query: str,
        access_filter: EvidenceAccessFilter,
        *,
        limit: int,
    ) -> list[EvidenceDocument]:
        self.filters.append(access_filter)
        return self.documents[:limit]


def _client(
    tmp_path: Path,
    evidence_store: RecordingEvidenceStore,
) -> TestClient:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=executor,
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    return TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                evidence_store=evidence_store,
                evidence_reranker=DeterministicCrossEncoderReranker(),
            )
        )
    )


def _evidence_question() -> str:
    return "What evidence may explain the APAC 51–200-seat Tenant decline?"


def test_evidence_filter_contains_all_entitlements_before_store_retrieval(tmp_path: Path) -> None:
    store = RecordingEvidenceStore([evidence_corpus()[0]])
    client = _client(tmp_path, store)

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 200
    assert len(store.filters) == 1
    access_filter = store.filters[0]
    assert access_filter.products == ("Jira",)
    assert access_filter.regions == ("APAC",)
    assert access_filter.tenant_ids
    assert "tenant-0011" in access_filter.tenant_ids
    assert "tenant-0001" not in access_filter.tenant_ids
    assert access_filter.classifications == ("internal",)
    assert access_filter.identifier_entitlements == ("none",)


def test_confluence_campaign_filter_contains_requested_seat_tier_before_retrieval(
    tmp_path: Path,
) -> None:
    documents = evidence_corpus()
    store = RecordingEvidenceStore(list(documents[4:7]))
    client = _client(tmp_path, store)

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What evidence may explain the Americas 11–50-seat Confluence New PEU "
                "movement after the acquisition campaign?"
            ),
        },
    )

    assert response.status_code == 200
    access_filter = store.filters[0]
    assert access_filter.seat_tiers == ("11-50",)
    assert access_filter.tenant_ids == tuple(documents[4].tenant_ids)


def test_confluence_emea_new_mau_filter_is_scoped_before_retrieval(tmp_path: Path) -> None:
    documents = evidence_corpus()
    store = RecordingEvidenceStore(list(documents[8:11]))
    client = _client(tmp_path, store)

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What evidence may explain the Confluence EMEA 51–200-seat New MAU "
                "decline after the onboarding-email regression?"
            ),
        },
    )

    assert response.status_code == 200
    access_filter = store.filters[0]
    assert access_filter.products == ("Confluence",)
    assert access_filter.regions == ("EMEA",)
    assert access_filter.seat_tiers == ("51-200",)
    assert access_filter.tenant_ids == tuple(documents[8].tenant_ids)


def test_apac_manager_receives_no_out_of_scope_or_restricted_documents(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    assert body["effective_access_scope"]["regions"] == ["APAC"]
    assert all(
        citation["affected_scope"]["region"] == "APAC"
        and citation["affected_scope"]["product"] == "Jira"
        and citation["document_id"] != "jira-apac-paid-provisioning-incident-restricted"
        for citation in body["evidence"]["citations"]
    )


def test_retrieved_restricted_document_is_not_added_to_response_context(tmp_path: Path) -> None:
    restricted_document = next(
        document
        for document in evidence_corpus()
        if document.document_id == "jira-apac-paid-provisioning-incident-restricted"
    )
    client = _client(tmp_path, RecordingEvidenceStore([restricted_document]))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "apac_regional_manager", "question": _evidence_question()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "inconclusive"
    assert body["evidence"]["citations"] == []
    assert restricted_document.document_id not in response.text


def test_product_entitlement_is_checked_before_evidence_query() -> None:
    jira_only_profile = AccessProfile(
        products=("Jira",),
        regions=("APAC",),
        tenant_scope="APAC Tenants only",
        permitted_columns=(),
        permitted_tenant_ids=tenant_ids_for_region("APAC"),
    )

    with pytest.raises(AccessDeniedError, match="Confluence"):
        jira_only_profile.evidence_filter("Confluence", "APAC")


def test_insufficient_evidence_is_explicitly_inconclusive(tmp_path: Path) -> None:
    client = _client(tmp_path, RecordingEvidenceStore([]))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": _evidence_question()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "inconclusive"
    assert body["evidence"]["citations"] == []
    assert body["evidence"]["support_status"] == "inconclusive"
    assert "Insufficient" in body["evidence"]["support_explanation"]
    assert "Hypothesis:" not in body["answer"]


def test_contradictory_evidence_is_inconclusive(tmp_path: Path) -> None:
    supporting, = evidence_corpus()[:1]
    contradicting = supporting.model_copy(
        update={
            "document_id": "jira-apac-paid-provisioning-incident-contradiction",
            "title": "Jira APAC incident review contradicts overlap",
            "support_status": EvidenceSupportStatus.CONTRADICTS,
            "support_explanation": (
                "The review says the incident did not overlap the APAC 51-200 Seat Tier "
                "Tenant decline period."
            ),
        }
    )
    client = _client(tmp_path, RecordingEvidenceStore([supporting, contradicting]))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": _evidence_question()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "inconclusive"
    assert body["evidence"]["support_status"] == "inconclusive"
    assert "contradictory" in body["evidence"]["support_explanation"]
    assert "does not establish causation" not in body["answer"]

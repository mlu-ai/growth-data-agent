from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.contracts import EvidenceSupportStatus
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
from growth_data_agent.policy import tenant_ids_for_segment
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.synthetic import evidence_corpus

_APAC_QUESTION = "What evidence may explain the APAC 51–200-seat Tenant decline?"
_APAC_TENANTS = tenant_ids_for_segment("APAC", "51-200")


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


def _apac_document(
    *,
    document_id: str,
    title: str = "Jira APAC provisioning incident",
    text: str | None = None,
    relevant_date: date = date(2026, 6, 12),
    support_status: EvidenceSupportStatus = EvidenceSupportStatus.SUPPORTS,
    is_high_authority_operational_record: bool = False,
    source_document_id: str | None = None,
) -> EvidenceDocument:
    """A minimal, self-contained Jira APAC 51-200 evidence document for a small,
    controlled test store — not part of the shared synthetic corpus."""
    return EvidenceDocument(
        document_id=document_id,
        metric_name="jira_new_peu",
        title=title,
        text=text or (
            "Paid provisioning errors affected Jira APAC 51-200 Seat Tier Tenants "
            "from 2026-06-10 through 2026-06-12, overlapping the June New PEU decline."
        ),
        product="Jira",
        region="APAC",
        tenant_ids=_APAC_TENANTS,
        tenant_scope="APAC 51-200 Seat Tier Tenants",
        classification="internal",
        identifier_entitlement="none",
        relevant_date=relevant_date,
        freshness=datetime(2026, 6, 13, tzinfo=UTC),
        support_status=support_status,
        support_explanation="Overlaps the APAC 51-200 Seat Tier Tenant scope and period.",
        accountable_team="Jira Platform Provisioning Team",
        is_high_authority_operational_record=is_high_authority_operational_record,
        source_document_id=source_document_id or document_id,
    )


def test_inaccessible_source_revision_stops_supporting_the_next_answer(tmp_path: Path) -> None:
    documents = list(evidence_corpus())
    store = _MutableEvidenceStore(documents)
    client = _client_for(tmp_path, store)

    first = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    assert first.status_code == 200
    first_factors = first.json()["candidate_causal_factors"]
    assert len(first_factors) == 1
    assert first_factors[0]["citations"][0]["source_document_id"] == (
        "jira-apac-paid-provisioning-incident"
    )

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
    assert second_body["candidate_causal_factors"] == []
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
    assert body["candidate_causal_factors"] == []
    assert body["result_classification"] == "inconclusive"
    assert "no rank-eligible Candidate Causal Factor" in body["answer"]


def test_two_independent_supports_yield_supported_status(tmp_path: Path) -> None:
    doc_a = _apac_document(document_id="jira-apac-two-support-a")
    doc_b = _apac_document(
        document_id="jira-apac-two-support-b",
        text=(
            "A follow-up provisioning report independently confirms Jira APAC 51-200 Seat "
            "Tier Tenants were affected from 2026-06-10 through 2026-06-12, overlapping the "
            "June New PEU decline."
        ),
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([doc_a, doc_b]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    factors = body["candidate_causal_factors"]
    assert len(factors) == 1
    factor = factors[0]
    assert factor["status"] == "supported"
    assert factor["ranking_signals"]["independent_source_count"] == 2
    assert {c["source_document_id"] for c in factor["citations"]} == {
        "jira-apac-two-support-a",
        "jira-apac-two-support-b",
    }


def test_duplicate_source_revision_does_not_inflate_independent_source_count(
    tmp_path: Path,
) -> None:
    original = _apac_document(document_id="jira-apac-duplicate-original")
    second_chunk = _apac_document(
        document_id="jira-apac-duplicate-chunk-2",
        text="A second chunk of the same incident page, distinct wording.",
        source_document_id="jira-apac-duplicate-original",
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([original, second_chunk]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    factors = body["candidate_causal_factors"]
    assert len(factors) == 1
    factor = factors[0]
    assert factor["status"] == "inconclusive"
    assert factor["ranking_signals"]["independent_source_count"] == 1
    assert {c["source_document_id"] for c in factor["citations"]} == {
        "jira-apac-duplicate-original"
    }


def test_high_authority_operational_record_yields_supported_with_single_source(
    tmp_path: Path,
) -> None:
    doc = _apac_document(
        document_id="jira-apac-high-authority-record",
        is_high_authority_operational_record=True,
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([doc]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    factors = response.json()["candidate_causal_factors"]
    assert len(factors) == 1
    assert factors[0]["status"] == "supported"
    assert factors[0]["ranking_signals"]["independent_source_count"] == 1


def test_material_contradiction_yields_contradicted_status(tmp_path: Path) -> None:
    supporting = _apac_document(document_id="jira-apac-contradiction-support")
    contradicting = _apac_document(
        document_id="jira-apac-contradiction-refute",
        title="Jira APAC provisioning incident retrospective",
        text=(
            "A retrospective review found the Jira APAC provisioning incident was fully "
            "mitigated before the 51-200 Seat Tier Tenant New PEU decline began."
        ),
        support_status=EvidenceSupportStatus.CONTRADICTS,
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([supporting, contradicting]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    factors = body["candidate_causal_factors"]
    assert len(factors) == 1
    assert factors[0]["status"] == "contradicted"
    assert factors[0]["ranking_signals"]["counterevidence"] == "material"
    # A card is still visible (the "challenge" half of ranking), but the overall
    # classification does not present a materially contradicted hypothesis as governed.
    assert body["result_classification"] == "inconclusive"


def test_background_only_evidence_is_excluded_entirely(tmp_path: Path) -> None:
    doc = _apac_document(
        document_id="jira-apac-background-only",
        support_status=EvidenceSupportStatus.INCONCLUSIVE,
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([doc]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    assert response.json()["candidate_causal_factors"] == []


def test_out_of_scope_temporal_document_is_excluded_from_a_multi_candidate_response(
    tmp_path: Path,
) -> None:
    in_scope = _apac_document(
        document_id="jira-apac-in-scope",
        is_high_authority_operational_record=True,
    )
    out_of_scope = _apac_document(
        document_id="jira-apac-out-of-scope",
        title="Jira APAC entitlement review",
        text="An unrelated Jira APAC entitlement review from March 2026.",
        relevant_date=date(2026, 3, 1),
        is_high_authority_operational_record=True,
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([in_scope, out_of_scope]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    factors = body["candidate_causal_factors"]
    assert len(factors) == 1
    assert factors[0]["citations"][0]["source_document_id"] == "jira-apac-in-scope"
    # The out-of-scope document is still authorized evidence (visible in the raw
    # `evidence` aggregate), but it must never surface as — or be cited by — a
    # ranked Candidate Causal Factor card.
    factor_source_ids = {
        citation["source_document_id"]
        for factor in factors
        for citation in factor["citations"]
    }
    assert "jira-apac-out-of-scope" not in factor_source_ids


def test_response_reports_at_most_three_candidate_cards(tmp_path: Path) -> None:
    categories = [
        ("Jira APAC provisioning incident A", "provisioning"),
        ("Jira APAC billing subscription incident B", "billing"),
        ("Jira APAC identity access incident C", "identity"),
        ("Jira APAC incident report D", "incident"),
    ]
    documents = [
        _apac_document(
            document_id=f"jira-apac-cap-{index}",
            title=title,
            text=f"{title} affecting Jira APAC 51-200 Seat Tier Tenants in June 2026.",
            relevant_date=date(2026, 6, 10 + index),
            is_high_authority_operational_record=True,
        )
        for index, (title, _keyword) in enumerate(categories)
    ]
    client = _client_for(tmp_path, _MutableEvidenceStore(documents))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    assert len(response.json()["candidate_causal_factors"]) <= 3


def test_ranking_signals_are_typed_and_no_opaque_score_is_present(tmp_path: Path) -> None:
    doc = _apac_document(
        document_id="jira-apac-signals-shape",
        is_high_authority_operational_record=True,
    )
    client = _client_for(tmp_path, _MutableEvidenceStore([doc]))

    response = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )

    assert response.status_code == 200
    factor = response.json()["candidate_causal_factors"][0]
    signals = factor["ranking_signals"]
    assert set(signals.keys()) == {
        "temporal_alignment",
        "population_overlap",
        "metric_mechanism_fit",
        "independent_source_count",
        "counterevidence",
    }
    assert signals["temporal_alignment"] in ("within_movement_window", "within_initial_lookback")
    assert signals["population_overlap"] in ("exact_segment_match", "partial_or_broader_scope")
    assert isinstance(signals["metric_mechanism_fit"], bool)
    assert isinstance(signals["independent_source_count"], int)
    assert signals["counterevidence"] in ("none", "material")
    assert "score" not in signals
    assert "confidence" not in signals

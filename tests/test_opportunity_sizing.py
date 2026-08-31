from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.contracts import EvidenceSupportStatus
from growth_data_agent.evidence import EvidenceAccessFilter, EvidenceDocument
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
    document_id: str = "jira-apac-paid-provisioning-incident",
    title: str = "Jira APAC provisioning incident",
    text: str | None = None,
) -> EvidenceDocument:
    """The Jira APAC 51-200 provisioning-incident document — its category resolves
    to PROVISIONING_OR_ENTITLEMENT, the sole Sizing Eligible category in this
    delivery, and its driver metric (jira_new_peu) has a governed Eligible
    Population mapping (jira_new_peu_eligible_population)."""
    return EvidenceDocument(
        document_id=document_id,
        metric_name="jira_new_peu",
        title=title,
        text=text
        or (
            "Paid provisioning errors affected Jira APAC 51-200 Seat Tier Tenants "
            "from 2026-06-10 through 2026-06-12, overlapping the June New PEU decline."
        ),
        product="Jira",
        region="APAC",
        tenant_ids=_APAC_TENANTS,
        tenant_scope="APAC 51-200 Seat Tier Tenants",
        classification="internal",
        identifier_entitlement="none",
        relevant_date=date(2026, 6, 12),
        freshness=datetime(2026, 6, 13, tzinfo=UTC),
        support_status=EvidenceSupportStatus.SUPPORTS,
        support_explanation="Overlaps the APAC 51-200 Seat Tier Tenant scope and period.",
        accountable_team="Jira Platform Provisioning Team",
        is_high_authority_operational_record=True,
        source_document_id=document_id,
    )


def _select(client: TestClient, *, agent_user_id: str = "data_analyst", factor_id: str):
    return client.post(
        "/answer_question",
        json={
            "agent_user_id": agent_user_id,
            "question": _APAC_QUESTION,
            "selected_factor_id": factor_id,
        },
    )


def _size(
    client: TestClient,
    *,
    agent_user_id: str = "data_analyst",
    factor_id: str | None = None,
    scenario_percentage_points: float,
):
    payload: dict[str, object] = {
        "agent_user_id": agent_user_id,
        "question": _APAC_QUESTION,
        "opportunity_scenario_percentage_points": scenario_percentage_points,
    }
    if factor_id is not None:
        payload["selected_factor_id"] = factor_id
    return client.post("/answer_question", json=payload)


def test_governed_mapping_and_scenario_produce_an_exact_opportunity_estimate(
    tmp_path: Path,
) -> None:
    store = _MutableEvidenceStore([_apac_document(document_id="jira-apac-provisioning")])
    client = _client_for(tmp_path, store)

    discover = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    assert discover.status_code == 200
    factors = discover.json()["candidate_causal_factors"]
    assert len(factors) == 1
    assert factors[0]["sizing_eligible"] is True
    factor_id = factors[0]["factor_id"]

    sized = _size(client, factor_id=factor_id, scenario_percentage_points=5.0)

    assert sized.status_code == 200
    body = sized.json()
    assert body["result_classification"] == "opportunity_estimate"
    estimate = body["opportunity_estimate"]
    assert estimate["factor_id"] == factor_id
    # Fixture comparison_value for APAC 51-200 (RecordingPostgresExecutor) is 380;
    # eligible_population is the conftest fake's fixed jira_new_peu_eligible_population row (40).
    assert estimate["eligible_population"] == 40
    assert estimate["baseline_rate_percentage"] == round(380 / 40 * 100, 2)
    assert estimate["scenario_percentage_point_change"] == 5.0
    assert estimate["incremental_product_users"] == round(40 * 5.0 / 100)
    assert "eligible_population" in estimate["formula"]
    assert estimate["scenario_window_start"] == "2026-06-01"
    assert estimate["scenario_window_end"] == "2026-06-30"
    assert body["candidate_causal_factors"] == [factors[0]]


def test_missing_selection_with_a_scenario_returns_a_limitation(tmp_path: Path) -> None:
    store = _MutableEvidenceStore([_apac_document()])
    client = _client_for(tmp_path, store)

    response = _size(client, factor_id=None, scenario_percentage_points=5.0)

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "limitation"
    assert body["opportunity_estimate"] is None
    assert len(body["candidate_causal_factors"]) == 1


def test_selection_without_a_scenario_is_unaffected_by_sizing(tmp_path: Path) -> None:
    store = _MutableEvidenceStore([_apac_document()])
    client = _client_for(tmp_path, store)

    discover = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    factor_id = discover.json()["candidate_causal_factors"][0]["factor_id"]

    selected = _select(client, factor_id=factor_id)

    assert selected.status_code == 200
    body = selected.json()
    assert body["result_classification"] == "hypothesis"
    assert body["opportunity_estimate"] is None
    assert body["opportunity_sizing_gap"] is None
    assert [card["factor_id"] for card in body["candidate_causal_factors"]] == [factor_id]


def test_ungoverned_category_offers_a_mapping_request_instead_of_an_estimate(
    tmp_path: Path,
) -> None:
    store = _MutableEvidenceStore(
        [
            _apac_document(
                document_id="jira-apac-service-incident",
                title="Jira APAC service incident",
                text=(
                    "A service incident affected Jira APAC 51-200 Seat Tier Tenants "
                    "from 2026-06-10 through 2026-06-12, overlapping the June New PEU "
                    "decline."
                ),
            )
        ]
    )
    client = _client_for(tmp_path, store)

    discover = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    factors = discover.json()["candidate_causal_factors"]
    assert len(factors) == 1
    assert factors[0]["category"] == "incident"
    assert factors[0]["sizing_eligible"] is False
    factor_id = factors[0]["factor_id"]

    sized = _size(client, factor_id=factor_id, scenario_percentage_points=5.0)

    assert sized.status_code == 200
    body = sized.json()
    assert body["result_classification"] == "hypothesis"
    assert body["opportunity_estimate"] is None
    gap = body["opportunity_sizing_gap"]
    assert gap["factor_id"] == factor_id
    assert gap["category"] == "incident"
    assert gap["mapping_request_offered"] is True


def test_entitlement_narrower_than_seat_tier_is_rejected_before_an_estimate(
    tmp_path: Path,
) -> None:
    store = _MutableEvidenceStore([_apac_document()])
    client = _client_for(tmp_path, store)

    discover = client.post(
        "/answer_question", json={"agent_user_id": "data_analyst", "question": _APAC_QUESTION}
    )
    factor_id = discover.json()["candidate_causal_factors"][0]["factor_id"]

    # customer_success_manager's permitted_query_columns omits seat_tier, so it
    # cannot complete a seat-tier-scoped investigation for this segment at all —
    # confirming sizing never leaks an Opportunity Estimate to an under-entitled
    # profile, regardless of which authorization check in the pipeline rejects it.
    discovery_denied = client.post(
        "/answer_question",
        json={"agent_user_id": "customer_success_manager", "question": _APAC_QUESTION},
    )
    assert discovery_denied.status_code == 403

    sized = _size(
        client,
        agent_user_id="customer_success_manager",
        factor_id=factor_id,
        scenario_percentage_points=5.0,
    )

    assert sized.status_code == 403

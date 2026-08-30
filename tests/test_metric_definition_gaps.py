from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.contracts import (
    ProvisionalMetric,
    ProvisionalMetricFreshness,
    ProvisionalMetricInput,
)
from growth_data_agent.main import create_app
from growth_data_agent.metric_definition_gaps import (
    InMemoryDataTeamVerificationRequestRecorder,
    ProvisionalMetricInputRequest,
    ScopedProvisionalInputs,
)
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


class SupportedProvisionalMetricCalculator:
    def required_inputs(self, metric_name):
        if metric_name != "jira_paid_enablement_event_count":
            return None
        return (
            ProvisionalMetricInput(
                name="paid_enablement_id",
                source="permitted immutable Paid Enablement events",
            ),
        )

    def calculate(self, scoped_inputs, semantic_freshness):
        return ProvisionalMetric(
            name="jira_paid_enablement_event_count",
            value=len(scoped_inputs.records),
            formula="count(paid_enablement_id) where product = Jira",
            inputs=[
                ProvisionalMetricInput(
                    name="paid_enablement_id",
                    source="permitted immutable Paid Enablement events",
                )
            ],
            scope=scoped_inputs.request.scope,
            freshness=ProvisionalMetricFreshness(
                source="synthetic Paid Enablement events",
                observed_at=datetime(2026, 8, 25, tzinfo=UTC),
            ),
            material_caveats=[
                "This event count is unverified and is not a canonical metric.",
                "It does not deduplicate Product Users or apply a validated time rule.",
            ],
        )


class RecordingProvisionalMetricInputGateway:
    def __init__(self) -> None:
        self.requests: list[ProvisionalMetricInputRequest] = []

    def read(self, request: ProvisionalMetricInputRequest) -> ScopedProvisionalInputs:
        self.requests.append(request)
        return ScopedProvisionalInputs(
            request=request,
            records=tuple({"paid_enablement_id": f"event-{number}"} for number in range(17)),
        )


def _client_and_recorder(tmp_path: Path, *, calculator=None, input_gateway=None):
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=executor,
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    recorder = InMemoryDataTeamVerificationRequestRecorder(
        now=lambda: datetime(2026, 8, 25, 2, tzinfo=UTC)
    )
    service = AnswerQuestionService(
        gateway,
        provisional_metric_calculator=calculator,
        provisional_metric_input_gateway=input_gateway,
        verification_request_recorder=recorder,
    )
    return TestClient(create_app(service)), recorder


def test_missing_canonical_metric_is_a_gap_and_never_a_canonical_metric(tmp_path: Path) -> None:
    client, recorder = _client_and_recorder(tmp_path)

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira Activation?"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "metric_definition_gap"
    assert body["metric_definition_gap"] == {
        "requested_metric_name": "jira_activation",
        "semantic_authority": "dbt/MetricFlow",
        "verification_request_offered": True,
    }
    assert body["canonical_definition"] is None
    assert body["provisional_metric"] is None
    assert "cannot support it safely" in body["answer"]
    assert recorder.requests == []


def test_generic_named_metric_is_classified_as_a_metric_definition_gap(tmp_path: Path) -> None:
    client, _ = _client_and_recorder(tmp_path)

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Foo?"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "metric_definition_gap"
    assert body["metric_definition_gap"]["requested_metric_name"] == "foo"


def test_provisional_metric_includes_required_provenance_and_caveats(tmp_path: Path) -> None:
    input_gateway = RecordingProvisionalMetricInputGateway()
    client, _ = _client_and_recorder(
        tmp_path,
        calculator=SupportedProvisionalMetricCalculator(),
        input_gateway=input_gateway,
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "Count Jira paid enablement events",
            "requested_metric_name": "Jira Paid Enablement Event Count",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "provisional_metric"
    assert body["canonical_definition"] is None
    assert body["metric_definition_gap"]["requested_metric_name"] == (
        "jira_paid_enablement_event_count"
    )
    assert body["provisional_metric"] == {
        "name": "jira_paid_enablement_event_count",
        "value": 17,
        "formula": "count(paid_enablement_id) where product = Jira",
        "inputs": [
            {
                "name": "paid_enablement_id",
                "source": "permitted immutable Paid Enablement events",
            }
        ],
        "scope": {
            "products": ["Jira", "Confluence"],
            "regions": ["APAC"],
            "tenant_scope": "APAC Tenants only",
            "permitted_columns": [
                "metric_name",
                "definition",
                "formula",
                "grain",
                "time_rule",
                "semantic_version",
                "source_freshness",
                "paid_enablement_id",
            ],
        },
        "verification_status": "unverified",
        "freshness": {
            "source": "synthetic Paid Enablement events",
            "observed_at": "2026-08-25T00:00:00Z",
        },
        "material_caveats": [
            "This event count is unverified and is not a canonical metric.",
            "It does not deduplicate Product Users or apply a validated time rule.",
        ],
    }
    assert len(input_gateway.requests) == 1
    assert input_gateway.requests[0].metric_name == "jira_paid_enablement_event_count"
    assert input_gateway.requests[0].inputs == (
        ProvisionalMetricInput(
            name="paid_enablement_id",
            source="permitted immutable Paid Enablement events",
        ),
    )
    assert input_gateway.requests[0].scope.regions == ["APAC"]
    assert input_gateway.requests[0].scope.tenant_scope == "APAC Tenants only"


def test_provisional_calculation_with_an_unpermitted_input_is_refused(tmp_path: Path) -> None:
    class UnsafeCalculator:
        def required_inputs(self, metric_name):
            return (ProvisionalMetricInput(name="unpermitted_input", source="raw event"),)

        def calculate(self, scoped_inputs, semantic_freshness):
            raise AssertionError("The service must authorize inputs before retrieving data.")

    class FailingInputGateway:
        def read(self, request):
            raise AssertionError("The service must authorize inputs before retrieving data.")

    client, _ = _client_and_recorder(
        tmp_path,
        calculator=UnsafeCalculator(),
        input_gateway=FailingInputGateway(),
    )

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira Activation?"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "metric_definition_gap"
    assert body["provisional_metric"] is None


def test_provisional_calculation_is_refused_when_gateway_returns_extra_input(
    tmp_path: Path,
) -> None:
    class Calculator:
        def required_inputs(self, metric_name):
            return (
                ProvisionalMetricInput(
                    name="paid_enablement_id",
                    source="permitted immutable Paid Enablement events",
                ),
            )

        def calculate(self, scoped_inputs, semantic_freshness):
            raise AssertionError("The calculator must not receive unrequested inputs.")

    class Gateway:
        def read(self, request):
            return ScopedProvisionalInputs(
                request=request,
                records=(
                    {
                        "paid_enablement_id": "event-1",
                        "unpermitted_input": "must-not-reach-calculator",
                    },
                ),
            )

    client, _ = _client_and_recorder(tmp_path, calculator=Calculator(), input_gateway=Gateway())

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira Activation?"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "metric_definition_gap"
    assert body["provisional_metric"] is None


def test_data_team_request_requires_explicit_confirmation_and_records_context(
    tmp_path: Path,
) -> None:
    client, recorder = _client_and_recorder(tmp_path)
    request = {
        "agent_user_id": "data_analyst",
        "question": "What is Jira Activation?",
        "verification_request_confirmation": {
            "approved": True,
            "approval_context": "The Agent User approved review of the Jira Activation definition.",
        },
    }

    response = client.post("/answer_question", json=request)

    body = response.json()
    assert response.status_code == 200
    assert len(recorder.requests) == 1
    assert body["data_team_verification_request"] == {
        "request_id": recorder.requests[0].request_id,
        "requested_metric_name": "jira_activation",
        "requested_by_agent_user_id": "data_analyst",
        "approval_context": "The Agent User approved review of the Jira Activation definition.",
        "approval_context_sha256": hashlib.sha256(
            b"The Agent User approved review of the Jira Activation definition."
        ).hexdigest(),
        "approved_at": "2026-08-25T02:00:00Z",
        "decision_outcome": "approved",
        "trace_id": body["trace_id"],
    }


def test_declined_verification_confirmation_does_not_create_a_request(tmp_path: Path) -> None:
    client, recorder = _client_and_recorder(tmp_path)

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What is Jira Activation?",
            "verification_request_confirmation": {
                "approved": False,
                "approval_context": "The Agent User declined the verification request.",
            },
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["data_team_verification_request"] is None
    assert recorder.requests == []

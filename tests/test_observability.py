from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.main import create_app
from growth_data_agent.observability import MlflowTraceSink, TraceRecord, policy_fingerprint
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


class RecordingMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, dict] = {}

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    @contextmanager
    def start_run(self, **kwargs):
        self.run_kwargs = kwargs
        yield self

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_params(self, values: dict[str, str]) -> None:
        self.params.update(values)

    def log_metrics(self, values: dict[str, float]) -> None:
        self.metrics.update(values)

    def log_dict(self, value: dict, artifact_file: str) -> None:
        self.artifacts[artifact_file] = value


def test_mlflow_trace_is_redacted_and_contains_governance_fields() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)
    record = TraceRecord(
        trace_id="trace-123",
        request_route="answer_question",
        response_classification="direct_identifier_response",
        policy_fingerprint="policy-abc",
        source_versions={"semantic_version": "1.0.0"},
        tool_outcomes={"semantic_query": "not_used", "retrieval": "success"},
        retrieval_scores=(0.91, 0.42),
        evaluation_outcome="pass",
        response={
            "answer": "tenant-0011 was affected",
            "direct_identifier_answer": {"value": "tenant-0011"},
        },
    )

    sink.record(record)

    assert mlflow.tags == {
        "trace_id": "trace-123",
        "route": "answer_question",
        "request_route": "answer_question",
        "response_classification": "direct_identifier_response",
        "policy_fingerprint": "policy-abc",
        "evaluation_outcome": "pass",
    }
    assert mlflow.params == {
        "semantic_version": "1.0.0",
        "semantic_query_outcome": "not_used",
        "retrieval_outcome": "success",
    }
    assert mlflow.metrics == {
        "retrieval_count": 2.0,
        "retrieval_top_score": 0.91,
        "retrieval_mean_score": 0.665,
    }
    payload = mlflow.artifacts["governed_trace.json"]
    assert "tenant-0011" not in str(payload)
    assert "[redacted identifier]" in str(payload)
    assert payload["trace_id"] == "trace-123"


def test_policy_fingerprint_includes_row_entitlements_without_exposing_them() -> None:
    profile = resolve_access_profile("apac_regional_manager")
    narrowed_profile = replace(
        profile,
        permitted_tenant_ids=profile.permitted_tenant_ids[:-1],
    )

    fingerprint = policy_fingerprint(profile)
    narrowed_fingerprint = policy_fingerprint(narrowed_profile)

    assert fingerprint != narrowed_fingerprint
    assert "tenant-" not in fingerprint


class RecordingTraceSink:
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def record(self, trace: TraceRecord) -> None:
        self.records.append(trace)


def test_governed_response_records_route_tools_and_source_versions(
    tmp_path: Path,
) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    trace_sink = RecordingTraceSink()
    client = TestClient(
        create_app(AnswerQuestionService(gateway, trace_sink=trace_sink))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 200
    assert len(trace_sink.records) == 1
    trace = trace_sink.records[0]
    assert trace.trace_id == response.json()["trace_id"]
    assert trace.response_classification == "hypothesis"
    assert trace.policy_fingerprint
    assert trace.source_versions["semantic_version"] == "1.0.0"
    assert trace.tool_outcomes == {
        "semantic_query": "success",
        "retrieval": "success",
        "graph": "success",
        "direct_identifier_audit": "not_used",
    }


def test_authorization_denial_is_traced_before_source_retrieval(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    trace_sink = RecordingTraceSink()
    client = TestClient(
        create_app(AnswerQuestionService(gateway, trace_sink=trace_sink))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "confluence_product_manager",
            "question": "What is Jira New PEU?",
        },
    )

    assert response.status_code == 403
    assert len(trace_sink.records) == 1
    trace = trace_sink.records[0]
    assert f"trace_id={trace.trace_id}" in response.json()["detail"]
    assert trace.response_classification == "safe_refusal"
    assert trace.source_versions["semantic_version"] == "1.0.0"
    assert trace.tool_outcomes == {
        "semantic_query": "not_used",
        "retrieval": "not_used",
        "graph": "not_used",
        "direct_identifier_audit": "not_used",
    }

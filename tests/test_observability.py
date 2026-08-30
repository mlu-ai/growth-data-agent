from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.graph import EvidenceGraphUnavailableError
from growth_data_agent.main import create_app
from growth_data_agent.observability import (
    MlflowTraceSink,
    TraceRecord,
    TraceSpan,
    policy_fingerprint,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


class RecordingMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, dict] = {}
        self.spans: list[dict[str, str]] = []

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    @contextmanager
    def start_run(self, **kwargs):
        self.run_kwargs = kwargs
        yield self

    @contextmanager
    def start_span(self, **kwargs):
        self.spans.append(kwargs)
        yield self

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_params(self, values: dict[str, str]) -> None:
        self.params.update(values)

    def log_param(self, key: str, value: str) -> None:
        self.params[key] = value

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
    assert payload["response"]["has_direct_identifier_answer"] is True
    assert payload["trace_id"] == "trace-123"
    assert [span["name"] for span in mlflow.spans] == ["answer_question:trace-123"]


def test_mlflow_trace_redacts_span_payloads() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)
    record = TraceRecord(
        trace_id="trace-123",
        request_route="answer_question",
        response_classification="hypothesis",
        policy_fingerprint="policy-abc",
        source_versions={},
        tool_outcomes={},
        retrieval_scores=(),
        evaluation_outcome="not_evaluated",
        response={},
        node_spans=(
            TraceSpan(
                name="evidence_retrieval",
                kind="tool",
                status="success",
                attributes={
                    "chunk_text": "restricted tenant-0011 details",
                    "entity_name": "tenant-0011",
                },
            ),
        ),
    )

    sink.record(record)

    span = mlflow.artifacts["governed_trace.json"]["node_spans"][0]
    assert "restricted tenant-0011 details" not in str(span)
    assert span["attributes"] == {"entity_name": "[redacted identifier]"}
    assert [span["name"] for span in mlflow.spans] == [
        "answer_question:trace-123",
        "evidence_retrieval",
    ]


def test_mlflow_trace_does_not_persist_source_page_bodies() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)
    record = TraceRecord(
        trace_id="trace-123",
        request_route="answer_question",
        response_classification="hypothesis",
        policy_fingerprint="policy-abc",
        source_versions={},
        tool_outcomes={},
        retrieval_scores=(),
        evaluation_outcome="not_evaluated",
        response={
            "answer": "A safe summary",
            "approval_context": "confidential source page body",
            "source_page_body": "confidential source page body",
            "evidence": {"text": "confidential source page body"},
        },
    )

    sink.record(record)

    payload = mlflow.artifacts["governed_trace.json"]
    assert "confidential source page body" not in str(payload)


def test_mlflow_trace_drops_unstructured_prose_and_arbitrary_identifier_values() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)
    raw_identifier = "customer-identity-ALPHA"
    raw_source = "unstructured source-page content"
    raw_span_value = "source-page body tenant-0099"
    record = TraceRecord(
        trace_id="trace-123",
        request_route="answer_question",
        response_classification="hypothesis",
        policy_fingerprint="policy-abc",
        source_versions={},
        tool_outcomes={},
        retrieval_scores=(),
        evaluation_outcome="not_evaluated",
        response={
            "result_classification": "hypothesis",
            "answer": raw_source,
            "support_explanation": raw_source,
            "page_content": raw_source,
            "direct_identifier_answer": {
                "identifiers": [
                    {"identifier_type": "tenant_id", "value": raw_identifier}
                ]
            },
        },
        node_spans=(
            TraceSpan(
                name=raw_identifier,
                kind="tool",
                status="success",
                attributes={"entity_name": raw_span_value},
            ),
        ),
    )

    sink.record(record)

    assert raw_identifier not in str(mlflow.artifacts["governed_trace.json"])
    assert raw_source not in str(mlflow.artifacts["governed_trace.json"])
    assert raw_span_value not in str(mlflow.artifacts["governed_trace.json"])
    assert raw_identifier not in str(mlflow.spans)
    assert raw_span_value not in str(mlflow.spans)


def test_mlflow_evaluation_hook_links_judgement_to_parent_trace() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)

    sink.record_evaluation(
        trace_id="trace-123",
        fixture_id="apac-incident",
        category="answer_faithfulness",
        model_name="qwen3:8b",
        passed=True,
        metrics={"reciprocal_rank": 1.0},
    )

    assert mlflow.tags == {
        "trace_id": "trace-123",
        "fixture_id": "apac-incident",
        "evaluation_category": "answer_faithfulness",
        "evaluation_outcome": "pass",
    }
    assert mlflow.params == {"model_name": "qwen3:8b"}
    assert mlflow.metrics == {"fixture_passed": 1.0, "reciprocal_rank": 1.0}


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


class FailingGraphStore:
    def traverse(self, query, access_filter, *, limit, metric_name=None):
        raise EvidenceGraphUnavailableError("backend detail must not reach the response")


@pytest.mark.parametrize(
    ("question", "agent_user_id", "expected_node"),
    [
        ("What is Jira New PEU?", "data_analyst", "canonical_definition"),
        ("Why did Jira New PEU fall from May to June?", "data_analyst", "driver_decomposition"),
        (
            "What is the causal estimate for the registered Jira New MAU onboarding "
            "treatment/control experiment?",
            "data_analyst",
            "causal_analysis",
        ),
        ("Who owns the Jira New PEU metric?", "data_analyst", "catalog_ownership"),
        (
            "Which Tenant IDs were affected by the Jira APAC paid provisioning incident?",
            "customer_success_manager",
            "direct_identifier",
        ),
        ("What is the weather in Sydney?", "data_analyst", "limitation"),
        ("Define New Trials", "data_analyst", "metric_definition_gap"),
        (
            "What evidence may explain the APAC 51–200-seat Tenant decline?",
            "data_analyst",
            "legacy",
        ),
    ],
)
def test_each_supported_route_records_one_parent_trace(
    tmp_path: Path,
    question: str,
    agent_user_id: str,
    expected_node: str,
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
        create_app(
            AnswerQuestionService(
                gateway,
                trace_sink=trace_sink,
                evidence_reranker=DeterministicCrossEncoderReranker(),
            )
        )
    )

    response = client.post(
        "/answer_question",
        json={"agent_user_id": agent_user_id, "question": question},
    )

    assert response.status_code == 200
    assert len(trace_sink.records) == 1
    trace = trace_sink.records[0]
    assert trace.trace_id == response.json()["trace_id"]
    assert [span.name for span in trace.node_spans][-1] == expected_node
    assert all(span.kind == "node" for span in trace.node_spans)
    if expected_node == "legacy":
        citation = response.json()["evidence"]["citations"][0]
        assert {
            "source_document_id",
            "source_url",
            "source_revision",
            "chunk_id",
        } <= citation.keys()


def test_clarification_route_records_one_parent_trace(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    trace_sink = RecordingTraceSink()
    service = AnswerQuestionService(gateway, trace_sink=trace_sink)

    class MalformedInterpreter:
        def interpret(self, request):
            return {"route": "not-a-route"}

    service.execution_graph._intent_interpreter = MalformedInterpreter()
    client = TestClient(create_app(service))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "Clarify this request"},
    )

    assert response.status_code == 200
    assert len(trace_sink.records) == 1
    trace = trace_sink.records[0]
    assert [span.name for span in trace.node_spans][-1] == "clarification"


def test_authorization_denial_trace_stops_before_tool_spans(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    trace_sink = RecordingTraceSink()
    client = TestClient(create_app(AnswerQuestionService(gateway, trace_sink=trace_sink)))

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
    assert trace.tool_spans == ()
    assert [span.name for span in trace.node_spans] == [
        "authorize",
        "intent_interpretation",
        "intent_validation",
        "canonical_definition",
    ]
    assert trace.node_spans[-1].status == "error"


def test_dependency_failure_is_fail_closed_and_traced(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    trace_sink = RecordingTraceSink()
    client = TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                graph_store=FailingGraphStore(),
                trace_sink=trace_sink,
            )
        )
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 503
    assert "backend detail" not in response.text
    assert len(trace_sink.records) == 1
    trace = trace_sink.records[0]
    assert f"trace_id={trace.trace_id}" in response.json()["detail"]
    assert trace.response_classification == "safe_refusal"
    assert trace.tool_spans[-1].name == "graph_traversal"
    assert trace.tool_spans[-1].status == "error"


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
        create_app(
            AnswerQuestionService(
                gateway,
                trace_sink=trace_sink,
                evidence_reranker=DeterministicCrossEncoderReranker(),
            )
        )
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
        "causal_pipeline": "not_used",
    }
    assert [span.name for span in trace.node_spans] == [
        "authorize",
        "intent_interpretation",
        "intent_validation",
        "legacy",
    ]
    assert [span.name for span in trace.tool_spans] == [
        "semantic_driver_decomposition",
        "graph_traversal",
        "evidence_retrieval",
    ]


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
        "causal_pipeline": "not_used",
    }

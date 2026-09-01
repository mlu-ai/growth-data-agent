from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent import observability
from growth_data_agent.graph import EvidenceGraphUnavailableError
from growth_data_agent.lightrag import (
    InMemoryLightRAGStore,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
)
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
from growth_data_agent.synthetic import evidence_corpus


class RecordingMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, dict] = {}
        self.spans: list[dict[str, str]] = []
        self.run_count = 0

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    @contextmanager
    def start_run(self, **kwargs):
        self.run_count += 1
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
        "trace_delivery_state": "attempted",
    }
    assert mlflow.params == {
        "semantic_version": "1.0.0",
        "semantic_query_outcome": "not_used",
        "retrieval_outcome": "success",
    }
    assert mlflow.metrics == {
        "retrieval_count": 2.0,
        "turn_latency_ms": 0.0,
        "retrieval_top_score": 0.91,
        "retrieval_mean_score": 0.665,
    }
    payload = mlflow.artifacts["governed_trace.json"]
    assert "tenant-0011" not in str(payload)
    assert payload["response"]["has_direct_identifier_answer"] is True
    assert payload["trace_id"] == "trace-123"
    assert [span["name"] for span in mlflow.spans] == ["answer_question:trace-123"]


def test_mlflow_trace_sink_uses_private_tracking_uri_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow = RecordingMlflow()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://go/mlflow")
    monkeypatch.setattr(observability, "_load_mlflow", lambda: mlflow)

    sink = MlflowTraceSink.from_environment()

    assert sink.experiment_name == "growth-data-agent"
    assert mlflow.tracking_uri == "http://go/mlflow"


@pytest.mark.parametrize("tracking_uri", (None, "", "   "))
def test_mlflow_trace_sink_requires_private_uri_outside_development(
    monkeypatch: pytest.MonkeyPatch, tracking_uri: str | None
) -> None:
    if tracking_uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("GROWTH_DATA_AGENT_ENVIRONMENT", "production")

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        MlflowTraceSink.from_environment()


def test_mlflow_trace_records_safe_latency_investigation_and_sizing_metadata() -> None:
    mlflow = RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)
    record = TraceRecord(
        trace_id="trace-456",
        request_route="answer_question",
        response_classification="opportunity_estimate",
        policy_fingerprint="policy-abc",
        source_versions={"semantic_version": "1.0.0"},
        tool_outcomes={"retrieval": "success"},
        retrieval_scores=(0.91,),
        evaluation_outcome="not_evaluated",
        latency_ms=12.5,
        response={
            "candidate_causal_factors": [
                {"status": "supported", "sizing_eligible": True},
                {"status": "inconclusive", "sizing_eligible": False},
            ],
            "opportunity_estimate": {"incremental_product_users": 2},
        },
    )

    sink.record(record)

    assert mlflow.tags["trace_delivery_state"] == "attempted"
    assert mlflow.metrics["turn_latency_ms"] == 12.5
    assert mlflow.artifacts["governed_trace.json"]["response"] == {
        "has_canonical_definition": False,
        "has_data_team_verification_request": False,
        "has_direct_identifier_answer": False,
        "has_direct_identifier_audit": False,
        "has_driver_decomposition": False,
        "has_evidence": False,
        "has_metric_definition_gap": False,
        "has_provisional_metric": False,
        "evidence_citation_count": 0,
        "graph_path_count": 0,
        "caveat_count": 0,
        "has_conversation_id": False,
        "has_lead_agent_metadata": False,
        "candidate_factor_count": 2,
        "candidate_factor_statuses": ["inconclusive", "supported"],
        "sizing_eligible_factor_count": 1,
        "opportunity_result": "estimate",
    }


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
                "identifiers": [{"identifier_type": "tenant_id", "value": raw_identifier}]
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


class FailingTraceSink:
    def record(self, trace: TraceRecord) -> None:
        del trace
        raise ConnectionError("MLflow delivery is unavailable")


def test_trace_delivery_failure_preserves_response_and_marks_readiness(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    service = AnswerQuestionService(
        gateway,
        trace_sink=FailingTraceSink(),
        evidence_reranker=DeterministicCrossEncoderReranker(),
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    assert response.status_code == 200
    assert response.json()["canonical_definition"]["name"] == "jira_new_peu"
    assert service.readiness()["trace_delivery"] == {
        "provider": "custom",
        "status": "unavailable",
        "attempt_count": 1,
        "failure_count": 1,
        "last_error_type": "ConnectionError",
    }
    readiness = client.get("/readiness")
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "degraded"


def test_hosted_mlflow_sink_records_hypothesis_and_opportunity_turns(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    mlflow = RecordingMlflow()
    service = AnswerQuestionService(
        gateway,
        trace_sink=MlflowTraceSink(
            tracking_uri="http://go/mlflow",
            mlflow_module=mlflow,
        ),
        evidence_reranker=DeterministicCrossEncoderReranker(),
    )
    client = TestClient(create_app(service))

    hypothesis = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert hypothesis.status_code == 200
    factor_id = hypothesis.json()["candidate_causal_factors"][0]["factor_id"]
    conversation_id = hypothesis.json()["conversation_id"]
    assert mlflow.run_count == 1
    assert mlflow.artifacts["governed_trace.json"]["response"]["candidate_factor_count"] == 1
    assert mlflow.params["config_workflow"] == "governed-response-v1"

    selection = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
            "conversation_id": conversation_id,
            "selected_factor_id": factor_id,
        },
    )

    assert selection.status_code == 200
    assert selection.json()["candidate_causal_factors"][0]["factor_id"] == factor_id
    assert mlflow.run_count == 2

    opportunity = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
            "conversation_id": conversation_id,
            "opportunity_scenario_percentage_points": 5.0,
        },
    )

    assert opportunity.status_code == 200
    assert opportunity.json()["result_classification"] == "opportunity_estimate"
    assert mlflow.run_count == 3
    assert mlflow.artifacts["governed_trace.json"]["response"]["opportunity_result"] == "estimate"


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
                lightrag_adapter=LightRAGEvidenceAdapter(
                    LightRAGBackend(
                        InMemoryLightRAGStore(
                            chunks=[
                                LightRAGChunkRecord(
                                    reference=LightRAGEvidenceReference.from_document(document),
                                    text=document.text,
                                )
                                for document in evidence_corpus()
                            ]
                        )
                    )
                ),
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
    assert trace.tool_spans[-1].name == "evidence_retrieval"
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
    }
    assert [span.name for span in trace.node_spans] == [
        "authorize",
        "intent_interpretation",
        "intent_validation",
        "legacy",
    ]
    assert [span.name for span in trace.tool_spans] == [
        "semantic_driver_decomposition",
        "lightrag_retrieval",
        "evidence_retrieval",
        "graph_traversal",
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
    assert f"trace_id={trace.trace_id}" in response.json()["detail"]
    assert trace.response_classification == "safe_refusal"
    assert trace.source_versions["semantic_version"] == "1.0.0"
    assert trace.tool_outcomes == {
        "semantic_query": "not_used",
        "retrieval": "not_used",
        "graph": "not_used",
        "direct_identifier_audit": "not_used",
    }

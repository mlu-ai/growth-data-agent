from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from runpy import run_path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.evaluation import (
    FixtureResponse,
    LocalModelResult,
    RetrievalResult,
    build_evaluation_report,
    compare_with_baseline,
    evaluate_generation_fixtures,
    evaluate_local_model_fixtures,
    evaluate_retrieval_fixtures,
    load_fixture_catalog,
    record_baseline,
)
from growth_data_agent.main import create_app
from growth_data_agent.principal import development_token_environment_variable
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_invoke = run_path(str(Path(__file__).parents[1] / "scripts/run_evaluations.py"))["_invoke"]


def test_generation_evaluation_checks_observable_governed_response_fields() -> None:
    fixtures = [
        {
            "id": "definition",
            "category": "definition",
            "request": {"agent_user_id": "data_analyst", "question": "Define Jira New PEU"},
            "expected": {
                "status_code": 200,
                "result_classification": "canonical_definition",
                "fields": {"canonical_definition.semantic_version": "1.0.0"},
                "contains": ["first-ever Paid Enablement"],
                "not_contains": ["Causal Estimate"],
            },
        }
    ]

    results = evaluate_generation_fixtures(
        fixtures,
        lambda request: FixtureResponse(
            status_code=200,
            body={
                "answer": "A Product User's first-ever Paid Enablement for Jira.",
                "result_classification": "canonical_definition",
                "canonical_definition": {"semantic_version": "1.0.0"},
            },
        ),
    )

    assert results[0].passed is True
    assert results[0].category == "definition"
    assert results[0].evaluation_category == "governed_response"
    assert results[0].failures == ()


def test_evaluation_invocation_authenticates_without_body_identity(monkeypatch) -> None:
    token = "evaluation-token-" + secrets.token_urlsafe(16)
    monkeypatch.setenv(development_token_environment_variable("data_analyst"), token)

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, str]:
            return {"result_classification": "canonical_definition"}

    class RecordingClient:
        def post(self, url, *, headers, json):
            self.url = url
            self.headers = headers
            self.json = json
            return Response()

    client = RecordingClient()
    result = _invoke(
        {
            "agent_user_id": "data_analyst",
            "question": "Define Jira New PEU",
            "_fixture_id": "definition",
        },
        client,
        client,
    )

    assert result.status_code == 200
    assert client.url == "/answer_question"
    assert client.headers == {"Authorization": f"Bearer {token}"}
    assert client.json == {"question": "Define Jira New PEU"}


def test_retrieval_evaluation_reports_ranking_metrics_without_generation_judgement() -> None:
    fixtures = [
        {
            "id": "apac-incident",
            "category": "hypothesis",
            "expected_document_ids": ["incident"],
            "k": 3,
        }
    ]

    results = evaluate_retrieval_fixtures(
        fixtures,
        lambda fixture: ["incident", "distractor-a", "distractor-b"],
    )

    assert results[0].passed is True
    assert results[0].recall_at_k == 1.0
    assert results[0].precision_at_k == 1 / 3
    assert results[0].reciprocal_rank == 1.0


def test_local_model_evaluation_records_redacted_result_hashes() -> None:
    fixtures = [{"id": "definition", "request": {"question": "Define Jira New PEU"}}]

    results = evaluate_local_model_fixtures(
        fixtures,
        lambda fixture: f"Generated answer for {fixture['id']} with tenant-0011",
    )

    assert results[0].status == "recorded"
    assert results[0].output_sha256
    assert results[0].output_length > 0
    assert "tenant-0011" not in results[0].redacted_output


def test_local_model_evaluation_preserves_governed_trace_link() -> None:
    results = evaluate_local_model_fixtures(
        [{"id": "definition", "trace_id": "trace-123"}],
        lambda fixture: "governed answer",
    )

    assert results[0].trace_id == "trace-123"


def test_empty_fixture_catalog_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "empty-fixtures.json"
    path.write_text(json.dumps({"generation": [], "retrieval": []}))

    with pytest.raises(ValueError, match="must contain generation and retrieval"):
        load_fixture_catalog(path)


def test_baseline_result_records_model_and_separate_scores(tmp_path: Path) -> None:
    report = build_evaluation_report(
        model_name="qwen3:8b",
        generation_results=[],
        retrieval_results=[],
    )
    path = record_baseline(report, tmp_path / "baseline.json", provider="ollama")

    payload = json.loads(path.read_text())
    assert payload["model_name"] == "qwen3:8b"
    assert payload["provider"] == "ollama"
    assert payload["generation"]["fixture_pass_rate"] == 1.0
    assert payload["retrieval"]["recall_at_k"] == 1.0


def test_candidate_model_report_is_compared_with_canonical_baseline(tmp_path: Path) -> None:
    baseline = build_evaluation_report(
        model_name="qwen3:8b",
        generation_results=[],
        retrieval_results=[
            RetrievalResult("retrieval", "hypothesis", True, 1.0, 1.0, 1.0, ("incident",))
        ],
        model_results=[LocalModelResult("definition", "recorded", "baseline-hash", 10, "answer")],
    )
    baseline_path = record_baseline(baseline, tmp_path / "baseline.json", provider="ollama")
    candidate = build_evaluation_report(
        model_name="candidate:8b",
        generation_results=[],
        retrieval_results=[
            RetrievalResult("retrieval", "hypothesis", True, 0.5, 0.5, 0.5, ("distractor",))
        ],
        model_results=[LocalModelResult("definition", "recorded", "candidate-hash", 10, "answer")],
    )

    comparison = compare_with_baseline(candidate, baseline_path)

    assert {item["metric"] for item in comparison["regressions"]} == {
        "retrieval.recall_at_k",
        "retrieval.precision_at_k",
        "retrieval.reciprocal_rank",
        "local_model.output_changed",
    }
    assert comparison["local_model_changes"] == [
        {
            "fixture_id": "definition",
            "baseline_output_sha256": "baseline-hash",
            "current_output_sha256": "candidate-hash",
        }
    ]


def test_fixture_catalog_covers_issue_6_contract() -> None:
    catalog = load_fixture_catalog()

    assert {
        fixture["category"] for fixture in catalog["generation"]
    } >= {
        "definition",
        "driver_decomposition",
        "hypothesis",
        "authorization",
        "identifiers",
        "stale_semantics",
        "unsupported",
    }
    assert catalog["retrieval"][0]["expected_first_document_id"] == (
        "jira-apac-paid-provisioning-incident"
    )
    assert any(
        fixture.get("evaluation_category") == "answer_faithfulness"
        for fixture in catalog["generation"]
    )


def test_generation_catalog_passes_against_the_public_response_seam(client, tmp_path: Path) -> None:
    stale_path = write_artifact(tmp_path / "stale.json")
    stale_artifact = json.loads(stale_path.read_text())
    stale_artifact["validation"]["validated_at"] = "2020-01-01T00:00:00+00:00"
    stale_path.write_text(json.dumps(stale_artifact))
    stale_gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(stale_path),
        metricflow_planner=RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json"),
        postgres_executor=RecordingPostgresExecutor(),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    stale_client = TestClient(create_app(AnswerQuestionService(stale_gateway)))

    def invoke(request: dict) -> FixtureResponse:
        selected_client = stale_client if request.get("requires") == "stale_artifact" else client
        payload = {key: value for key, value in request.items() if key != "requires"}
        response = selected_client.post("/answer_question", json=payload)
        return FixtureResponse(response.status_code, response.json())

    results = evaluate_generation_fixtures(load_fixture_catalog()["generation"], invoke)

    assert all(result.passed for result in results), [
        (result.fixture_id, result.failures) for result in results if not result.passed
    ]

"""Run local deterministic retrieval and governed-response evaluations."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from growth_data_agent.evaluation import (
    FixtureResponse,
    build_evaluation_report,
    compare_with_baseline,
    evaluate_generation_fixtures,
    evaluate_local_model_fixtures,
    evaluate_retrieval_fixtures,
    load_fixture_catalog,
    record_baseline,
)
from growth_data_agent.local_model import (
    OllamaBaselineModel,
    build_local_model_baseline_context,
)
from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import MetricFlowPlanner, PostgresMetricFlowExecutor
from growth_data_agent.observability import (
    MlflowTraceSink,
    TraceRecord,
    policy_fingerprint,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"


def main() -> int:
    catalog = load_fixture_catalog()
    base_client, service = _client(_ARTIFACT)
    stale_client = _stale_client()
    retrieval_trace_ids: dict[str, str] = {}

    retrieval_results = evaluate_retrieval_fixtures(
        catalog["retrieval"],
        lambda fixture: _retrieve(service, fixture, retrieval_trace_ids),
    )
    governed_bodies: dict[str, dict] = {}
    generation_fixtures = []
    for fixture in catalog["generation"]:
        generation_fixture = dict(fixture)
        generation_fixture["request"] = {
            **fixture["request"],
            "_fixture_id": fixture["id"],
        }
        generation_fixtures.append(generation_fixture)

    def invoke(request: dict) -> FixtureResponse:
        response = _invoke(request, base_client, stale_client)
        governed_bodies[request["_fixture_id"]] = response.body
        return response

    generation_results = evaluate_generation_fixtures(
        generation_fixtures,
        invoke,
    )
    model_name = os.environ.get("LOCAL_MODEL_NAME", "qwen3:8b")
    model_fixtures = [
        {
            "id": fixture["id"],
            "trace_id": next(
                result.trace_id
                for result in generation_results
                if result.fixture_id == fixture["id"]
            ),
            "governed_context": build_local_model_baseline_context(
                _stable_governed_context(governed_bodies[fixture["id"]])
            ),
        }
        for fixture in catalog["generation"]
    ]
    local_model = OllamaBaselineModel(
        model_name=model_name,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    model_results = evaluate_local_model_fixtures(
        model_fixtures,
        lambda fixture: local_model.generate(fixture["governed_context"]),
    )
    report = build_evaluation_report(
        model_name=model_name,
        generation_results=generation_results,
        retrieval_results=retrieval_results,
        model_results=model_results,
    )
    canonical_baseline_path = _REPOSITORY_ROOT / "evaluations/baseline.json"
    baseline_path = _output_path(model_name, canonical_baseline_path)
    comparison = (
        compare_with_baseline(report, canonical_baseline_path)
        if baseline_path != canonical_baseline_path and canonical_baseline_path.exists()
        else None
    )
    provider = os.environ.get("LOCAL_MODEL_PROVIDER", "ollama")
    local_model_status = report.as_baseline(provider=provider)["local_model"]["status"]
    if baseline_path == canonical_baseline_path and (
        local_model_status != "recorded" or not report.passed
    ):
        raise SystemExit(
            "Refusing to write the canonical baseline because the evaluation report "
            f"passed={report.passed} and the local model result is {local_model_status}."
        )
    record_baseline(report, baseline_path, provider=provider, comparison=comparison)
    _record_evaluation_traces(
        service.trace_sink,
        generation_results,
        retrieval_results,
        model_results,
        model_name,
        comparison,
        retrieval_trace_ids,
    )

    print(json.dumps(report.as_baseline(provider=provider), indent=2))
    return 0 if report.passed and not (comparison and comparison["regressions"]) else 1


def _client(artifact_path: Path) -> tuple[TestClient, AnswerQuestionService]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data",
    )
    if not _MANIFEST.exists():
        raise SystemExit("Missing dbt/target/semantic_manifest.json; run make dbt-build first.")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=MetricFlowPlanner(_MANIFEST),
        postgres_executor=PostgresMetricFlowExecutor(database_url),
    )
    service = AnswerQuestionService(
        gateway,
        trace_sink=MlflowTraceSink.from_environment(),
    )
    return TestClient(create_app(service)), service


def _stale_client() -> TestClient:
    directory = tempfile.TemporaryDirectory(prefix="growth-data-agent-evaluation-")
    path = Path(directory.name) / "stale-semantic.json"
    artifact = json.loads(_ARTIFACT.read_text())
    artifact["validation"]["validated_at"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(artifact))
    client, _ = _client(path)
    # Keep the artifact alive for the duration of the evaluation run.
    client._evaluation_tempdir = directory
    return client


def _invoke(request: dict, base_client: TestClient, stale_client: TestClient) -> FixtureResponse:
    client = stale_client if request.get("requires") == "stale_artifact" else base_client
    payload = {
        key: value for key, value in request.items() if key not in {"requires", "_fixture_id"}
    }
    response = client.post("/answer_question", json=payload)
    return FixtureResponse(status_code=response.status_code, body=response.json())


def _retrieve(
    service: AnswerQuestionService,
    fixture: dict,
    retrieval_trace_ids: dict[str, str],
) -> list[str]:
    profile = resolve_access_profile(fixture["agent_user_id"])
    access_filter = profile.evidence_filter(fixture["product"], fixture["region"])
    documents = service.evidence_store.retrieve(
        fixture["query"], access_filter, limit=int(fixture.get("k", 3))
    )
    document_ids = [document.document_id for document in documents]
    trace_id = str(uuid4())
    retrieval_trace_ids[str(fixture["id"])] = trace_id
    if isinstance(service.trace_sink, MlflowTraceSink):
        artifact = service.semantic_gateway.artifact_store.load()
        service.trace_sink.record(
            TraceRecord(
                trace_id=trace_id,
                request_route="retrieval_evaluation",
                response_classification="retrieval",
                policy_fingerprint=policy_fingerprint(profile),
                source_versions={
                    "semantic_version": artifact.semantic_version,
                    "semantic_manifest_sha256": artifact.semantic_manifest_sha256,
                    "evidence_corpus": "synthetic-v1",
                },
                tool_outcomes={"retrieval": "success"},
                retrieval_scores=tuple(
                    float(score)
                    for score in getattr(service.evidence_store, "last_scores", ())
                ),
                evaluation_outcome="not_evaluated",
                response={"retrieved_document_ids": document_ids},
            )
        )
    return document_ids


def _record_evaluation_traces(
    trace_sink,
    generation_results,
    retrieval_results,
    model_results,
    model_name: str,
    comparison: dict | None,
    retrieval_trace_ids: dict[str, str],
) -> None:
    if not isinstance(trace_sink, MlflowTraceSink):
        return
    for result in generation_results:
        if result.trace_id is None:
            raise RuntimeError(
                f"Generation fixture {result.fixture_id} has no governed trace ID."
            )
        trace_sink.record_evaluation(
            trace_id=result.trace_id,
            fixture_id=result.fixture_id,
            category=result.evaluation_category,
            model_name=model_name,
            passed=result.passed,
        )
    for result in retrieval_results:
        trace_sink.record_evaluation(
            trace_id=retrieval_trace_ids[result.fixture_id],
            fixture_id=result.fixture_id,
            category="retrieval",
            model_name=model_name,
            passed=result.passed,
            metrics={
                "recall_at_k": result.recall_at_k,
                "precision_at_k": result.precision_at_k,
                "reciprocal_rank": result.reciprocal_rank,
            },
        )
    for result in model_results:
        if result.trace_id is None:
            raise RuntimeError(
                f"Local-model fixture {result.fixture_id} has no governed trace ID."
            )
        changed = bool(
            comparison
            and any(
                item["fixture_id"] == result.fixture_id
                for item in comparison["local_model_changes"]
            )
        )
        trace_sink.record_evaluation(
            trace_id=result.trace_id,
            fixture_id=result.fixture_id,
            category="local_model",
            model_name=model_name,
            passed=result.status == "recorded" and not changed,
            metrics={"output_length": float(result.output_length)},
        )


def _output_path(model_name: str, canonical_path: Path) -> Path:
    configured = os.environ.get("BASELINE_EVALUATION_PATH")
    if configured:
        return Path(configured)
    if not canonical_path.exists():
        return canonical_path
    safe_name = "".join(character if character.isalnum() else "-" for character in model_name)
    return _REPOSITORY_ROOT / "evaluations/comparisons" / f"{safe_name}-candidate.json"


def _stable_governed_context(body: dict) -> dict:
    """Remove per-request audit IDs before using a response as model input."""
    if isinstance(body, dict):
        return {
            key: _stable_governed_context(value)
            for key, value in body.items()
            if key not in {"trace_id", "audit_event_id", "request_id"}
        }
    if isinstance(body, list):
        return [_stable_governed_context(value) for value in body]
    if isinstance(body, str):
        return re.sub(r"\s*\(trace_id=[a-f0-9-]+\)", "", body)
    return body


if __name__ == "__main__":
    sys.exit(main())

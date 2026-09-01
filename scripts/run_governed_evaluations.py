"""Run the Governed Evaluation Dataset (issue #84) against a real deployment
shape and publish an Evaluation Scorecard.

This is the real-infrastructure counterpart to
`tests/test_evaluation_runner.py` — it needs a live Postgres with a
successful `make dbt-build` behind it, matching `scripts/run_evaluations.py`'s
own prerequisites. It reuses the same `run_dataset(...)` orchestrator the
tests exercise against fakes; only the `client_factory` differs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from growth_data_agent.evaluation_dataset import EvaluationDatasetStore
from growth_data_agent.evaluation_runner import EVALUATOR_VERSION, run_dataset
from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import MetricFlowPlanner, PostgresMetricFlowExecutor
from growth_data_agent.observability import MlflowTraceSink
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/dataset/v1/cases.json"


def _client(sink: MlflowTraceSink) -> tuple[TestClient, AnswerQuestionService]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data",
    )
    if not _MANIFEST.exists():
        raise SystemExit("Missing dbt/target/semantic_manifest.json; run make dbt-build first.")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(_ARTIFACT),
        metricflow_planner=MetricFlowPlanner(_MANIFEST),
        postgres_executor=PostgresMetricFlowExecutor(database_url),
    )
    service = AnswerQuestionService(
        gateway,
        evidence_reranker=DeterministicCrossEncoderReranker(),
        trace_sink=sink,
    )
    return TestClient(create_app(service)), service


def _source_versions() -> dict[str, str]:
    if not _ARTIFACT.exists():
        return {}
    artifact = json.loads(_ARTIFACT.read_text())
    return {"semantic_version": str(artifact.get("semantic_version", "unknown"))}


def main() -> None:
    dataset = EvaluationDatasetStore(_DATASET_PATH).load()
    sink = MlflowTraceSink.from_environment()
    scorecard = run_dataset(
        dataset,
        lambda: _client(sink),
        evaluator_version=EVALUATOR_VERSION,
        source_versions=_source_versions(),
    )

    print(
        f"Governed Evaluation Dataset v{scorecard.dataset_version}: "
        f"{scorecard.automated_cases} automated, "
        f"{scorecard.not_yet_automated_cases} not yet automated "
        f"(of {scorecard.total_cases} total)."
    )
    for category in (
        scorecard.safety,
        scorecard.semantic_correctness,
        scorecard.trace_delivery,
    ):
        print(
            f"  {category.name}: {category.passed}/{category.total} passed "
            f"({category.pass_rate:.1%})"
        )
        for detail in category.details:
            print(f"    - {detail}")
    print(f"  latency_ms: {scorecard.latency_ms}")
    print(f"  token_cost: {scorecard.token_cost}")

    sink.record_scorecard(scorecard)
    print("Published the Evaluation Scorecard to MLflow.")


if __name__ == "__main__":
    main()

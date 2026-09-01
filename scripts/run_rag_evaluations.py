"""Run the RAG Evaluation Dataset (issue #86) against a real deployment shape
and publish a RAG Evaluation Scorecard.

Retrieval quality and generation quality are scored independently: retrieval
goes straight through `evidence_store.retrieve(...)` (the same authorized
seam `scripts/run_evaluations.py`'s retrieval fixture already uses);
generation goes through the full `POST /answer_question` seam so the judged
answer is the real governed response, not a bypassed shortcut. Needs the
same live Postgres + `make dbt-build` prerequisites as
`scripts/run_governed_evaluations.py`. A RAGAS judge is optional — set
`RAGAS_JUDGE_MODEL_NAME` (an Ollama model available at `RAGAS_JUDGE_BASE_URL`,
default `http://127.0.0.1:11434/v1`) to score generation for real; otherwise
generation is honestly reported `not_configured`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from growth_data_agent.evidence import EvidenceDocument
from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import MetricFlowPlanner, PostgresMetricFlowExecutor
from growth_data_agent.observability import MlflowTraceSink
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.principal import development_token_environment_variable
from growth_data_agent.rag_evaluation import (
    CHUNKING_STRATEGY_VERSION,
    EVALUATOR_VERSION,
    RagJudge,
    run_rag_dataset,
)
from growth_data_agent.rag_evaluation_dataset import RagEvaluationCase, RagEvaluationDatasetStore
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/rag_dataset/v1/cases.json"


def _service(sink: MlflowTraceSink) -> AnswerQuestionService:
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
    return AnswerQuestionService(
        gateway,
        evidence_reranker=DeterministicCrossEncoderReranker(),
        trace_sink=sink,
    )


def _retrieve(service: AnswerQuestionService, case: RagEvaluationCase) -> list[EvidenceDocument]:
    profile = resolve_access_profile(case.agent_user_id)
    access_filter = profile.evidence_filter(case.product, case.region)
    return service.evidence_store.retrieve(case.retrieval_query, access_filter, limit=case.k)


def _answer(client: TestClient, case: RagEvaluationCase) -> tuple[str, list[str]]:
    token = os.environ.get(development_token_environment_variable(case.agent_user_id))
    if not token:
        raise SystemExit(
            f"Missing development bearer token configuration for {case.agent_user_id!r}."
        )
    response = client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {token}"},
        json={"agent_user_id": case.agent_user_id, "question": case.question},
    )
    body = response.json()
    answer_text = str(body.get("answer", ""))
    evidence_chain = body.get("evidence_chain") or {}
    contexts = [chunk["text"] for chunk in evidence_chain.get("supporting_chunks", [])]
    if not contexts:
        citations = (body.get("evidence") or {}).get("citations", [])
        contexts = [citation["support_explanation"] for citation in citations]
    return answer_text, contexts


def _configuration_versions(
    service: AnswerQuestionService, sample_document: EvidenceDocument | None
) -> dict[str, str]:
    versions = {"chunking_strategy": CHUNKING_STRATEGY_VERSION}
    reranker = service.evidence_reranker
    if reranker is not None:
        versions["reranker_model"] = reranker.model_name
        versions["reranker_version"] = reranker.model_version
    if sample_document is not None and sample_document.embedding_model:
        versions["embedding_model"] = sample_document.embedding_model
        versions["embedding_version"] = sample_document.embedding_version or "unknown"
    return versions


def main() -> int:
    dataset = RagEvaluationDatasetStore(_DATASET_PATH).load()
    sink = MlflowTraceSink.from_environment()
    service = _service(sink)
    client = TestClient(create_app(service))
    judge = RagJudge.from_environment()

    sample_documents = _retrieve(service, dataset.cases[0])
    sample_document = sample_documents[0] if sample_documents else None

    scorecard = run_rag_dataset(
        dataset,
        retrieve=lambda case: _retrieve(service, case),
        answer=lambda case: _answer(client, case),
        judge=judge,
        evaluator_version=EVALUATOR_VERSION,
        configuration_versions=_configuration_versions(service, sample_document),
    )

    print(f"RAG Evaluation Dataset v{scorecard.dataset_version}:")
    for category in (scorecard.retrieval, scorecard.generation):
        print(
            f"  {category.name}: {category.passed}/{category.total} passed "
            f"({category.pass_rate:.1%})"
        )
        for detail in category.details:
            print(f"    - {detail}")
    print(f"  configuration_versions: {scorecard.configuration_versions}")
    if judge is None:
        print("  No RAGAS judge configured (set RAGAS_JUDGE_MODEL_NAME to enable one).")

    sink.record_rag_scorecard(scorecard)
    print("Published the RAG Evaluation Scorecard to MLflow.")

    return 0 if scorecard.retrieval.failed == 0 and scorecard.generation.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

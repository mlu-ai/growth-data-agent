"""Tests for the governed evaluation runner (issue #85).

Unit tests exercise each deterministic evaluator directly (pass and fail
cases — proving failures are actually detected, not silently passed).
Integration tests exercise the full `run_dataset` orchestrator through a
fake-backed `/answer_question` seam, matching this repo's existing
`RecordingMetricFlowPlanner`/`RecordingPostgresExecutor` test-double pattern.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.contracts import ResultClassification
from growth_data_agent.evaluation_dataset import EvaluationDatasetStore
from growth_data_agent.evaluation_runner import (
    EVALUATOR_VERSION,
    evaluate_active_investigation_reauthorization,
    evaluate_artifact_freshness,
    evaluate_authorization,
    evaluate_candidate_causal_factor_status,
    evaluate_citation_revision_validity,
    evaluate_driver_decomposition_arithmetic,
    evaluate_opportunity_estimate_formula,
    evaluate_refusal,
    evaluate_semantic_provenance,
    evaluate_tool_policy_arguments,
    run_dataset,
)
from growth_data_agent.main import create_app
from growth_data_agent.observability import MlflowTraceSink
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/dataset/v1/cases.json"


def _client_factory(tmp_path: Path):
    def factory():
        artifact_path = write_artifact(tmp_path / "semantic.json")
        planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
        executor = RecordingPostgresExecutor()
        gateway = ValidatedMetricFlowGateway(
            SemanticArtifactStore(artifact_path),
            metricflow_planner=planner,
            postgres_executor=executor,
            now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
        )
        service = AnswerQuestionService(
            gateway, evidence_reranker=DeterministicCrossEncoderReranker()
        )
        return TestClient(create_app(service)), service

    return factory


# --- Evaluator unit tests -----------------------------------------------


def test_evaluate_authorization_passes_and_fails() -> None:
    ok = evaluate_authorization(
        {"effective_access_scope": {"products": ["Jira"], "regions": ["APAC"], "tenant_scope": "x"}}
    )
    assert ok is not None and ok.passed
    bad = evaluate_authorization(
        {"effective_access_scope": {"products": [], "regions": ["APAC"], "tenant_scope": "x"}}
    )
    assert bad is not None and not bad.passed
    assert evaluate_authorization({}) is None


def test_evaluate_refusal_passes_and_fails() -> None:
    ok = evaluate_refusal({"result_classification": "safe_refusal"}, 200)
    assert ok is not None and ok.passed
    leaked = evaluate_refusal(
        {"result_classification": "safe_refusal", "candidate_causal_factors": [{"factor_id": "x"}]},
        200,
    )
    assert leaked is not None and not leaked.passed
    assert evaluate_refusal({"result_classification": "hypothesis"}, 200) is None


def test_evaluate_semantic_provenance_passes_and_fails() -> None:
    ok = evaluate_semantic_provenance(
        {
            "canonical_definition": {
                "semantic_version": "1.0.0",
                "citation": {"authority": "dbt/MetricFlow", "artifact_path": "a.yml#m"},
            }
        }
    )
    assert ok is not None and ok.passed
    bad = evaluate_semantic_provenance(
        {"canonical_definition": {"semantic_version": "1.0.0", "citation": {"authority": "llm"}}}
    )
    assert bad is not None and not bad.passed
    assert evaluate_semantic_provenance({}) is None


def test_evaluate_artifact_freshness_passes_fails_and_exempts_safe_refusal() -> None:
    ok = evaluate_artifact_freshness({"source_freshness": {"is_current": True}})
    assert ok is not None and ok.passed
    bad = evaluate_artifact_freshness(
        {"source_freshness": {"is_current": False}, "result_classification": "hypothesis"}
    )
    assert bad is not None and not bad.passed
    stale_but_correct = evaluate_artifact_freshness(
        {"source_freshness": {"is_current": False}, "result_classification": "limitation"}
    )
    assert stale_but_correct is not None and stale_but_correct.passed
    # A safe_refusal's is_current=False is a placeholder (no semantic check ran),
    # not a genuine staleness signal — the invariant must not apply here.
    refusal_placeholder = evaluate_artifact_freshness(
        {"source_freshness": {"is_current": False}, "result_classification": "safe_refusal"}
    )
    assert refusal_placeholder is None


def test_evaluate_driver_decomposition_arithmetic_passes_and_fails() -> None:
    ok = evaluate_driver_decomposition_arithmetic({"driver_decomposition": {"residual": 0}})
    assert ok is not None and ok.passed
    bad = evaluate_driver_decomposition_arithmetic({"driver_decomposition": {"residual": 40}})
    assert bad is not None and not bad.passed
    # A single declining segment offset by rising segments can legitimately
    # report > 100% of the net decline — that alone must not fail the check.
    offsetting = evaluate_driver_decomposition_arithmetic(
        {
            "driver_decomposition": {
                "residual": 0,
                "decline": 276,
                "contributions": [{"percentage_of_decline": 108.7}],
            }
        }
    )
    assert offsetting is not None and offsetting.passed
    assert evaluate_driver_decomposition_arithmetic({}) is None


def test_evaluate_citation_revision_validity_passes_and_fails() -> None:
    ok = evaluate_citation_revision_validity(
        {
            "evidence": {
                "citations": [
                    {
                        "document_id": "d1",
                        "source_revision": "1",
                        "source_document_id": "d1",
                        "source_url": "https://x",
                    }
                ]
            }
        }
    )
    assert ok is not None and ok.passed
    bad = evaluate_citation_revision_validity(
        {"evidence": {"citations": [{"document_id": "d1", "source_revision": ""}]}}
    )
    assert bad is not None and not bad.passed
    assert evaluate_citation_revision_validity({"evidence": {"citations": []}}) is None


def test_evaluate_candidate_causal_factor_status_passes_and_fails() -> None:
    ok = evaluate_candidate_causal_factor_status(
        {
            "candidate_causal_factors": [
                {
                    "factor_id": "f1",
                    "status": "supported",
                    "ranking_signals": {"independent_source_count": 2, "counterevidence": "none"},
                }
            ]
        }
    )
    assert ok is not None and ok.passed
    bad = evaluate_candidate_causal_factor_status(
        {
            "candidate_causal_factors": [
                {
                    "factor_id": "f1",
                    "status": "contradicted",
                    "ranking_signals": {"independent_source_count": 1, "counterevidence": "none"},
                }
            ]
        }
    )
    assert bad is not None and not bad.passed
    assert evaluate_candidate_causal_factor_status({"candidate_causal_factors": []}) is None


def test_evaluate_opportunity_estimate_formula_passes_and_fails() -> None:
    ok = evaluate_opportunity_estimate_formula(
        {
            "opportunity_estimate": {
                "eligible_population": 40,
                "scenario_percentage_point_change": 5.0,
                "incremental_product_users": 2,
            }
        }
    )
    assert ok is not None and ok.passed
    bad = evaluate_opportunity_estimate_formula(
        {
            "opportunity_estimate": {
                "eligible_population": 40,
                "scenario_percentage_point_change": 5.0,
                "incremental_product_users": 99,
            }
        }
    )
    assert bad is not None and not bad.passed
    assert evaluate_opportunity_estimate_formula({}) is None


def test_evaluate_tool_policy_arguments_passes_and_fails() -> None:
    ok = evaluate_tool_policy_arguments(
        {"lead_agent_metadata": {"tool_outcomes": [{"action": "metricflow", "status": "success"}]}}
    )
    assert ok is not None and ok.passed
    bad = evaluate_tool_policy_arguments(
        {"lead_agent_metadata": {"tool_outcomes": [{"action": "unknown_tool", "status": "ok"}]}}
    )
    assert bad is not None and not bad.passed
    assert evaluate_tool_policy_arguments({}) is None


def test_evaluate_active_investigation_reauthorization_passes_and_fails() -> None:
    ok = evaluate_active_investigation_reauthorization(["a", "b", "c"])
    assert ok is not None and ok.passed
    bad = evaluate_active_investigation_reauthorization(["a", "a"])
    assert bad is not None and not bad.passed
    assert evaluate_active_investigation_reauthorization(["a"]) is None


# --- Integration tests ----------------------------------------------------


def test_run_dataset_executes_the_automatable_subset(tmp_path: Path) -> None:
    dataset = EvaluationDatasetStore(_DATASET_PATH).load()

    scorecard = run_dataset(dataset, _client_factory(tmp_path))

    assert scorecard.dataset_version == dataset.dataset_version
    assert scorecard.evaluator_version == EVALUATOR_VERSION
    assert scorecard.total_cases == len(dataset.cases)
    assert scorecard.automated_cases > 0
    assert scorecard.not_yet_automated_cases > 0
    assert scorecard.automated_cases + scorecard.not_yet_automated_cases == scorecard.total_cases
    for category in (scorecard.safety, scorecard.semantic_correctness, scorecard.trace_delivery):
        assert category.total > 0
        assert category.failed == 0, f"{category.name} had unexpected failures: {category.details}"
    assert scorecard.latency_ms["count"] > 0
    assert scorecard.token_cost["total_tokens"] == 0


def test_run_dataset_detects_a_deliberate_expected_behavior_failure(tmp_path: Path) -> None:
    dataset = EvaluationDatasetStore(_DATASET_PATH).load()
    target = next(
        case
        for case in dataset.cases
        if case.case_id == "definition-jira-new-peu-full"
    )
    broken_turn = target.turns[0].model_copy(
        update={
            "expected": target.turns[0].expected.model_copy(
                update={"result_classification": ResultClassification.LIMITATION}
            )
        }
    )
    broken_case = target.model_copy(update={"turns": [broken_turn]})
    updated_cases = [
        broken_case if case.case_id == target.case_id else case for case in dataset.cases
    ]
    broken_dataset = dataset.model_copy(update={"cases": updated_cases})

    scorecard = run_dataset(broken_dataset, _client_factory(tmp_path))

    assert scorecard.semantic_correctness.failed >= 1
    assert any(
        "expected_behavior" in detail for detail in scorecard.semantic_correctness.details
    )


class _RecordingMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.run_names: list[str] = []

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    @contextmanager
    def start_run(self, **kwargs):
        self.run_names.append(kwargs.get("run_name", ""))
        yield self

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_params(self, values: dict) -> None:
        self.params.update({str(k): str(v) for k, v in values.items()})

    def log_param(self, key: str, value: str) -> None:
        self.params[key] = value

    def log_metrics(self, values: dict) -> None:
        self.metrics.update(values)

    def log_dict(self, value: dict, artifact_file: str) -> None:
        pass


def test_record_scorecard_publishes_safely_and_separately(tmp_path: Path) -> None:
    dataset = EvaluationDatasetStore(_DATASET_PATH).load()
    scorecard = run_dataset(dataset, _client_factory(tmp_path))

    mlflow = _RecordingMlflow()
    sink = MlflowTraceSink(mlflow_module=mlflow)

    sink.record_scorecard(scorecard)

    assert mlflow.tags["dataset_version"] == scorecard.dataset_version
    assert mlflow.tags["evaluator_version"] == EVALUATOR_VERSION
    assert mlflow.metrics["total_cases"] == float(scorecard.total_cases)
    assert mlflow.metrics["safety_pass_rate"] == scorecard.safety.pass_rate
    # One run for the scorecard, distinct from any per-request governed trace run.
    assert mlflow.run_names == [
        f"scorecard-{scorecard.dataset_version}-{scorecard.generated_at.isoformat()}"
    ]
    serialized_payload = str(mlflow.tags) + str(mlflow.params) + str(mlflow.metrics)
    for case in dataset.cases:
        assert case.provenance.source_reference not in serialized_payload

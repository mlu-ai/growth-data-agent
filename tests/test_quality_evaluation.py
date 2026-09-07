from __future__ import annotations

from pathlib import Path

import pytest

from growth_data_agent.evaluation_dataset import EvaluationDatasetStore
from growth_data_agent.quality_evaluation import (
    EvaluatorKind,
    EvaluatorMetadata,
    approve_calibration,
    calibrate_quality_evaluator,
    compare_quality_baseline,
    make_llm_judge_evaluator,
    quality_gate_decision,
    record_calibration,
)

_DATASET_PATH = Path(__file__).parents[1] / "evaluations/dataset/v1/cases.json"


def _dataset():
    return EvaluationDatasetStore(_DATASET_PATH).load()


def _candidate_outputs(dataset):
    return {
        case.case_id: {"quality": "meets"}
        for case in dataset.cases
        if case.split.value == "held_out"
    }


def test_calibration_evaluates_only_untouched_held_out_cases() -> None:
    dataset = _dataset()
    called_case_ids: list[str] = []

    def score(case, candidate_output):
        called_case_ids.append(case.case_id)
        assert candidate_output == {"quality": "meets"}
        return {criterion: candidate_output["quality"] for criterion in case.criteria}

    evaluator = make_llm_judge_evaluator(
        score,
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )

    result = calibrate_quality_evaluator(
        dataset, evaluator, candidate_outputs=_candidate_outputs(dataset)
    )

    expected_case_ids = [case.case_id for case in dataset.cases if case.split.value == "held_out"]
    assert called_case_ids == expected_case_ids
    assert result.case_count == len(expected_case_ids)
    assert result.comparison_count > 0
    assert result.metadata.kind is EvaluatorKind.LLM_JUDGE


def test_calibration_requires_candidate_output_for_every_held_out_case() -> None:
    dataset = _dataset()
    evaluator = make_llm_judge_evaluator(
        lambda _case, _candidate: {},
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )

    with pytest.raises(ValueError, match="Missing candidate output"):
        calibrate_quality_evaluator(dataset, evaluator, candidate_outputs={})


def test_calibration_records_identity_uncertainty_and_disagreements(tmp_path: Path) -> None:
    dataset = _dataset()
    first_held_out = next(case for case in dataset.cases if case.split.value == "held_out")
    candidate_outputs = _candidate_outputs(dataset)
    candidate_outputs[first_held_out.case_id]["correctness"] = "fails"

    def score(case, candidate_output):
        return {
            criterion: candidate_output.get(criterion, candidate_output["quality"])
            for criterion in case.criteria
        }

    evaluator = make_llm_judge_evaluator(
        score,
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )

    result = calibrate_quality_evaluator(
        dataset, evaluator, candidate_outputs=candidate_outputs
    )
    payload = result.as_dict()

    assert payload["metadata"] == {
        "name": "llm_judge",
        "kind": "llm_judge",
        "provider": "ollama",
        "model": "judge:8b",
        "prompt_version": "grounded-quality-v1",
        "evaluator_version": "judge-evaluator-v1",
        "configuration_version": "judge:8b|grounded-quality-v1",
    }
    assert result.agreement_rate < 1.0
    assert result.uncertainty.comparison_count == result.comparison_count
    assert result.disagreement_count == 1
    assert result.disagreements[0].case_id == first_held_out.case_id
    assert result.disagreements[0].criterion == "correctness"
    assert result.quality_gate_status == "report_only"
    assert result.dataset_version == dataset.dataset_version

    path = record_calibration(result, tmp_path / "calibration.json")
    assert path.exists()
    assert '"disagreement_count": 1' in path.read_text()


def test_uncalibrated_judge_is_report_only_even_when_baseline_regresses() -> None:
    dataset = _dataset()
    evaluator = make_llm_judge_evaluator(
        lambda case, _candidate: {criterion: "meets" for criterion in case.criteria},
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )
    calibration = calibrate_quality_evaluator(
        dataset, evaluator, candidate_outputs=_candidate_outputs(dataset)
    )
    comparison = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8},
        configuration_version="judge:8b|grounded-quality-v1",
        baseline_configuration_version="judge:8b|grounded-quality-v1",
        baseline_status="approved",
        baseline_approval={
            "approver": "quality-owner",
            "approval_reference": "baseline-88",
        },
    )

    decision = quality_gate_decision(calibration, comparison)

    assert comparison.regressions == ("faithfulness",)
    assert decision.status == "report_only"
    assert not decision.blocking
    assert "not calibrated" in decision.reason


def test_explicit_calibration_approval_is_required_for_a_blocking_quality_gate() -> None:
    dataset = _dataset()
    evaluator = make_llm_judge_evaluator(
        lambda case, _candidate: {criterion: "meets" for criterion in case.criteria},
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )
    calibration = calibrate_quality_evaluator(
        dataset, evaluator, candidate_outputs=_candidate_outputs(dataset)
    )
    approved = approve_calibration(
        calibration,
        approver="quality-owner",
        approval_reference="approval-88",
    )
    comparison = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8},
        configuration_version="judge:8b|grounded-quality-v1",
        baseline_configuration_version="judge:8b|grounded-quality-v1",
        baseline_status="approved",
        baseline_approval={
            "approver": "quality-owner",
            "approval_reference": "baseline-88",
        },
    )

    decision = quality_gate_decision(approved, comparison)

    assert decision.status == "blocking"
    assert decision.blocking

    mismatched = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8},
        configuration_version="judge:8b|other-prompt",
        baseline_configuration_version="judge:8b|other-prompt",
    )
    assert quality_gate_decision(approved, mismatched).status == "report_only"


def test_quality_gate_requires_an_explicitly_approved_baseline() -> None:
    dataset = _dataset()
    evaluator = make_llm_judge_evaluator(
        lambda case, _candidate: {criterion: "meets" for criterion in case.criteria},
        provider="ollama",
        model="judge:8b",
        prompt_version="grounded-quality-v1",
        evaluator_version="judge-evaluator-v1",
        configuration_version="judge:8b|grounded-quality-v1",
    )
    calibration = calibrate_quality_evaluator(
        dataset, evaluator, candidate_outputs=_candidate_outputs(dataset)
    )
    approved = approve_calibration(
        calibration,
        approver="quality-owner",
        approval_reference="approval-88",
    )
    comparison = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8},
        configuration_version="judge:8b|grounded-quality-v1",
        baseline_configuration_version="judge:8b|grounded-quality-v1",
    )

    decision = quality_gate_decision(approved, comparison)

    assert decision.status == "report_only"
    assert not decision.blocking
    assert "not approved" in decision.reason


def test_quality_baseline_compares_each_metric_and_rejects_configuration_drift() -> None:
    comparison = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7, "answer_relevance": 0.9},
        baseline_metrics={"faithfulness": 0.8, "answer_relevance": 0.8},
        configuration_version="judge:8b|prompt-v2",
        baseline_configuration_version="judge:8b|prompt-v2",
    )

    assert comparison.comparable
    assert comparison.regressions == ("faithfulness",)
    assert comparison.metric_comparisons["faithfulness"].current == 0.7
    assert comparison.metric_comparisons["answer_relevance"].regressed is False

    drifted = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8},
        configuration_version="judge:8b|prompt-v3",
        baseline_configuration_version="judge:8b|prompt-v2",
    )

    assert not drifted.comparable
    assert drifted.regressions == ()
    assert drifted.configuration_mismatch

    missing_metric = compare_quality_baseline(
        current_metrics={"faithfulness": 0.7},
        baseline_metrics={"faithfulness": 0.8, "answer_relevance": 0.8},
        configuration_version="judge:8b|prompt-v2",
        baseline_configuration_version="judge:8b|prompt-v2",
    )
    assert not missing_metric.comparable
    assert missing_metric.missing_current_metrics == ("answer_relevance",)


def test_evaluator_metadata_can_describe_deterministic_reference_based_evaluators() -> None:
    metadata = EvaluatorMetadata(
        name="retrieval_recall",
        kind=EvaluatorKind.REFERENCE_BASED,
        provider="deterministic",
        model="not_applicable",
        prompt_version="not_applicable",
        evaluator_version="retrieval-evaluator-v1",
        configuration_version="retrieval-k3-v1",
    )

    assert metadata.as_dict()["kind"] == "reference_based"

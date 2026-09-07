"""Calibrated quality evaluators and configuration-scoped baselines.

Quality evaluators are evidence for review, not policy by default.  The
calibration boundary deliberately consumes only the dataset's untouched
``held_out`` split and keeps deterministic safety evaluators separate from
quality-judge results.  A judge becomes eligible for a blocking quality gate
only after an explicit approval is recorded for its calibration result.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from .evaluation_dataset import EvaluationCase, EvaluationSplit, GovernedEvaluationDataset


class EvaluatorKind(StrEnum):
    REFERENCE_BASED = "reference_based"
    REFERENCE_FREE = "reference_free"
    LLM_JUDGE = "llm_judge"


@dataclass(frozen=True)
class EvaluatorMetadata:
    """Reproducibility identity for one quality evaluator configuration."""

    name: str
    kind: EvaluatorKind
    provider: str
    model: str
    prompt_version: str
    evaluator_version: str
    configuration_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "evaluator_version": self.evaluator_version,
            "configuration_version": self.configuration_version,
        }


QualityLabels = Mapping[str, str]


@dataclass(frozen=True)
class CalibrationCase:
    """The non-label, non-reference portion exposed to an evaluator."""

    case_id: str
    category: str
    split: EvaluationSplit
    permitted_scope: str
    primary_error_taxonomy: str
    criteria: tuple[str, ...]


QualityScorer = Callable[[CalibrationCase, Any], QualityLabels]


@dataclass(frozen=True)
class QualityEvaluator:
    metadata: EvaluatorMetadata
    score: QualityScorer

    def evaluate(self, case: CalibrationCase, candidate_output: Any) -> QualityLabels:
        return self.score(case, candidate_output)


def make_reference_based_evaluator(
    score: QualityScorer,
    *,
    evaluator_version: str,
    configuration_version: str,
    name: str = "reference_based",
) -> QualityEvaluator:
    return QualityEvaluator(
        metadata=EvaluatorMetadata(
            name=name,
            kind=EvaluatorKind.REFERENCE_BASED,
            provider="deterministic",
            model="not_applicable",
            prompt_version="not_applicable",
            evaluator_version=evaluator_version,
            configuration_version=configuration_version,
        ),
        score=score,
    )


def make_reference_free_evaluator(
    score: QualityScorer,
    *,
    evaluator_version: str,
    configuration_version: str,
    name: str = "reference_free",
) -> QualityEvaluator:
    return QualityEvaluator(
        metadata=EvaluatorMetadata(
            name=name,
            kind=EvaluatorKind.REFERENCE_FREE,
            provider="deterministic",
            model="not_applicable",
            prompt_version="not_applicable",
            evaluator_version=evaluator_version,
            configuration_version=configuration_version,
        ),
        score=score,
    )


def make_llm_judge_evaluator(
    score: QualityScorer,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    evaluator_version: str,
    configuration_version: str,
    name: str = "llm_judge",
) -> QualityEvaluator:
    return QualityEvaluator(
        metadata=EvaluatorMetadata(
            name=name,
            kind=EvaluatorKind.LLM_JUDGE,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            evaluator_version=evaluator_version,
            configuration_version=configuration_version,
        ),
        score=score,
    )


@dataclass(frozen=True)
class CalibrationDisagreement:
    case_id: str
    criterion: str
    expected: str
    actual: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "case_id": self.case_id,
            "criterion": self.criterion,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class CalibrationUncertainty:
    comparison_count: int
    standard_error: float
    confidence_level: float
    lower_bound: float
    upper_bound: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "comparison_count": self.comparison_count,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class CalibrationResult:
    metadata: EvaluatorMetadata
    dataset_version: str
    split: EvaluationSplit
    case_count: int
    comparison_count: int
    agreement_count: int
    agreement_rate: float
    uncertainty: CalibrationUncertainty
    disagreements: tuple[CalibrationDisagreement, ...]
    quality_gate_status: Literal["report_only", "approved"] = "report_only"
    approval: Mapping[str, str] | None = None

    @property
    def disagreement_count(self) -> int:
        return len(self.disagreements)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.as_dict(),
            "dataset_version": self.dataset_version,
            "split": self.split.value,
            "case_count": self.case_count,
            "comparison_count": self.comparison_count,
            "agreement_count": self.agreement_count,
            "agreement_rate": self.agreement_rate,
            "uncertainty": self.uncertainty.as_dict(),
            "disagreement_count": self.disagreement_count,
            "disagreements": [item.as_dict() for item in self.disagreements],
            "quality_gate_status": self.quality_gate_status,
            "approval": dict(self.approval) if self.approval is not None else None,
        }


def calibrate_quality_evaluator(
    dataset: GovernedEvaluationDataset,
    evaluator: QualityEvaluator,
    *,
    candidate_outputs: Mapping[str, Any],
    confidence_level: float = 0.95,
) -> CalibrationResult:
    """Compare candidate outputs with human labels from untouched held-out cases.

    The function intentionally has no parameter for another split.  It also
    rejects overlap samples in the held-out partition so human-agreement
    samples cannot leak into judge calibration. Candidate outputs are supplied
    separately from the reference case so an evaluator cannot silently score
    the expected behavior instead of the system output.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    held_out_cases = [case for case in dataset.cases if case.split is EvaluationSplit.HELD_OUT]
    if not held_out_cases:
        raise ValueError("Calibration requires at least one held-out Evaluation Case.")

    disagreements: list[CalibrationDisagreement] = []
    comparison_count = 0
    agreement_count = 0
    for case in held_out_cases:
        if case.overlap_sample:
            raise ValueError(
                f"Held-out overlap sample {case.case_id!r} cannot be used for calibration."
            )
        if len(case.reviewer_labels) != 1:
            raise ValueError(
                f"Held-out case {case.case_id!r} must have exactly one human label."
            )
        if case.case_id not in candidate_outputs:
            raise ValueError(f"Missing candidate output for held-out case {case.case_id!r}.")
        expected = case.reviewer_labels[0].rubric_scores
        actual = evaluator.evaluate(_calibration_case(case), candidate_outputs[case.case_id])
        for criterion, expected_value in expected.items():
            comparison_count += 1
            actual_value = actual.get(criterion)
            if actual_value == expected_value:
                agreement_count += 1
            else:
                disagreements.append(
                    CalibrationDisagreement(
                        case_id=case.case_id,
                        criterion=criterion,
                        expected=expected_value,
                        actual=actual_value,
                    )
                )

    agreement_rate = agreement_count / comparison_count if comparison_count else 0.0
    uncertainty = _agreement_uncertainty(
        agreement_rate, comparison_count, confidence_level=confidence_level
    )
    return CalibrationResult(
        metadata=evaluator.metadata,
        dataset_version=dataset.dataset_version,
        split=EvaluationSplit.HELD_OUT,
        case_count=len(held_out_cases),
        comparison_count=comparison_count,
        agreement_count=agreement_count,
        agreement_rate=agreement_rate,
        uncertainty=uncertainty,
        disagreements=tuple(disagreements),
    )


def approve_calibration(
    calibration: CalibrationResult,
    *,
    approver: str,
    approval_reference: str,
) -> CalibrationResult:
    """Record explicit human approval before a quality judge can block."""
    if not approver or not approval_reference:
        raise ValueError("approver and approval_reference are required.")
    return replace(
        calibration,
        quality_gate_status="approved",
        approval={"approver": approver, "approval_reference": approval_reference},
    )


def record_calibration(calibration: CalibrationResult, path: Path) -> Path:
    """Persist a redacted metadata-only calibration record for auditability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.as_dict(), indent=2) + "\n")
    return path


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    current: float
    baseline: float
    regressed: bool


@dataclass(frozen=True)
class QualityBaselineComparison:
    configuration_version: str
    baseline_configuration_version: str
    comparable: bool
    metric_comparisons: Mapping[str, MetricComparison]
    regressions: tuple[str, ...]
    configuration_mismatch: bool
    metric_set_mismatch: bool
    missing_current_metrics: tuple[str, ...]
    missing_baseline_metrics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "configuration_version": self.configuration_version,
            "baseline_configuration_version": self.baseline_configuration_version,
            "comparable": self.comparable,
            "metric_comparisons": {
                name: {
                    "current": item.current,
                    "baseline": item.baseline,
                    "regressed": item.regressed,
                }
                for name, item in self.metric_comparisons.items()
            },
            "regressions": list(self.regressions),
            "configuration_mismatch": self.configuration_mismatch,
            "metric_set_mismatch": self.metric_set_mismatch,
            "missing_current_metrics": list(self.missing_current_metrics),
            "missing_baseline_metrics": list(self.missing_baseline_metrics),
        }


def compare_quality_baseline(
    *,
    current_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    configuration_version: str,
    baseline_configuration_version: str,
) -> QualityBaselineComparison:
    """Compare like-for-like scorecard metrics without a composite score."""
    if not configuration_version:
        raise ValueError("The current configuration version is required for comparison.")
    mismatch = (
        not baseline_configuration_version
        or configuration_version != baseline_configuration_version
    )
    shared_metrics = sorted(set(current_metrics) & set(baseline_metrics))
    missing_current_metrics = tuple(sorted(set(baseline_metrics) - set(current_metrics)))
    missing_baseline_metrics = tuple(sorted(set(current_metrics) - set(baseline_metrics)))
    metric_set_mismatch = bool(missing_current_metrics or missing_baseline_metrics)
    comparisons = {
        metric: MetricComparison(
            metric=metric,
            current=float(current_metrics[metric]),
            baseline=float(baseline_metrics[metric]),
            regressed=float(current_metrics[metric]) < float(baseline_metrics[metric]),
        )
        for metric in shared_metrics
    }
    regressions = tuple(metric for metric in shared_metrics if comparisons[metric].regressed)
    return QualityBaselineComparison(
        configuration_version=configuration_version,
        baseline_configuration_version=baseline_configuration_version,
        comparable=not mismatch and not metric_set_mismatch,
        metric_comparisons=comparisons if not mismatch and not metric_set_mismatch else {},
        regressions=regressions if not mismatch and not metric_set_mismatch else (),
        configuration_mismatch=mismatch,
        metric_set_mismatch=metric_set_mismatch,
        missing_current_metrics=missing_current_metrics,
        missing_baseline_metrics=missing_baseline_metrics,
    )


@dataclass(frozen=True)
class QualityGateDecision:
    status: Literal["report_only", "pass", "blocking"]
    blocking: bool
    reason: str


def quality_gate_decision(
    calibration: CalibrationResult,
    comparison: QualityBaselineComparison,
    *,
    configuration_version: str | None = None,
) -> QualityGateDecision:
    """Make the quality gate fail closed unless calibration was approved."""
    if calibration.quality_gate_status != "approved":
        return QualityGateDecision(
            status="report_only",
            blocking=False,
            reason="Quality judge is not calibrated and approved; result is report-only.",
        )
    active_configuration = configuration_version or comparison.configuration_version
    if active_configuration != calibration.metadata.configuration_version:
        return QualityGateDecision(
            status="report_only",
            blocking=False,
            reason="Quality judge configuration differs from its approved calibration.",
        )
    if not comparison.comparable:
        return QualityGateDecision(
            status="report_only",
            blocking=False,
            reason="Quality baseline configuration differs; no gate decision was made.",
        )
    if comparison.regressions:
        return QualityGateDecision(
            status="blocking",
            blocking=True,
            reason="Approved baseline regression: " + ", ".join(comparison.regressions),
        )
    return QualityGateDecision(status="pass", blocking=False, reason="No per-metric regressions.")


def _agreement_uncertainty(
    agreement_rate: float,
    comparison_count: int,
    *,
    confidence_level: float,
) -> CalibrationUncertainty:
    if comparison_count == 0:
        return CalibrationUncertainty(0, 0.0, confidence_level, 0.0, 0.0)
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    standard_error = sqrt(agreement_rate * (1 - agreement_rate) / comparison_count)
    denominator = 1 + z**2 / comparison_count
    centre = agreement_rate + z**2 / (2 * comparison_count)
    margin = z * sqrt(
        agreement_rate * (1 - agreement_rate) / comparison_count
        + z**2 / (4 * comparison_count**2)
    )
    return CalibrationUncertainty(
        comparison_count=comparison_count,
        standard_error=standard_error,
        confidence_level=confidence_level,
        lower_bound=max(0.0, (centre - margin) / denominator),
        upper_bound=min(1.0, (centre + margin) / denominator),
    )


def _calibration_case(case: EvaluationCase) -> CalibrationCase:
    return CalibrationCase(
        case_id=case.case_id,
        category=case.category.value,
        split=case.split,
        permitted_scope=case.permitted_scope,
        primary_error_taxonomy=case.primary_error_taxonomy.value,
        criteria=tuple(case.reviewer_labels[0].rubric_scores),
    )

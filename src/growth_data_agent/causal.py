"""Deterministic eligibility and review gates for governed causal analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

from .contracts import (
    CausalAnalysisPlan,
    CausalDesignRegistration,
    CausalDesignType,
    CausalDiagnostic,
    CausalEstimate,
    CausalEvaluation,
    CausalOutcomeData,
    CausalReview,
    CausalReviewStatus,
    CausalSupportCheck,
    DescriptiveComparison,
    EstimatorApproval,
)

REGISTERED_JIRA_NEW_MAU_EXPERIMENT_ID = "jira-new-mau-onboarding-experiment"
FAILED_SUPPORT_JIRA_NEW_MAU_EXPERIMENT_ID = (
    "jira-new-mau-onboarding-experiment-failed-support"
)
PENDING_REVIEW_JIRA_NEW_MAU_EXPERIMENT_ID = (
    "jira-new-mau-onboarding-experiment-pending-review"
)
ALL_USER_PRE_POST_JIRA_NEW_MAU_EXPERIMENT_ID = "jira-new-mau-all-user-pre-post"

_PRE_APPROVED_ESTIMATORS = {
    CausalDesignType.RANDOMIZED_EXPERIMENT: frozenset({"difference_in_means"}),
}
_REVIEWED_AT = datetime(2026, 8, 25, 0, 0, tzinfo=UTC)


class CausalAnalysisPipeline:
    """Evaluate registered designs without accepting an estimator from the caller."""

    def __init__(
        self,
        registrations: Iterable[CausalDesignRegistration],
        *,
        outcomes: Mapping[str, CausalOutcomeData] | None = None,
    ):
        self._registrations = {item.experiment_id: item for item in registrations}
        self._outcomes = dict(outcomes or {})

    def evaluate(self, experiment_id: str) -> CausalEvaluation:
        registration = self._registrations.get(experiment_id)
        if registration is None:
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=None,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=experiment_id,
                    design_type=None,
                    proposed_estimator=None,
                    reason="The design is unregistered in the governed experiment registry.",
                    required_actions=[
                        "Register treatment, control, outcome, support checks, and design type.",
                        "Record the pre-approved estimator and required human review.",
                    ],
                ),
            )

        if (
            registration.product != "Jira"
            or registration.metric_name != "jira_new_mau"
            or registration.outcome != "jira_new_mau"
            or not registration.regions
            or not registration.tenant_scope
            or not registration.seat_tier
        ):
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=None,
                    review_status=CausalReviewStatus.PENDING,
                    reason=(
                        "Only a scoped registration for the Jira New MAU outcome is enabled "
                        "for this pipeline."
                    ),
                    required_actions=[
                        "Register the Jira New MAU outcome through the governed experiment path.",
                    ],
                ),
            )

        if registration.design_type == CausalDesignType.ALL_USER_PRE_POST:
            return self._descriptive(
                registration,
                "All-user pre/post comparisons are descriptive and cannot produce a "
                "Causal Estimate.",
            )

        if registration.design_type != CausalDesignType.RANDOMIZED_EXPERIMENT:
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=None,
                    reason=(
                        "Observational and quasi-experimental designs require a reviewed "
                        "analysis plan before any estimate."
                    ),
                    required_actions=[
                        "Document identification assumptions and the estimand.",
                        "Obtain required human review before estimation.",
                    ],
                ),
            )

        if not registration.support_checks or not all(
            check.passed for check in registration.support_checks
        ):
            return self._descriptive(
                registration,
                "A required support check failed, so no Causal Estimate was produced.",
            )

        estimator_approval = registration.estimator_approval
        if (
            estimator_approval is None
            or not estimator_approval.approved
            or estimator_approval.estimator
            not in _PRE_APPROVED_ESTIMATORS.get(registration.design_type, frozenset())
        ):
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=(
                        estimator_approval.estimator if estimator_approval is not None else None
                    ),
                    reason=(
                        "The design has no approved estimator in the pre-approved estimator "
                        "set."
                    ),
                    required_actions=[
                        "Record approval for the governed difference_in_means estimator.",
                        "Do not substitute an estimator supplied by the request.",
                    ],
                ),
            )

        if not registration.diagnostics or not all(
            diagnostic.passed for diagnostic in registration.diagnostics
        ):
            return self._descriptive(
                registration,
                "A required diagnostic failed, so no Causal Estimate was produced.",
            )

        if (
            registration.review is None
            or registration.review.status != CausalReviewStatus.APPROVED
            or not registration.review.reviewer
            or registration.review.reviewed_at is None
        ):
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=estimator_approval.estimator,
                    review_status=(
                        registration.review.status
                        if registration.review is not None
                        else CausalReviewStatus.PENDING
                    ),
                    reason="Required human review is not approved.",
                    required_actions=[
                        "Complete and record the required human review.",
                        "Re-run support checks and diagnostics after any design change.",
                    ],
                ),
            )

        if not registration.assumptions:
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=estimator_approval.estimator,
                    reason="The registered causal assumptions are missing.",
                    required_actions=["Record the assumptions required by the approved estimator."],
                ),
            )

        outcome_data = self._outcomes.get(experiment_id)
        if outcome_data is None:
            return CausalEvaluation(
                outcome="analysis_plan",
                registration=registration,
                causal_estimate=None,
                analysis_plan=CausalAnalysisPlan(
                    experiment_id=registration.experiment_id,
                    design_type=registration.design_type,
                    proposed_estimator=estimator_approval.estimator,
                    reason="The bounded treatment/control outcome data is missing.",
                    required_actions=["Record bounded treatment and control outcome values."],
                ),
            )

        estimate = outcome_data.treatment_value - outcome_data.control_value
        standard_error = outcome_data.standard_error
        if not all(isfinite(value) for value in (estimate, standard_error)):
            return self._descriptive(
                registration,
                "The registered outcome values are not finite, so no Causal Estimate was produced.",
            )

        return CausalEvaluation(
            outcome="causal_estimate",
            registration=registration,
            causal_estimate=CausalEstimate(
                experiment_id=registration.experiment_id,
                treatment=registration.treatment,
                control=registration.control,
                outcome=registration.outcome,
                estimator=estimator_approval.estimator,
                estimate=round(estimate, 10),
                standard_error=standard_error,
                confidence_interval=(
                    round(estimate - 1.96 * standard_error, 10),
                    round(estimate + 1.96 * standard_error, 10),
                ),
                assumptions=registration.assumptions,
                diagnostics=registration.diagnostics,
            ),
            analysis_plan=None,
        )

    def _descriptive(
        self,
        registration: CausalDesignRegistration, reason: str
    ) -> CausalEvaluation:
        outcome_data = self._outcomes.get(registration.experiment_id)
        descriptive_comparison = None
        if outcome_data is not None:
            difference = outcome_data.treatment_value - outcome_data.control_value
            descriptive_comparison = DescriptiveComparison(
                experiment_id=registration.experiment_id,
                treatment=registration.treatment,
                control=registration.control,
                outcome=registration.outcome,
                treatment_value=outcome_data.treatment_value,
                control_value=outcome_data.control_value,
                difference=round(difference, 10),
            )
        return CausalEvaluation(
            outcome="descriptive_result",
            registration=registration,
            causal_estimate=None,
            analysis_plan=CausalAnalysisPlan(
                experiment_id=registration.experiment_id,
                design_type=registration.design_type,
                proposed_estimator=(
                    registration.estimator_approval.estimator
                    if registration.estimator_approval is not None
                    else None
                ),
                reason=reason,
                required_actions=[
                    "Report the observed result as descriptive until the gate passes.",
                ],
            ),
            descriptive_comparison=descriptive_comparison,
        )


def default_causal_pipeline() -> CausalAnalysisPipeline:
    """Return the small deterministic registry for the Jira New MAU scenario."""
    passing = _registered_jira_new_mau_experiment()
    failed_support = passing.model_copy(deep=True, update={
        "experiment_id": FAILED_SUPPORT_JIRA_NEW_MAU_EXPERIMENT_ID,
        "support_checks": [
            *passing.support_checks[:-1],
            CausalSupportCheck(
                name="treatment_control_overlap",
                passed=False,
                details="Treatment/control support overlap is below the registered threshold.",
            ),
        ],
    })
    pending_review = passing.model_copy(deep=True, update={
        "experiment_id": PENDING_REVIEW_JIRA_NEW_MAU_EXPERIMENT_ID,
        "review": CausalReview(
            status=CausalReviewStatus.PENDING,
            reviewer="causal-review-board",
            reviewed_at=None,
            decision="Required review has not been completed.",
        ),
    })
    all_user_pre_post = passing.model_copy(deep=True, update={
        "experiment_id": ALL_USER_PRE_POST_JIRA_NEW_MAU_EXPERIMENT_ID,
        "treatment": "June 2026 all users",
        "control": "May 2026 all users",
        "design_type": CausalDesignType.ALL_USER_PRE_POST,
        "estimator_approval": None,
        "review": CausalReview(
            status=CausalReviewStatus.APPROVED,
            reviewer="descriptive-analysis-reviewer",
            reviewed_at=_REVIEWED_AT,
            decision="Approved for descriptive reporting only.",
        ),
    })
    observational = passing.model_copy(deep=True, update={
        "experiment_id": "jira-new-mau-observational-design",
        "design_type": CausalDesignType.OBSERVATIONAL,
        "estimator_approval": None,
    })
    return CausalAnalysisPipeline(
        (passing, failed_support, pending_review, all_user_pre_post, observational),
        outcomes=_load_default_outcomes(
            (passing, failed_support, pending_review, all_user_pre_post, observational)
        ),
    )


def _load_default_outcomes(
    registrations: Iterable[CausalDesignRegistration],
) -> dict[str, CausalOutcomeData]:
    fixture_path = Path(__file__).with_name("causal_outcomes.json")
    payload = json.loads(fixture_path.read_text())
    return {
        registration.experiment_id: CausalOutcomeData.model_validate(
            payload[registration.experiment_id]
        )
        for registration in registrations
    }


def _registered_jira_new_mau_experiment() -> CausalDesignRegistration:
    return CausalDesignRegistration(
        experiment_id=REGISTERED_JIRA_NEW_MAU_EXPERIMENT_ID,
        product="Jira",
        metric_name="jira_new_mau",
        treatment="onboarding_email_v2",
        control="no_onboarding_email",
        outcome="jira_new_mau",
        design_type=CausalDesignType.RANDOMIZED_EXPERIMENT,
        assignment_unit="Product User in Tenant",
        regions=["Americas"],
        tenant_scope="Americas 1-10 Seat Tier Tenants",
        seat_tier="1-10",
        support_checks=[
            CausalSupportCheck(
                name="randomized_assignment",
                passed=True,
                details="Assignment was randomized at Product User in Tenant level.",
            ),
            CausalSupportCheck(
                name="minimum_cell_size",
                passed=True,
                details="Treatment and control each exceed the registered minimum cell size.",
            ),
            CausalSupportCheck(
                name="treatment_control_overlap",
                passed=True,
                details="Treatment and control have adequate support overlap.",
            ),
        ],
        estimator_approval=EstimatorApproval(
            estimator="difference_in_means",
            approved=True,
            approved_by="causal-review-board",
            approved_at=_REVIEWED_AT,
        ),
        diagnostics=[
            CausalDiagnostic(
                name="baseline_balance",
                passed=True,
                details="Registered baseline covariates are balanced across arms.",
            ),
            CausalDiagnostic(
                name="outcome_completeness",
                passed=True,
                details="Jira New MAU outcome completeness exceeds the registered threshold.",
            ),
            CausalDiagnostic(
                name="assignment_attrition",
                passed=True,
                details="Post-assignment attrition is within the registered tolerance.",
            ),
        ],
        review=CausalReview(
            status=CausalReviewStatus.APPROVED,
            reviewer="causal-review-board",
            reviewed_at=_REVIEWED_AT,
            decision="Approved for the pre-approved estimator after passing the gate.",
        ),
        assumptions=[
            "Treatment assignment is randomized and precedes the Jira New MAU outcome.",
            "The canonical Jira New MAU definition is used for both arms.",
            "There is no material treatment spillover between treatment and control.",
        ],
    )

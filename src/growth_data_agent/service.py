"""The narrow answer_question application seam."""

from __future__ import annotations

import re
from uuid import uuid4

from .contracts import (
    AnswerQuestionRequest,
    GovernedAnalyticalResponse,
    MetricDefinitionGap,
    ResultClassification,
)
from .metric_definition_gaps import (
    DataTeamVerificationRequestRecorder,
    InMemoryDataTeamVerificationRequestRecorder,
    NoProvisionalMetricCalculator,
    NoProvisionalMetricInputGateway,
    ProvisionalMetricCalculator,
    ProvisionalMetricInputGateway,
    ProvisionalMetricInputRequest,
)
from .policy import resolve_access_profile
from .semantic import ValidatedMetricFlowGateway


class AnswerQuestionService:
    def __init__(
        self,
        semantic_gateway: ValidatedMetricFlowGateway,
        *,
        provisional_metric_calculator: ProvisionalMetricCalculator | None = None,
        provisional_metric_input_gateway: ProvisionalMetricInputGateway | None = None,
        verification_request_recorder: DataTeamVerificationRequestRecorder | None = None,
    ):
        self.semantic_gateway = semantic_gateway
        self.provisional_metric_calculator = (
            provisional_metric_calculator or NoProvisionalMetricCalculator()
        )
        self.provisional_metric_input_gateway = (
            provisional_metric_input_gateway or NoProvisionalMetricInputGateway()
        )
        self.verification_request_recorder = (
            verification_request_recorder or InMemoryDataTeamVerificationRequestRecorder()
        )

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        access_profile = resolve_access_profile(request.agent_user_id)
        scope = access_profile.as_effective_scope()
        trace_id = str(uuid4())

        metric_name = self._requested_metric_name(request)
        if metric_name is None:
            artifact = self.semantic_gateway.artifact_store.load()
            return GovernedAnalyticalResponse(
                answer=(
                    "This first delivery supports governed metric-definition questions, starting "
                    "with Jira New PEU. Name a metric to check its semantic status."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=self.semantic_gateway.freshness(artifact),
                effective_access_scope=scope,
                caveats=["The request did not identify a governed metric."],
                trace_id=trace_id,
            )

        if (
            metric_name == "jira_new_peu"
            and self._requests_may_to_june_driver_decomposition(request.question)
        ):
            return self._answer_may_to_june_driver_decomposition(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        definition, freshness = self.semantic_gateway.canonical_definition(metric_name)
        if definition is None:
            if freshness.is_current:
                return self._metric_definition_gap_response(
                    request=request,
                    metric_name=metric_name,
                    freshness=freshness,
                    scope=scope,
                    trace_id=trace_id,
                    access_profile=access_profile,
                )
            return GovernedAnalyticalResponse(
                answer=(
                    f"{metric_name} cannot be returned as canonical because the dbt/MetricFlow "
                    "semantic artifact is failed, stale, or unavailable."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=trace_id,
            )

        query_evidence, freshness = self.semantic_gateway.execute_scoped_metric(
            metric_name, access_profile
        )
        if query_evidence is None:
            return GovernedAnalyticalResponse(
                answer=(
                    "Jira New PEU cannot be returned as canonical because "
                    "validation is not current."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=trace_id,
            )

        return GovernedAnalyticalResponse(
            answer=(
                "Jira New PEU is a Product User's first-ever Paid Enablement for Jira. "
                "Later restorations of paid access do not create another New PEU."
            ),
            result_classification=ResultClassification.CANONICAL_DEFINITION,
            canonical_definition=definition,
            semantic_query_evidence=query_evidence,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                "This is a canonical definition, not a count for a particular period.",
                "The grain is Product User in a Tenant and product; it is not Person-level.",
            ],
            trace_id=trace_id,
        )

    def _answer_may_to_june_driver_decomposition(self, *, scope, access_profile, trace_id: str):
        definition, decomposition, query_evidence, freshness = (
            self.semantic_gateway.driver_decomposition(
                "jira_new_peu",
                access_profile,
                baseline_period="2026-05",
                comparison_period="2026-06",
            )
        )
        if definition is None or decomposition is None or query_evidence is None:
            return GovernedAnalyticalResponse(
                answer=(
                    "Jira New PEU cannot be decomposed as canonical because semantic validation "
                    "is not current."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=trace_id,
            )

        leading = decomposition.contributions[0] if decomposition.contributions else None
        leading_text = "No segment movement was returned."
        if leading is not None:
            leading_text = (
                f"{leading.region} / {leading.seat_tier} Seat Tier Tenants are the leading "
                "observed driver, contributing "
                f"{leading.contribution_to_decline:,} of the {decomposition.decline:,} decline "
                f"({leading.percentage_of_decline:g}%)."
            )
        return GovernedAnalyticalResponse(
            answer=(
                "Driver Decomposition (observed, non-causal): Jira New PEU moved from "
                f"{decomposition.baseline_value:,} in May 2026 to "
                f"{decomposition.comparison_value:,} in June 2026 "
                f"({decomposition.net_change:+,}). Semantic definition v"
                f"{definition.semantic_version}: {definition.definition} {leading_text} "
                "The approved Region and "
                "Seat Tier contributions reconcile to the scoped movement; this observation "
                "does not establish causation."
            ),
            result_classification=ResultClassification.DRIVER_DECOMPOSITION,
            canonical_definition=definition,
            semantic_query_evidence=query_evidence,
            driver_decomposition=decomposition,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                (
                    "This is a Driver Decomposition of observed canonical metric values, "
                    "not a causal conclusion."
                ),
                "Only Region and Seat Tier are approved dimensions for this decomposition.",
            ],
            trace_id=trace_id,
        )

    @staticmethod
    def _requests_jira_new_peu(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return "jira" in normalized and (
            "new peu" in normalized or "new paid enabled" in normalized
        )

    @staticmethod
    def _requests_may_to_june_driver_decomposition(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return (
            ("why" in normalized or "driver" in normalized or "decomposition" in normalized)
            and "may" in normalized
            and "june" in normalized
        )

    def _metric_definition_gap_response(
        self,
        *,
        request: AnswerQuestionRequest,
        metric_name: str,
        freshness,
        scope,
        trace_id: str,
        access_profile,
    ) -> GovernedAnalyticalResponse:
        gap = MetricDefinitionGap(requested_metric_name=metric_name)
        verification_request = None
        confirmation = request.verification_request_confirmation
        if confirmation is not None and confirmation.approved:
            verification_request = self.verification_request_recorder.record(
                metric_name=metric_name,
                agent_user_id=request.agent_user_id,
                confirmation=confirmation,
            )

        required_inputs = self.provisional_metric_calculator.required_inputs(metric_name)
        provisional_metric = None
        if required_inputs is not None and access_profile.permits_provisional_inputs(
            list(required_inputs)
        ):
            input_request = ProvisionalMetricInputRequest(
                metric_name=metric_name,
                inputs=required_inputs,
                scope=scope,
            )
            scoped_inputs = self.provisional_metric_input_gateway.read(input_request)
            if (
                scoped_inputs is not None
                and scoped_inputs.request == input_request
                and scoped_inputs.contains_only_requested_inputs()
            ):
                provisional_metric = self.provisional_metric_calculator.calculate(
                    scoped_inputs, freshness
                )
        if (
            provisional_metric is not None
            and provisional_metric.scope == scope
            and provisional_metric.inputs == list(required_inputs)
            and access_profile.permits_provisional_inputs(provisional_metric.inputs)
        ):
            return GovernedAnalyticalResponse(
                answer=(
                    f"Provisional Metric: {metric_name}. This calculation is explicitly "
                    "unverified and is not a canonical metric."
                ),
                result_classification=ResultClassification.PROVISIONAL_METRIC,
                metric_definition_gap=gap,
                provisional_metric=provisional_metric,
                data_team_verification_request=verification_request,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=provisional_metric.material_caveats,
                trace_id=trace_id,
            )

        request_status = (
            " An approved data-team verification request has been recorded."
            if verification_request is not None
            else " You may explicitly confirm a data-team verification request."
        )
        return GovernedAnalyticalResponse(
            answer=(
                f"Metric Definition Gap: {metric_name} is absent from the validated "
                "dbt/MetricFlow semantic authority, so it is not a canonical metric and "
                "no provisional calculation was made because permitted inputs cannot support "
                "it safely."
                f"{request_status}"
            ),
            result_classification=ResultClassification.METRIC_DEFINITION_GAP,
            metric_definition_gap=gap,
            data_team_verification_request=verification_request,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                "No permitted, pre-approved provisional calculation is available for this metric.",
                "A data-team verification request is never created without affirmative approval "
                "and approval context from the Agent User.",
            ],
            trace_id=trace_id,
        )

    @staticmethod
    def _requested_metric_name(request: AnswerQuestionRequest) -> str | None:
        if request.requested_metric_name is not None:
            return _metric_identifier(request.requested_metric_name)

        normalized = " ".join(request.question.casefold().split())
        if "jira" in normalized and ("new peu" in normalized or "new paid enabled" in normalized):
            return "jira_new_peu"
        if "jira" in normalized and "new mau" in normalized:
            return "jira_new_mau"
        if "confluence" in normalized and "new peu" in normalized:
            return "confluence_new_peu"
        if "confluence" in normalized and "new mau" in normalized:
            return "confluence_new_mau"
        if any(term in normalized for term in ("metric", "rate", "count", "revenue")):
            return _metric_identifier(request.question)
        named_metric = _named_metric_question(request.question)
        if named_metric is not None:
            return _metric_identifier(named_metric)
        return None


def _metric_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _named_metric_question(question: str) -> str | None:
    match = re.fullmatch(r"\s*(?:what is|define)\s+(.+?)\s*\??\s*", question, re.IGNORECASE)
    if match is None:
        return None
    name = match.group(1).strip()
    if name.casefold() in {"this", "that", "it"}:
        return None
    return name

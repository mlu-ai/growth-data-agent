"""The narrow answer_question application seam."""

from __future__ import annotations

from uuid import uuid4

from .contracts import AnswerQuestionRequest, GovernedAnalyticalResponse, ResultClassification
from .policy import resolve_access_profile
from .semantic import ValidatedMetricFlowGateway


class AnswerQuestionService:
    def __init__(self, semantic_gateway: ValidatedMetricFlowGateway):
        self.semantic_gateway = semantic_gateway

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        access_profile = resolve_access_profile(request.agent_user_id)
        scope = access_profile.as_effective_scope()
        trace_id = str(uuid4())

        if not self._requests_jira_new_peu(request.question):
            artifact = self.semantic_gateway.artifact_store.load()
            return GovernedAnalyticalResponse(
                answer="This first delivery supports the canonical Jira New PEU definition.",
                result_classification=ResultClassification.LIMITATION,
                source_freshness=self.semantic_gateway.freshness(artifact),
                effective_access_scope=scope,
                caveats=["The requested metric is outside the validated semantic scope."],
                trace_id=trace_id,
            )

        if self._requests_may_to_june_driver_decomposition(request.question):
            return self._answer_may_to_june_driver_decomposition(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        definition, freshness = self.semantic_gateway.canonical_definition("jira_new_peu")
        if definition is None:
            return GovernedAnalyticalResponse(
                answer=(
                    "Jira New PEU cannot be returned as canonical because the dbt/MetricFlow "
                    "semantic artifact is failed, stale, unavailable, or missing the metric."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=trace_id,
            )

        query_evidence, freshness = self.semantic_gateway.execute_scoped_metric(
            "jira_new_peu", access_profile
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

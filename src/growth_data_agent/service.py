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
        scope = resolve_access_profile(request.agent_user_id).as_effective_scope()
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

        return GovernedAnalyticalResponse(
            answer=(
                "Jira New PEU is a Product User's first-ever Paid Enablement for Jira. "
                "Later restorations of paid access do not create another New PEU."
            ),
            result_classification=ResultClassification.CANONICAL_DEFINITION,
            canonical_definition=definition,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                "This is a canonical definition, not a count for a particular period.",
                "The grain is Product User in a Tenant and product; it is not Person-level.",
            ],
            trace_id=trace_id,
        )

    @staticmethod
    def _requests_jira_new_peu(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return "jira" in normalized and (
            "new peu" in normalized or "new paid enabled" in normalized
        )

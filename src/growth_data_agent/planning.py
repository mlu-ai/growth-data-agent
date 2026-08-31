"""Bounded lead-agent planning metadata for governed investigations."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    InvestigationComplexity,
    LeadAgentMetadata,
    PlanAction,
    PlanToolOutcome,
    ToolOutcomeStatus,
)
from .policy import policy_fingerprint

if TYPE_CHECKING:
    from .execution import AuthorizedExecution

_MAX_PLAN_ACTIONS = 3
_MAX_REPLANS = 2


class PlanningInvariantError(ValueError):
    """Raised when a replan would cross a governance boundary."""


class LeadAgentPlanner:
    """Create bounded plans and advance only within their original action set."""

    def start(
        self,
        intent: AnalyticalIntent,
        authorized_execution: AuthorizedExecution,
        *,
        semantic_current: bool,
        evidence_revision_keys: Iterable[tuple[str, str, str]] = (),
    ) -> LeadAgentMetadata | None:
        if intent.route not in {
            AnalyticalRoute.DRIVER_DECOMPOSITION,
            AnalyticalRoute.LEGACY,
            AnalyticalRoute.DIRECT_IDENTIFIER,
            AnalyticalRoute.CAUSAL_ANALYSIS,
        }:
            return None
        complexity, actions = _plan_for_route(intent.route)
        bounded_actions = list(actions[:_MAX_PLAN_ACTIONS])
        return LeadAgentMetadata(
            plan_id=str(uuid4()),
            complexity=complexity,
            actions=bounded_actions,
            current_action=bounded_actions[0],
            plan_revision=0,
            replan_count=0,
            policy_fingerprint=policy_fingerprint(authorized_execution.access_profile),
            semantic_current=semantic_current,
            evidence_revision_fingerprints=_fingerprint_revision_keys(evidence_revision_keys),
            tool_outcomes=[],
        )

    def replan(
        self,
        metadata: LeadAgentMetadata,
        outcome: PlanToolOutcome,
        *,
        current_policy_fingerprint: str,
        semantic_current: bool,
        evidence_revision_keys: Iterable[tuple[str, str, str]],
    ) -> LeadAgentMetadata:
        """Record one result and select only the next action already in the plan."""
        if metadata.current_action is None:
            raise PlanningInvariantError("The bounded plan has no remaining action.")
        if outcome.action is not metadata.current_action:
            raise PlanningInvariantError("Replan action is outside the bounded plan.")
        if metadata.policy_fingerprint != current_policy_fingerprint:
            raise PlanningInvariantError("Replanning cannot expand the current policy.")
        if not semantic_current or not metadata.semantic_current:
            raise PlanningInvariantError("Replanning requires current semantic freshness.")
        current_revisions = _fingerprint_revision_keys(evidence_revision_keys)
        if metadata.evidence_revision_fingerprints and (
            current_revisions != metadata.evidence_revision_fingerprints
        ):
            raise PlanningInvariantError("Replanning cannot reuse a stale evidence revision.")
        if metadata.replan_count >= _MAX_REPLANS:
            raise PlanningInvariantError("The bounded replan limit has been reached.")

        action_index = metadata.actions.index(metadata.current_action)
        next_action = (
            metadata.actions[action_index + 1]
            if action_index + 1 < len(metadata.actions)
            else None
        )
        return metadata.model_copy(
            update={
                "current_action": next_action,
                "plan_revision": metadata.plan_revision + 1,
                "replan_count": metadata.replan_count
                + (1 if outcome.status is ToolOutcomeStatus.FAILED else 0),
                "evidence_revision_fingerprints": (
                    metadata.evidence_revision_fingerprints or current_revisions
                ),
                "tool_outcomes": [*metadata.tool_outcomes, outcome],
                "last_replan_reason": (
                    "tool_failure" if outcome.status is ToolOutcomeStatus.FAILED else "tool_success"
                ),
            }
        )


def _plan_for_route(
    route: AnalyticalRoute,
) -> tuple[InvestigationComplexity, tuple[PlanAction, ...]]:
    if route is AnalyticalRoute.CAUSAL_ANALYSIS:
        # This is only the existing eligibility boundary; it does not add causal analysis.
        return InvestigationComplexity.HARD, (PlanAction.CAUSAL_GATE,)
    if route is AnalyticalRoute.LEGACY:
        return InvestigationComplexity.HARD, (
            PlanAction.METRICFLOW,
            PlanAction.CITED_EVIDENCE,
            PlanAction.LIGHTRAG,
        )
    if route is AnalyticalRoute.DIRECT_IDENTIFIER:
        return InvestigationComplexity.HARD, (
            PlanAction.CITED_EVIDENCE,
            PlanAction.LIGHTRAG,
        )
    return InvestigationComplexity.MEDIUM, (PlanAction.METRICFLOW,)


def _fingerprint_revision_keys(
    revision_keys: Iterable[tuple[str, str, str]],
) -> list[str]:
    return sorted(
        {
            sha256("\x1f".join(key).encode("utf-8")).hexdigest()
            for key in revision_keys
        }
    )

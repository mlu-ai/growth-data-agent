"""Bounded lead-agent planning metadata for governed investigations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PlanExecutionSnapshot:
    """Fresh governance inputs checked immediately before each action."""

    policy_fingerprint: str
    semantic_current: bool
    evidence_revision_keys: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class PlanActionExecution:
    """Opaque action result carried only in memory between bounded actions."""

    value: object | None = None
    payload: object | None = None
    evidence_revision_keys: tuple[tuple[str, str, str], ...] = ()
    stop: bool = False


@dataclass(frozen=True)
class PlanExecutionResult:
    """Final bounded action value and safe trace metadata."""

    value: object | None
    metadata: LeadAgentMetadata


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
        if outcome.status is ToolOutcomeStatus.FAILED and metadata.replan_count >= _MAX_REPLANS:
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

    def execute(
        self,
        metadata: LeadAgentMetadata,
        *,
        run_action: Callable[[PlanAction, object | None], PlanActionExecution],
        snapshot_provider: Callable[[object | None], PlanExecutionSnapshot],
        should_replan: Callable[[Exception], bool] | None = None,
        on_metadata: Callable[[LeadAgentMetadata], None] | None = None,
    ) -> PlanExecutionResult:
        """Run each remaining action, recording its result before choosing the next one."""
        current = metadata
        payload: object | None = None
        value: object | None = None
        while current.current_action is not None:
            action = current.current_action
            snapshot = snapshot_provider(payload)
            try:
                _validate_snapshot(current, snapshot)
            except PlanningInvariantError:
                current = current.model_copy(update={"last_replan_reason": "invariant_blocked"})
                if on_metadata is not None:
                    on_metadata(current)
                break

            try:
                execution = run_action(action, payload)
            except Exception as error:
                outcome = PlanToolOutcome(
                    action=action,
                    status=ToolOutcomeStatus.FAILED,
                    error_type=type(error).__name__,
                )
                if should_replan is not None and not should_replan(error):
                    current = _record_blocked_outcome(current, outcome, reason="tool_failure")
                    if on_metadata is not None:
                        on_metadata(current)
                    raise
                try:
                    current = self.replan(
                        current,
                        outcome,
                        current_policy_fingerprint=snapshot.policy_fingerprint,
                        semantic_current=snapshot.semantic_current,
                        evidence_revision_keys=snapshot.evidence_revision_keys,
                    )
                except PlanningInvariantError:
                    current = _record_blocked_outcome(current, outcome)
                    if on_metadata is not None:
                        on_metadata(current)
                    break
                if on_metadata is not None:
                    on_metadata(current)
                continue

            payload = execution.payload
            value = execution.value if execution.value is not None else value
            outcome = PlanToolOutcome(action=action, status=ToolOutcomeStatus.SUCCESS)
            try:
                current = self.replan(
                    current,
                    outcome,
                    current_policy_fingerprint=snapshot.policy_fingerprint,
                    semantic_current=snapshot.semantic_current,
                    evidence_revision_keys=(
                        execution.evidence_revision_keys or snapshot.evidence_revision_keys
                    ),
                )
            except PlanningInvariantError:
                current = _record_blocked_outcome(current, outcome)
                if on_metadata is not None:
                    on_metadata(current)
                break
            if on_metadata is not None:
                on_metadata(current)
            if execution.stop:
                current = current.model_copy(
                    update={"current_action": None, "last_replan_reason": "stopped"}
                )
                if on_metadata is not None:
                    on_metadata(current)
                break
        return PlanExecutionResult(value=value, metadata=current)


def _plan_for_route(
    route: AnalyticalRoute,
) -> tuple[InvestigationComplexity, tuple[PlanAction, ...]]:
    if route is AnalyticalRoute.LEGACY:
        return InvestigationComplexity.HARD, (
            PlanAction.METRICFLOW,
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


def _validate_snapshot(metadata: LeadAgentMetadata, snapshot: PlanExecutionSnapshot) -> None:
    if metadata.policy_fingerprint != snapshot.policy_fingerprint:
        raise PlanningInvariantError("Replanning cannot expand the current policy.")
    if not snapshot.semantic_current or not metadata.semantic_current:
        raise PlanningInvariantError("Replanning requires current semantic freshness.")
    current_revisions = _fingerprint_revision_keys(snapshot.evidence_revision_keys)
    if metadata.evidence_revision_fingerprints and (
        current_revisions != metadata.evidence_revision_fingerprints
    ):
        raise PlanningInvariantError("Replanning cannot reuse a stale evidence revision.")


def _record_blocked_outcome(
    metadata: LeadAgentMetadata, outcome: PlanToolOutcome, *, reason: str = "invariant_blocked"
) -> LeadAgentMetadata:
    return metadata.model_copy(
        update={
            "tool_outcomes": [*metadata.tool_outcomes, outcome],
            "last_replan_reason": reason,
        }
    )

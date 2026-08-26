"""Governed LangGraph execution for the answer-question application seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    EffectiveAccessScope,
    GovernedAnalyticalResponse,
)
from .policy import AccessProfile, resolve_access_profile


class IntentInterpreter(Protocol):
    """Propose a typed intent without deciding authorization or tool use."""

    def interpret(self, request: AnswerQuestionRequest) -> object: ...


class RuleBasedIntentInterpreter:
    """Baseline interpreter until the bounded local-model adapter is introduced."""

    def __init__(
        self,
        *,
        metric_name_resolver: Callable[[AnswerQuestionRequest], str | None],
        is_canonical_definition_request: Callable[[AnswerQuestionRequest, str | None], bool],
    ) -> None:
        self._metric_name_resolver = metric_name_resolver
        self._is_canonical_definition_request = is_canonical_definition_request

    def interpret(self, request: AnswerQuestionRequest) -> AnalyticalIntent:
        metric_name = self._metric_name_resolver(request)
        route = (
            AnalyticalRoute.CANONICAL_DEFINITION
            if self._is_canonical_definition_request(request, metric_name)
            else AnalyticalRoute.LEGACY
        )
        return AnalyticalIntent(route=route, metric_name=metric_name)


@dataclass(frozen=True)
class AuthorizedExecution:
    """The authorization result shared by every bounded graph handler."""

    request: AnswerQuestionRequest
    access_profile: AccessProfile
    effective_scope: EffectiveAccessScope
    trace_id: str


CanonicalDefinitionHandler = Callable[
    [AuthorizedExecution, AnalyticalIntent],
    GovernedAnalyticalResponse,
]
LegacyHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]
ClarificationHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]


class _ExecutionState(TypedDict, total=False):
    request: AnswerQuestionRequest
    authorized_execution: AuthorizedExecution
    intent: AnalyticalIntent
    trace_id: str
    response: GovernedAnalyticalResponse


class ExecutionGraph:
    """Authorize, validate intent, then invoke only the selected bounded route."""

    def __init__(
        self,
        *,
        intent_interpreter: IntentInterpreter,
        canonical_definition_handler: CanonicalDefinitionHandler,
        legacy_handler: LegacyHandler,
        clarification_handler: ClarificationHandler,
    ) -> None:
        self._intent_interpreter = intent_interpreter
        self._canonical_definition_handler = canonical_definition_handler
        self._legacy_handler = legacy_handler
        self._clarification_handler = clarification_handler
        graph = StateGraph(_ExecutionState)
        graph.add_node("authorize", self._authorize)
        graph.add_node("interpret", self._interpret)
        graph.add_node("canonical_definition", self._canonical_definition)
        graph.add_node("clarification", self._clarification)
        graph.add_node("legacy", self._legacy)
        graph.add_edge(START, "authorize")
        graph.add_edge("authorize", "interpret")
        graph.add_conditional_edges(
            "interpret",
            self._route,
            {
                AnalyticalRoute.CANONICAL_DEFINITION.value: "canonical_definition",
                AnalyticalRoute.CLARIFICATION.value: "clarification",
                AnalyticalRoute.LEGACY.value: "legacy",
            },
        )
        graph.add_edge("canonical_definition", END)
        graph.add_edge("clarification", END)
        graph.add_edge("legacy", END)
        self._compiled = graph.compile()

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        state = self._compiled.invoke({"request": request})
        return cast(GovernedAnalyticalResponse, state["response"])

    @staticmethod
    def _authorize(state: _ExecutionState) -> dict[str, object]:
        request = state["request"]
        access_profile = resolve_access_profile(request.agent_user_id)
        return {
            "authorized_execution": AuthorizedExecution(
                request=request,
                access_profile=access_profile,
                effective_scope=access_profile.as_effective_scope(),
                trace_id=str(uuid4()),
            ),
        }

    def _interpret(self, state: _ExecutionState) -> dict[str, AnalyticalIntent]:
        try:
            proposed_intent = self._intent_interpreter.interpret(
                state["authorized_execution"].request
            )
            if isinstance(proposed_intent, AnalyticalIntent):
                proposed_intent = proposed_intent.model_dump(warnings=False)
            intent = AnalyticalIntent.model_validate(
                proposed_intent
            )
        except ValidationError:
            intent = AnalyticalIntent(route=AnalyticalRoute.CLARIFICATION)
        return {"intent": intent}

    @staticmethod
    def _route(state: _ExecutionState) -> str:
        return state["intent"].route.value

    def _canonical_definition(
        self, state: _ExecutionState
    ) -> dict[str, GovernedAnalyticalResponse]:
        return {
            "response": self._canonical_definition_handler(
                state["authorized_execution"],
                state["intent"],
            )
        }

    def _legacy(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        return {"response": self._legacy_handler(state["authorized_execution"])}

    def _clarification(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        return {"response": self._clarification_handler(state["authorized_execution"])}

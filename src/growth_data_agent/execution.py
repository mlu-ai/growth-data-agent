"""Governed LangGraph execution for the answer-question application seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    ConversationContext,
    EffectiveAccessScope,
    GovernedAnalyticalResponse,
)
from .local_model import LocalModelError
from .observability import set_lead_agent_metadata, trace_span
from .planning import LeadAgentPlanner
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
        route_resolver: Callable[[AnswerQuestionRequest, str | None], AnalyticalRoute],
    ) -> None:
        self._metric_name_resolver = metric_name_resolver
        self._route_resolver = route_resolver

    def interpret(self, request: AnswerQuestionRequest) -> AnalyticalIntent:
        metric_name = self._metric_name_resolver(request)
        route = self._route_resolver(request, metric_name)
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
DriverDecompositionHandler = Callable[
    [AuthorizedExecution, AnalyticalIntent], GovernedAnalyticalResponse
]
CausalAnalysisHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]
CatalogOwnershipHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]
DirectIdentifierHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]
LimitationHandler = Callable[[AuthorizedExecution], GovernedAnalyticalResponse]
MetricDefinitionGapHandler = Callable[
    [AuthorizedExecution, AnalyticalIntent], GovernedAnalyticalResponse
]


class _ExecutionState(TypedDict, total=False):
    request: AnswerQuestionRequest
    conversation_context: ConversationContext | None
    authorized_execution: AuthorizedExecution
    intent: AnalyticalIntent
    trace_id: str
    response: GovernedAnalyticalResponse
    lead_agent_metadata: object


class ExecutionGraph:
    """Authorize, validate intent, then invoke only the selected bounded route."""

    def __init__(
        self,
        *,
        intent_interpreter: IntentInterpreter,
        canonical_definition_handler: CanonicalDefinitionHandler,
        driver_decomposition_handler: DriverDecompositionHandler,
        causal_analysis_handler: CausalAnalysisHandler,
        catalog_ownership_handler: CatalogOwnershipHandler,
        direct_identifier_handler: DirectIdentifierHandler,
        limitation_handler: LimitationHandler,
        metric_definition_gap_handler: MetricDefinitionGapHandler,
        legacy_handler: LegacyHandler,
        clarification_handler: ClarificationHandler,
        lead_agent_planner: LeadAgentPlanner | None = None,
        semantic_freshness_provider: Callable[[AuthorizedExecution], bool] | None = None,
        checkpointer=None,
    ) -> None:
        self._intent_interpreter = intent_interpreter
        self._canonical_definition_handler = canonical_definition_handler
        self._driver_decomposition_handler = driver_decomposition_handler
        self._causal_analysis_handler = causal_analysis_handler
        self._catalog_ownership_handler = catalog_ownership_handler
        self._direct_identifier_handler = direct_identifier_handler
        self._limitation_handler = limitation_handler
        self._metric_definition_gap_handler = metric_definition_gap_handler
        self._legacy_handler = legacy_handler
        self._clarification_handler = clarification_handler
        self._lead_agent_planner = lead_agent_planner or LeadAgentPlanner()
        self._semantic_freshness_provider = semantic_freshness_provider or (lambda _: True)
        graph = StateGraph(_ExecutionState)
        graph.add_node("authorize", self._authorize)
        graph.add_node("interpret", self._interpret)
        graph.add_node("canonical_definition", self._canonical_definition)
        graph.add_node("driver_decomposition", self._driver_decomposition)
        graph.add_node("causal_analysis", self._causal_analysis)
        graph.add_node("catalog_ownership", self._catalog_ownership)
        graph.add_node("direct_identifier", self._direct_identifier)
        graph.add_node("limitation", self._limitation)
        graph.add_node("metric_definition_gap", self._metric_definition_gap)
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
                AnalyticalRoute.DRIVER_DECOMPOSITION.value: "driver_decomposition",
                AnalyticalRoute.CAUSAL_ANALYSIS.value: "causal_analysis",
                AnalyticalRoute.CATALOG_OWNERSHIP.value: "catalog_ownership",
                AnalyticalRoute.DIRECT_IDENTIFIER.value: "direct_identifier",
                AnalyticalRoute.LIMITATION.value: "limitation",
                AnalyticalRoute.METRIC_DEFINITION_GAP.value: "metric_definition_gap",
                AnalyticalRoute.LEGACY.value: "legacy",
            },
        )
        graph.add_edge("canonical_definition", END)
        graph.add_edge("driver_decomposition", END)
        graph.add_edge("causal_analysis", END)
        graph.add_edge("catalog_ownership", END)
        graph.add_edge("direct_identifier", END)
        graph.add_edge("limitation", END)
        graph.add_edge("metric_definition_gap", END)
        graph.add_edge("clarification", END)
        graph.add_edge("legacy", END)
        self._compiled = graph.compile(checkpointer=checkpointer)

    def answer_question(
        self,
        request: AnswerQuestionRequest,
        *,
        conversation_id: str | None = None,
    ) -> GovernedAnalyticalResponse:
        input_state: _ExecutionState = {
            "request": request,
            "conversation_context": request.conversation_context,
        }
        config: RunnableConfig | None = (
            {"configurable": {"thread_id": conversation_id}}
            if conversation_id is not None
            else None
        )
        state = self._compiled.invoke(input_state, config=config)
        return cast(GovernedAnalyticalResponse, state["response"])

    def update_conversation_context(
        self, conversation_id: str, conversation_context: ConversationContext
    ) -> None:
        """Persist the post-Turn context in the native LangGraph checkpoint."""
        config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
        self._compiled.update_state(config, {"conversation_context": conversation_context})

    @staticmethod
    def _authorize(state: _ExecutionState) -> dict[str, object]:
        with trace_span("authorize", kind="node"):
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
            with trace_span("intent_interpretation", kind="node"):
                proposed_intent = self._intent_interpreter.interpret(
                    state["authorized_execution"].request
                )
                if isinstance(proposed_intent, AnalyticalIntent):
                    proposed_intent = proposed_intent.model_dump(warnings=False)
            with trace_span("intent_validation", kind="node"):
                intent = AnalyticalIntent.model_validate(proposed_intent)
        except (LocalModelError, ValidationError):
            intent = AnalyticalIntent(route=AnalyticalRoute.CLARIFICATION)
        metadata = self._lead_agent_planner.start(
            intent,
            state["authorized_execution"],
            semantic_current=self._semantic_freshness_provider(state["authorized_execution"]),
        )
        set_lead_agent_metadata(metadata)
        return {"intent": intent, "lead_agent_metadata": metadata}

    @staticmethod
    def _route(state: _ExecutionState) -> str:
        return state["intent"].route.value

    def _canonical_definition(
        self, state: _ExecutionState
    ) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("canonical_definition", kind="node"):
            return {
                "response": self._with_plan(
                    state,
                    self._canonical_definition_handler(
                        state["authorized_execution"],
                        state["intent"],
                    ),
                )
            }

    def _legacy(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("legacy", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._legacy_handler(state["authorized_execution"])
                )
            }

    def _driver_decomposition(
        self, state: _ExecutionState
    ) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("driver_decomposition", kind="node"):
            return {
                "response": self._with_plan(
                    state,
                    self._driver_decomposition_handler(
                        state["authorized_execution"], state["intent"]
                    ),
                )
            }

    def _causal_analysis(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("causal_analysis", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._causal_analysis_handler(state["authorized_execution"])
                )
            }

    def _catalog_ownership(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("catalog_ownership", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._catalog_ownership_handler(state["authorized_execution"])
                )
            }

    def _direct_identifier(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("direct_identifier", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._direct_identifier_handler(state["authorized_execution"])
                )
            }

    def _limitation(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("limitation", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._limitation_handler(state["authorized_execution"])
                )
            }

    def _metric_definition_gap(
        self, state: _ExecutionState
    ) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("metric_definition_gap", kind="node"):
            return {
                "response": self._with_plan(
                    state,
                    self._metric_definition_gap_handler(
                        state["authorized_execution"], state["intent"]
                    ),
                )
            }

    def _clarification(self, state: _ExecutionState) -> dict[str, GovernedAnalyticalResponse]:
        with trace_span("clarification", kind="node"):
            return {
                "response": self._with_plan(
                    state, self._clarification_handler(state["authorized_execution"])
                )
            }

    @staticmethod
    def _with_plan(
        state: _ExecutionState, response: GovernedAnalyticalResponse | None
    ) -> GovernedAnalyticalResponse | None:
        if response is None:
            return None
        metadata = state.get("lead_agent_metadata")
        if metadata is None:
            return response
        return response.model_copy(update={"lead_agent_metadata": metadata})

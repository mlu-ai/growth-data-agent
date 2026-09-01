"""The narrow answer_question application seam."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from .audit import DirectIdentifierAuditRecorder, InMemoryDirectIdentifierAuditRecorder
from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    CandidateCausalFactor,
    CatalogFreshness,
    CatalogMetadata,
    ConversationSummary,
    ConversationTurn,
    DirectIdentifierAnswer,
    EvidenceChain,
    EvidenceChainChunk,
    EvidenceChainEntity,
    EvidenceChainReference,
    EvidenceChainRelation,
    FactorSupportStatus,
    GovernedAnalyticalResponse,
    MetricDefinitionGap,
    OpportunityEstimate,
    OpportunitySizingGap,
    PlanAction,
    ResultClassification,
    SensitiveIdentifier,
    SourceFreshness,
)
from .conversations import (
    ConversationAccessDeniedError,
    ConversationCheckpointStore,
    ConversationNotFoundError,
    InMemoryConversationCheckpointStore,
)
from .datahub import (
    DataHubCatalogStore,
    DataHubCatalogUnavailableError,
)
from .evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    QdrantEvidenceStore,
    VectorEvidenceStore,
    _document_revision_key,
    _evidence_revision_key,
    build_evidence_answer,
    embedding_readiness,
)
from .evidence_tools import BoundedEvidenceInvestigationTools, CitedEvidencePreparation
from .execution import (
    AuthorizedExecution,
    ExecutionGraph,
    IntentInterpreter,
    RuleBasedIntentInterpreter,
)
from .factor_ranking import rank_candidate_causal_factors, sizing_eligible_metric_name
from .factors import (
    DriverMovementWindow,
    build_evidence_investigation_query,
)
from .graph import (
    ApacheAgeEvidenceGraphStore,
    EvidenceGraphStore,
    EvidenceGraphUnavailableError,
    GraphAccessFilter,
    InMemoryEvidenceGraphStore,
)
from .lightrag import (
    AuthorizedEvidenceRevisionSet,
    LightRAGAuthorizationError,
    LightRAGBackend,
    LightRAGEvidenceAdapter,
    QdrantAGELightRAGStore,
    require_bound_qdrant_age_stores,
    require_governed_lightrag_adapter,
    validate_authorized_lightrag_references,
)
from .local_model import (
    EvidenceDraftingAdapter,
    LocalModelError,
    LocalModelEvidenceDraftingAdapter,
    LocalModelIntentInterpreter,
    LocalModelTransport,
    local_model_readiness,
    validate_local_model_draft,
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
from .observability import (
    NoOpTraceSink,
    TraceContext,
    TraceDeliveryHealth,
    TraceRecord,
    TraceSink,
    capture_trace,
    policy_fingerprint,
    safe_evaluation_execution_projection,
    trace_delivery_health_for,
    trace_span,
)
from .planning import (
    PlanActionExecution,
    PlanExecutionSnapshot,
    RevisionIdentity,
)
from .policy import (
    AccessDeniedError,
    UnknownAgentUserError,
    resolve_access_profile,
)
from .reranking import (
    EvidenceReranker,
    EvidenceRerankingError,
    reranker_readiness,
)
from .semantic import ValidatedMetricFlowGateway
from .synthetic import evidence_corpus, graph_corpus

_DIRECT_IDENTIFIER_RESULT_LIMIT = 3
_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)
RevisionReader = Callable[[EvidenceAccessFilter], Iterable[EvidenceDocument]]

_CURRENT_EVALUATION_TRACE: ContextVar[TraceRecord | None] = ContextVar(
    "growth_data_agent_evaluation_trace", default=None
)
_EVALUATION_CAPTURE_ACTIVE: ContextVar[bool] = ContextVar(
    "growth_data_agent_evaluation_capture_active", default=False
)


@dataclass
class _PlannedInvestigationState:
    """Opaque in-memory state shared by one bounded investigation turn."""

    authorized_execution: AuthorizedExecution
    intent: AnalyticalIntent
    metric_name: str
    region: str | None = None
    seat_tier: str | None = None
    evidence_query: str | None = None
    scope_evidence_to_seat_tier: bool = False
    matching_contribution: Any = None
    driver_window: Any = None
    definition: Any = None
    decomposition: Any = None
    query_evidence: Any = None
    freshness: SourceFreshness | None = None
    evidence_filter: EvidenceAccessFilter | None = None
    graph_filter: Any = None
    cited_evidence: CitedEvidencePreparation | None = None
    validated_evidence_revision_identities: tuple[RevisionIdentity, ...] = ()
    current_evidence_filter: EvidenceAccessFilter | None = None
    current_graph_filter: GraphAccessFilter | None = None
    graph_paths: list[Any] = field(default_factory=list)
    response: GovernedAnalyticalResponse | None = None


class AnswerQuestionService:
    def __init__(
        self,
        semantic_gateway: ValidatedMetricFlowGateway,
        *,
        provisional_metric_calculator: ProvisionalMetricCalculator | None = None,
        provisional_metric_input_gateway: ProvisionalMetricInputGateway | None = None,
        verification_request_recorder: DataTeamVerificationRequestRecorder | None = None,
        evidence_store: VectorEvidenceStore | None = None,
        lightrag_adapter: LightRAGEvidenceAdapter | None = None,
        evidence_reranker: EvidenceReranker | None = None,
        graph_store: EvidenceGraphStore | None = None,
        catalog_store: DataHubCatalogStore | None = None,
        direct_identifier_audit_recorder: DirectIdentifierAuditRecorder | None = None,
        trace_sink: TraceSink | None = None,
        execution_graph: ExecutionGraph | None = None,
        local_model: LocalModelTransport | None = None,
        evidence_model: LocalModelTransport | None = None,
        intent_interpreter: IntentInterpreter | None = None,
        evidence_drafting_adapter: EvidenceDraftingAdapter | None = None,
        conversation_store: ConversationCheckpointStore | None = None,
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
        self.evidence_store = evidence_store or QdrantEvidenceStore(evidence_corpus())
        self.evidence_reranker = evidence_reranker
        self.graph_store = graph_store or InMemoryEvidenceGraphStore(graph_corpus())
        if lightrag_adapter is None:
            if type(self.evidence_store) is QdrantEvidenceStore and type(
                self.graph_store
            ) in {InMemoryEvidenceGraphStore, ApacheAgeEvidenceGraphStore}:
                lightrag_adapter = LightRAGEvidenceAdapter(
                    LightRAGBackend(
                        QdrantAGELightRAGStore(
                            cast(QdrantEvidenceStore, self.evidence_store),
                            cast(
                                InMemoryEvidenceGraphStore | ApacheAgeEvidenceGraphStore,
                                self.graph_store,
                            ),
                        )
                    )
                )
        self.lightrag_adapter = lightrag_adapter
        self.evidence_tools = BoundedEvidenceInvestigationTools(
            self.evidence_store,
            self._traverse_graph_for_evidence_tool,
            self.evidence_reranker,
            self.lightrag_adapter,
        )
        self.catalog_store = catalog_store
        self.direct_identifier_audit_recorder = (
            direct_identifier_audit_recorder or InMemoryDirectIdentifierAuditRecorder()
        )
        self.trace_sink = trace_sink or NoOpTraceSink()
        self.trace_delivery_health: TraceDeliveryHealth = trace_delivery_health_for(self.trace_sink)
        self.conversation_store = conversation_store or InMemoryConversationCheckpointStore()
        self.local_model = local_model
        evidence_model = evidence_model if evidence_model is not None else local_model
        self.evidence_drafting_adapter = evidence_drafting_adapter or (
            LocalModelEvidenceDraftingAdapter(evidence_model)
            if evidence_model is not None
            else None
        )
        if intent_interpreter is not None:
            configured_intent_interpreter = intent_interpreter
        elif local_model is not None:
            configured_intent_interpreter = LocalModelIntentInterpreter(
                local_model,
                metric_names_provider=self._available_metric_names_for_request,
                route_resolver=self._route_for_validated_intent,
            )
        else:
            configured_intent_interpreter = RuleBasedIntentInterpreter(
                metric_name_resolver=self._requested_metric_name,
                route_resolver=self._route_for_validated_intent,
            )
        self.execution_graph = execution_graph or ExecutionGraph(
            intent_interpreter=configured_intent_interpreter,
            canonical_definition_handler=self._answer_canonical_definition,
            driver_decomposition_handler=self._answer_driver_decomposition,
            causal_analysis_handler=self._answer_causal_redirect_specialist,
            catalog_ownership_handler=self._answer_catalog_specialist,
            direct_identifier_handler=self._answer_direct_identifier_specialist,
            limitation_handler=self._answer_limitation_specialist,
            metric_definition_gap_handler=self._answer_metric_definition_gap_specialist,
            legacy_handler=self._answer_legacy_question,
            clarification_handler=self._answer_intent_clarification,
            semantic_freshness_provider=self._semantic_is_current,
            plan_action_executor=self._execute_plan_action,
            planning_snapshot_provider=self._planning_snapshot,
            planning_eligibility_provider=self._planning_eligible,
            planning_failure_policy=self._planning_failure_policy,
            planning_blocked_handler=self._planning_blocked_response,
            checkpointer=getattr(self.conversation_store, "checkpointer", None),
        )

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        with capture_trace() as trace_context:
            try:
                principal = request.verified_principal
                if principal is None:
                    raise AccessDeniedError("A verified Principal is required.")
                request = request.model_copy(
                    update={
                        "agent_user_id": principal.principal_id,
                        "verified_principal": principal,
                    }
                )
                access_profile = resolve_access_profile(principal.principal_id)
                checkpoint = self._conversation_checkpoint(request, principal)
                trace_context.conversation_id = checkpoint.conversation_id
                request = request.model_copy(
                    update={
                        "conversation_context": checkpoint.context(
                            effective_scope=access_profile.as_effective_scope()
                        )
                    }
                )
                response = self.execution_graph.answer_question(
                    request,
                    conversation_id=checkpoint.conversation_id,
                )
                response = self._draft_evidence_response(response)
                response = response.model_copy(
                    update={"conversation_id": checkpoint.conversation_id}
                )
                updated_checkpoint = self.conversation_store.append(
                    checkpoint.conversation_id,
                    principal,
                    turn=self._conversation_turn(request, response),
                    summary=self._conversation_summary(request, response),
                )
                trace_context.has_active_investigation_selection = (
                    updated_checkpoint.summary.active_investigation_factor_id is not None
                )
                if getattr(self.conversation_store, "checkpointer", None) is not None:
                    self.execution_graph.update_conversation_context(
                        checkpoint.conversation_id,
                        updated_checkpoint.context(
                            effective_scope=access_profile.as_effective_scope()
                        ),
                    )
            except (AccessDeniedError, UnknownAgentUserError) as error:
                error.trace_id = self._record_authorization_denial(request, trace_context)
                raise
            except (ConversationAccessDeniedError, ConversationNotFoundError) as error:
                denied = AccessDeniedError("Conversation is not available to this Agent User.")
                denied.trace_id = self._record_authorization_denial(request, trace_context)
                raise denied from error
            except Exception as error:
                error.trace_id = self._record_dependency_failure(request, trace_context)
                raise
            self._record_trace(request, response, trace_context)
            return response

    def answer_question_evaluation_projection(
        self, request: AnswerQuestionRequest
    ) -> dict[str, Any]:
        """Execute one governed turn and return only its evaluator-safe trace projection."""
        token = _CURRENT_EVALUATION_TRACE.set(None)
        capture_token = _EVALUATION_CAPTURE_ACTIVE.set(True)
        try:
            self.answer_question(request)
            trace = _CURRENT_EVALUATION_TRACE.get()
            if trace is None:
                raise RuntimeError("The governed request did not produce an evaluation trace.")
            return safe_evaluation_execution_projection(trace)
        finally:
            _EVALUATION_CAPTURE_ACTIVE.reset(capture_token)
            _CURRENT_EVALUATION_TRACE.reset(token)

    def _conversation_checkpoint(self, request, principal):
        if request.conversation_id is None:
            return self.conversation_store.create(principal)
        return self.conversation_store.load(request.conversation_id, principal)

    def _planning_eligible(
        self, authorized_execution: AuthorizedExecution, intent: AnalyticalIntent
    ) -> bool:
        if intent.route is AnalyticalRoute.DRIVER_DECOMPOSITION:
            return True
        if intent.route is not AnalyticalRoute.LEGACY:
            return False
        question = authorized_execution.request.question
        return any(
            (
                self._requests_apac_decline_evidence(question),
                self._requests_confluence_campaign_evidence(question),
                self._requests_confluence_emea_regression(question),
            )
        )

    def _planning_snapshot(
        self, authorized_execution: AuthorizedExecution, payload: object | None
    ) -> PlanExecutionSnapshot:
        current_profile = resolve_access_profile(authorized_execution.request.agent_user_id)
        semantic_current = self._semantic_is_current(authorized_execution)
        evidence_revision_keys: tuple[RevisionIdentity, ...] = ()
        if isinstance(payload, _PlannedInvestigationState) and payload.region is not None:
            try:
                current_filter = self._current_planned_evidence_filter(payload, current_profile)
                payload.current_evidence_filter = current_filter
                evidence_revision_keys = self._authorized_active_evidence_manifest(current_filter)
                if (
                    payload.validated_evidence_revision_identities
                    and payload.cited_evidence is not None
                ):
                    payload.current_graph_filter = self._current_planned_graph_filter(
                        payload, current_profile, current_filter
                    )
            except Exception:
                # A missing authoritative manifest is itself a failed freshness proof.
                semantic_current = False
                if payload.validated_evidence_revision_identities:
                    evidence_revision_keys = (("__manifest_unavailable__", "", "", ""),)
        return PlanExecutionSnapshot(
            policy_fingerprint=policy_fingerprint(current_profile),
            semantic_current=semantic_current,
            evidence_revision_keys=evidence_revision_keys,
        )

    def _current_planned_evidence_filter(
        self, state: _PlannedInvestigationState, profile
    ) -> EvidenceAccessFilter:
        product = cast(str, self._metric_product(state.metric_name))
        return profile.evidence_filter(
            product,
            cast(str, state.region),
            seat_tier=state.seat_tier if state.scope_evidence_to_seat_tier else None,
            metric_name=state.metric_name,
            agent_user_id=state.authorized_execution.request.agent_user_id,
        )

    def _authorized_active_evidence_manifest(
        self, access_filter: EvidenceAccessFilter
    ) -> tuple[RevisionIdentity, ...]:
        """Read current active revision identity from the authoritative evidence source."""
        revision_reader = getattr(self.evidence_store, "authorized_revisions", None)
        if callable(revision_reader):
            documents = tuple(revision_reader(access_filter))
        else:
            source_documents = getattr(self.evidence_store, "documents", None)
            if source_documents is None:
                raise LightRAGAuthorizationError(
                    "Current authorized evidence revisions are unavailable."
                )
            documents = tuple(source_documents)
        return tuple(
            sorted(
                _document_revision_key(document)
                for document in documents
                if access_filter.allows(document)
            )
        )

    def _current_planned_graph_filter(
        self,
        state: _PlannedInvestigationState,
        profile,
        evidence_filter: EvidenceAccessFilter,
    ) -> GraphAccessFilter:
        if state.cited_evidence is None or state.cited_evidence.graph_filter is None:
            raise LightRAGAuthorizationError("The graph scope is unavailable.")
        graph_filter = profile.graph_filter(
            cast(str, self._metric_product(state.metric_name)),
            cast(str, state.region),
            seat_tier=state.seat_tier if state.scope_evidence_to_seat_tier else None,
        )
        cited_graph_filter = state.cited_evidence.graph_filter
        return replace(
            graph_filter,
            groups=evidence_filter.groups,
            agent_user_id=evidence_filter.agent_user_id,
            as_of=evidence_filter.as_of,
            authorized_document_ids=cited_graph_filter.authorized_document_ids,
            authorized_revision_keys=cited_graph_filter.authorized_revision_keys,
        )

    @staticmethod
    def _planning_failure_policy(error: Exception) -> bool:
        """Do not reinterpret authorization or dependency failures as alternate work."""
        return not isinstance(
            error,
            (
                AccessDeniedError,
                EvidenceGraphUnavailableError,
                EvidenceRerankingError,
                LightRAGAuthorizationError,
            ),
        )

    def _planning_blocked_response(
        self,
        authorized_execution: AuthorizedExecution,
        _intent: AnalyticalIntent,
        _metadata,
    ) -> GovernedAnalyticalResponse:
        try:
            artifact = self.semantic_gateway.artifact_store.load()
            freshness = self.semantic_gateway.freshness(artifact)
        except (OSError, ValueError):
            freshness = SourceFreshness(
                validated_at=datetime.now(UTC), maximum_age_seconds=86_400, is_current=False
            )
        return GovernedAnalyticalResponse(
            answer=(
                "The bounded investigation was stopped before its next action because a current "
                "governance prerequisite was not satisfied."
            ),
            result_classification=ResultClassification.LIMITATION,
            source_freshness=freshness,
            effective_access_scope=authorized_execution.effective_scope,
            caveats=[
                "No subsequent action ran after the policy, semantic freshness, or evidence "
                "revision guard blocked it."
            ],
            trace_id=authorized_execution.trace_id,
        )

    def _execute_plan_action(
        self,
        authorized_execution: AuthorizedExecution,
        intent: AnalyticalIntent,
        action: PlanAction,
        payload: object | None,
    ) -> PlanActionExecution:
        state = payload
        if not isinstance(state, _PlannedInvestigationState):
            state = self._new_planned_investigation_state(authorized_execution, intent)
        if action is PlanAction.METRICFLOW:
            return self._run_plan_metricflow(state)
        if action is PlanAction.CITED_EVIDENCE:
            return self._run_plan_cited_evidence(state)
        if action is PlanAction.LIGHTRAG:
            return self._run_plan_lightrag(state)
        raise RuntimeError("Unsupported planned action.")

    def _new_planned_investigation_state(
        self, authorized_execution: AuthorizedExecution, intent: AnalyticalIntent
    ) -> _PlannedInvestigationState:
        request = authorized_execution.request
        if intent.route is AnalyticalRoute.DRIVER_DECOMPOSITION:
            return _PlannedInvestigationState(
                authorized_execution=authorized_execution,
                intent=intent,
                metric_name=cast(str, intent.metric_name),
            )
        question = request.question
        if self._requests_apac_decline_evidence(question):
            return _PlannedInvestigationState(
                authorized_execution=authorized_execution,
                intent=intent,
                metric_name="jira_new_peu",
                region="APAC",
                seat_tier="51-200",
                scope_evidence_to_seat_tier=True,
            )
        if self._requests_confluence_campaign_evidence(question):
            return _PlannedInvestigationState(
                authorized_execution=authorized_execution,
                intent=intent,
                metric_name="confluence_new_peu",
                region="Americas",
                seat_tier="11-50",
                scope_evidence_to_seat_tier=True,
            )
        return _PlannedInvestigationState(
            authorized_execution=authorized_execution,
            intent=intent,
            metric_name="confluence_new_mau",
            region="EMEA",
            seat_tier="51-200",
            scope_evidence_to_seat_tier=True,
        )

    def _run_plan_metricflow(
        self, state: _PlannedInvestigationState
    ) -> PlanActionExecution:
        profile = state.authorized_execution.access_profile
        product = self._metric_product(state.metric_name)
        if product is None:
            raise AccessDeniedError("The planned metric is not governed.")
        profile.authorize_product(product)
        if state.region is not None:
            profile.authorize_region(state.region)
        definition, decomposition, query_evidence, freshness = (
            self._semantic_driver_decomposition(
                state.metric_name,
                profile,
                baseline_period="2026-05",
                comparison_period="2026-06",
            )
        )
        state.definition = definition
        state.decomposition = decomposition
        state.query_evidence = query_evidence
        state.freshness = freshness
        if definition is None or decomposition is None or query_evidence is None:
            return PlanActionExecution(
                value=self._plan_limitation_response(state), payload=state, stop=True
            )
        if state.intent.route is AnalyticalRoute.DRIVER_DECOMPOSITION:
            return PlanActionExecution(
                value=self._driver_response_from_plan(state), payload=state
            )
        return PlanActionExecution(payload=state)

    def _run_plan_cited_evidence(
        self, state: _PlannedInvestigationState
    ) -> PlanActionExecution:
        if (
            state.definition is None
            or state.decomposition is None
            or state.query_evidence is None
            or state.region is None
            or state.seat_tier is None
        ):
            raise RuntimeError("Cited evidence requires a successful MetricFlow action.")
        profile = state.authorized_execution.access_profile
        product = cast(str, self._metric_product(state.metric_name))
        matching = next(
            (
                contribution
                for contribution in state.decomposition.contributions
                if contribution.region == state.region
                and contribution.seat_tier == state.seat_tier
            ),
            None,
        )
        if (
            matching is None
            or state.decomposition.reconciled_change != state.decomposition.net_change
            or state.decomposition.residual != 0
        ):
            return PlanActionExecution(
                value=self._unresolved_plan_response(state), payload=state, stop=True
            )
        state.matching_contribution = matching
        state.driver_window = DriverMovementWindow.from_periods(
            state.decomposition.baseline_period, state.decomposition.comparison_period
        )
        state.evidence_query = build_evidence_investigation_query(
            metric_label=self._metric_label(state.metric_name),
            product=product,
            region=state.region,
            seat_tier=state.seat_tier,
            driver_window=state.driver_window,
            canonical_time_rule=state.definition.time_rule,
            movement_direction="decline" if matching.change < 0 else "increase",
        )
        state.evidence_filter = profile.evidence_filter(
            product,
            state.region,
            seat_tier=state.seat_tier if state.scope_evidence_to_seat_tier else None,
            metric_name=state.metric_name,
            agent_user_id=state.authorized_execution.request.agent_user_id,
        )
        state.graph_filter = profile.graph_filter(
            product,
            state.region,
            seat_tier=state.seat_tier if state.scope_evidence_to_seat_tier else None,
        )
        if state.current_evidence_filter is not None:
            state.evidence_filter = state.current_evidence_filter
        state.cited_evidence = self.evidence_tools.retrieve_cited_evidence(
            query=state.evidence_query,
            evidence_filter=state.evidence_filter,
            graph_filter=state.graph_filter,
            metric_name=state.metric_name,
        )
        state.validated_evidence_revision_identities = (
            state.cited_evidence.authorized_revision_identities
        )
        return PlanActionExecution(
            payload=state,
            evidence_revision_keys=state.validated_evidence_revision_identities,
        )

    def _run_plan_lightrag(
        self, state: _PlannedInvestigationState
    ) -> PlanActionExecution:
        if state.cited_evidence is None:
            raise RuntimeError("LightRAG requires a successful cited-evidence action.")
        if state.current_graph_filter is not None and state.evidence_query is not None:
            state.graph_paths = self._traverse_graph(
                state.evidence_query,
                state.current_graph_filter,
                limit=3,
                metric_name=state.metric_name,
            )
        response = self._evidence_response_from_plan(state)
        return PlanActionExecution(
            value=response,
            payload=state,
            evidence_revision_keys=state.validated_evidence_revision_identities,
        )

    def _driver_response_from_plan(self, state: _PlannedInvestigationState):
        decomposition = state.decomposition
        leading = decomposition.contributions[0] if decomposition.contributions else None
        leading_text = "No segment movement was returned."
        if leading is not None:
            if decomposition.net_change > 0:
                leading_text = (
                    f"{leading.region} / {leading.seat_tier} Seat Tier Tenants are the leading "
                    f"observed movement, contributing {leading.change:+,} of the "
                    f"{decomposition.net_change:+,} net movement."
                )
            else:
                leading_text = (
                    f"{leading.region} / {leading.seat_tier} Seat Tier Tenants are the leading "
                    f"observed driver, contributing {leading.contribution_to_decline:,} of the "
                    f"{decomposition.decline:,} decline ({leading.percentage_of_decline:g}%)."
                )
        return GovernedAnalyticalResponse(
            answer=(
                "Driver Decomposition (observed, non-causal): "
                f"{self._metric_label(state.metric_name)} "
                f"moved from {decomposition.baseline_value:,} in May 2026 to "
                f"{decomposition.comparison_value:,} in June 2026 ({decomposition.net_change:+,}). "
                f"Semantic definition v{state.definition.semantic_version}: "
                f"{state.definition.definition} "
                f"{leading_text} The approved Region and Seat Tier contributions reconcile to the "
                "scoped movement; this observation does not establish causation."
            ),
            result_classification=ResultClassification.DRIVER_DECOMPOSITION,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=decomposition,
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                "This is a Driver Decomposition of observed canonical metric values, not a "
                "causal conclusion.",
                "Only Region and Seat Tier are approved dimensions for this decomposition.",
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    def _evidence_response_from_plan(self, state: _PlannedInvestigationState):
        assert state.cited_evidence is not None
        assert state.matching_contribution is not None
        assert state.evidence_filter is not None
        assert state.driver_window is not None
        evidence = build_evidence_answer(state.cited_evidence.documents)
        candidate_causal_factors = rank_candidate_causal_factors(
            state.cited_evidence.documents,
            driver_window=state.driver_window,
            access_filter=state.evidence_filter,
            contribution=state.matching_contribution,
            metric_name=state.metric_name,
        )
        selected_factor_id = self._active_selected_factor_id(
            state.authorized_execution.request, state.metric_name
        )
        if selected_factor_id is not None:
            matched = [
                card for card in candidate_causal_factors if card.factor_id == selected_factor_id
            ]
            if not matched:
                return self._selection_lost_response(state, selected_factor_id)
            candidate_causal_factors = matched
        scenario_percentage_points = (
            state.authorized_execution.request.opportunity_scenario_percentage_points
        )
        if scenario_percentage_points is not None:
            if selected_factor_id is None:
                return self._opportunity_sizing_missing_selection_response(
                    state, candidate_causal_factors
                )
            return self._opportunity_sizing_response(
                state, candidate_causal_factors[0], scenario_percentage_points
            )
        evidence_chain = self._empty_public_evidence_chain()
        graph_paths: list[Any] = []

        if not candidate_causal_factors:
            classification = ResultClassification.INCONCLUSIVE
            answer = (
                "Inconclusive: no rank-eligible Candidate Causal Factor was found for the "
                f"{state.region} {state.seat_tier} Seat Tier movement. "
                f"{evidence.support_explanation}"
            )
        else:
            top_card = candidate_causal_factors[0]
            classification = (
                ResultClassification.HYPOTHESIS
                if top_card.status == FactorSupportStatus.SUPPORTED
                else ResultClassification.INCONCLUSIVE
            )
            card_summaries = "; ".join(
                f"{card.category.value.replace('_', ' ')} ({card.status.value}): "
                f"{card.proposed_mechanism}"
                for card in candidate_causal_factors
            )
            answer = (
                f"{len(candidate_causal_factors)} Candidate Causal Factor(s) for the "
                f"{state.region} {state.seat_tier} Seat Tier "
                f"{self._metric_label(state.metric_name)} movement: {card_summaries} "
                f"{top_card.non_causal_caveat}"
            )
            evidence_chain = self._public_evidence_chain(state.cited_evidence.lightrag_chain)
            graph_paths = self._graph_path_citations(state.graph_paths)

        caveats = [
            (
                f"The {state.region} {state.seat_tier} Seat Tier result is an observed "
                "Driver Decomposition; each Candidate Causal Factor is a cited "
                "Hypothesis, not a causal conclusion."
            ),
            "Only evidence permitted by product, Region, Tenant, classification, and "
            "identifier entitlements was retrieved.",
        ]
        if (
            selected_factor_id is not None
            and len(candidate_causal_factors) == 1
            and candidate_causal_factors[0].sizing_eligible
        ):
            # A confirmed selection of this turn is itself a complete, valid action
            # (matching #75's Active Investigation contract) — this is guidance for
            # a possibly-forgotten scenario input, never an error classification.
            caveats.append(
                "This Candidate Causal Factor is Sizing Eligible; supply "
                "opportunity_scenario_percentage_points to receive an Opportunity "
                "Estimate for it."
            )

        return GovernedAnalyticalResponse(
            answer=answer,
            result_classification=classification,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=evidence,
            evidence_chain=evidence_chain,
            candidate_causal_factors=candidate_causal_factors,
            graph_paths=graph_paths,
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=caveats,
            trace_id=state.authorized_execution.trace_id,
        )

    def _selection_lost_response(
        self, state: _PlannedInvestigationState, selected_factor_id: str
    ) -> GovernedAnalyticalResponse:
        """The Active Investigation reference is a lookup key, never trusted content —
        this turn re-authorized and re-ranked from scratch, and the previously
        selected factor no longer matched an eligible, currently authorized card."""
        assert state.cited_evidence is not None
        evidence = build_evidence_answer(state.cited_evidence.documents)
        return GovernedAnalyticalResponse(
            answer=(
                "Limitation: the previously selected Candidate Causal Factor for the "
                f"{state.region} {state.seat_tier} Seat Tier "
                f"{self._metric_label(state.metric_name)} investigation could not be "
                "revalidated against currently authorized evidence and entitlements, "
                "so no Candidate Causal Factor is returned this turn."
            ),
            result_classification=ResultClassification.LIMITATION,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=evidence,
            evidence_chain=self._empty_public_evidence_chain(),
            candidate_causal_factors=[],
            graph_paths=[],
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                (
                    "The Active Investigation reference is never trusted on its own; this "
                    "turn re-authorized the Agent User and re-ranked evidence, and Factor ID "
                    f"{selected_factor_id!r} no longer matched an eligible, currently "
                    "authorized Candidate Causal Factor."
                ),
                "Only evidence permitted by product, Region, Tenant, classification, "
                "and identifier entitlements was retrieved.",
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    def _opportunity_sizing_missing_selection_response(
        self,
        state: _PlannedInvestigationState,
        candidate_causal_factors: list[CandidateCausalFactor],
    ) -> GovernedAnalyticalResponse:
        """Opportunity sizing requires an explicit, revalidated Factor ID selection.

        Unlike `_selection_lost_response` (a selection was made and specifically
        failed to revalidate), this is "no selection was ever made" — the currently
        ranked candidates are shown as-is so the analyst can see which Factor ID to
        select next turn.
        """
        assert state.cited_evidence is not None
        evidence = build_evidence_answer(state.cited_evidence.documents)
        return GovernedAnalyticalResponse(
            answer=(
                "Limitation: Opportunity sizing requires an explicit, revalidated "
                "selected Factor ID; none was provided or resolved for this turn, "
                f"so no Opportunity Estimate is returned for the {state.region} "
                f"{state.seat_tier} Seat Tier {self._metric_label(state.metric_name)} "
                "investigation."
            ),
            result_classification=ResultClassification.LIMITATION,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=evidence,
            evidence_chain=self._empty_public_evidence_chain(),
            candidate_causal_factors=candidate_causal_factors,
            graph_paths=[],
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                "Opportunity sizing requires both an explicit selected Factor ID and "
                "an analyst-supplied absolute percentage-point Opportunity Scenario.",
                "Only evidence permitted by product, Region, Tenant, classification, "
                "and identifier entitlements was retrieved.",
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    def _opportunity_sizing_gap_response(
        self, state: _PlannedInvestigationState, factor: CandidateCausalFactor
    ) -> GovernedAnalyticalResponse:
        """A factor without a governed event-and-audience mapping remains a
        Hypothesis and offers a data-team mapping request instead of an estimate."""
        assert state.cited_evidence is not None
        evidence = build_evidence_answer(state.cited_evidence.documents)
        return GovernedAnalyticalResponse(
            answer=(
                f"Hypothesis: Candidate Causal Factor {factor.factor_id!r} has no "
                "dbt/MetricFlow-governed event-and-audience mapping, so it is not "
                "Sizing Eligible. It remains a Hypothesis; a data-team mapping "
                "request is offered instead of an Opportunity Estimate."
            ),
            result_classification=ResultClassification.HYPOTHESIS,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=evidence,
            evidence_chain=self._public_evidence_chain(state.cited_evidence.lightrag_chain),
            candidate_causal_factors=[factor],
            graph_paths=self._graph_path_citations(state.graph_paths),
            opportunity_sizing_gap=OpportunitySizingGap(
                factor_id=factor.factor_id, category=factor.category
            ),
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                (
                    "This category has no reviewed dbt/MetricFlow event-and-audience "
                    "mapping; the Eligible Population is never inferred from documents "
                    "or the evidence graph, so Opportunity Estimate is not offered."
                ),
                "Only evidence permitted by product, Region, Tenant, classification, "
                "and identifier entitlements was retrieved.",
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    def _opportunity_sizing_response(
        self,
        state: _PlannedInvestigationState,
        factor: CandidateCausalFactor,
        scenario_percentage_points: float,
    ) -> GovernedAnalyticalResponse:
        """Ground an analyst-supplied scenario in the governed Eligible Population
        for this factor's category/driver metric; never a causal effect or forecast."""
        metric_name = sizing_eligible_metric_name(factor.category, state.metric_name)
        if metric_name is None:
            return self._opportunity_sizing_gap_response(state, factor)
        eligible_population, _freshness = self._semantic_eligible_population(
            metric_name,
            state.authorized_execution.access_profile,
            region=factor.documented_change.region,
            seat_tier=factor.documented_change.seat_tier,
        )
        if eligible_population is None:
            return self._opportunity_sizing_gap_response(state, factor)

        baseline_rate_percentage = (
            factor.documented_change.comparison_value / eligible_population * 100
            if eligible_population
            else 0.0
        )
        raw_incremental = eligible_population * scenario_percentage_points / 100
        incremental_product_users = round(raw_incremental)
        assert state.driver_window is not None
        assert state.cited_evidence is not None
        evidence = build_evidence_answer(state.cited_evidence.documents)
        caveats = [
            "This Opportunity Estimate is a conditional projection from an "
            "analyst-supplied scenario assumption, not a causal effect, forecast, "
            "or observed uplift.",
            "Only evidence permitted by product, Region, Tenant, classification, "
            "and identifier entitlements was retrieved.",
        ]
        if raw_incremental != incremental_product_users:
            caveats.append(
                f"Incremental Product Users is rounded from {raw_incremental:.2f} to "
                "a whole Product User count."
            )
        return GovernedAnalyticalResponse(
            answer=(
                f"Opportunity Estimate for Candidate Causal Factor {factor.factor_id!r}: "
                f"a {scenario_percentage_points:+.2f} percentage-point scenario against "
                f"{eligible_population} Eligible Population Product Users projects "
                f"{incremental_product_users} incremental Product Users."
            ),
            result_classification=ResultClassification.OPPORTUNITY_ESTIMATE,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=evidence,
            evidence_chain=self._public_evidence_chain(state.cited_evidence.lightrag_chain),
            candidate_causal_factors=[factor],
            graph_paths=self._graph_path_citations(state.graph_paths),
            opportunity_estimate=OpportunityEstimate(
                factor_id=factor.factor_id,
                baseline_rate_percentage=round(baseline_rate_percentage, 2),
                eligible_population=eligible_population,
                scenario_percentage_point_change=scenario_percentage_points,
                incremental_product_users=incremental_product_users,
                formula=(
                    "incremental_product_users = eligible_population × "
                    "scenario_percentage_point_change ÷ 100"
                ),
                scenario_window_start=state.driver_window.start,
                scenario_window_end=state.driver_window.end,
                non_causal_caveat=(
                    "This Opportunity Estimate is a conditional projection grounded in "
                    "the governed dbt/MetricFlow event-and-audience mapping, not a "
                    "causal effect, forecast, or observed uplift."
                ),
            ),
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=caveats,
            trace_id=state.authorized_execution.trace_id,
        )

    def _unresolved_plan_response(self, state: _PlannedInvestigationState):
        return GovernedAnalyticalResponse(
            answer=(
                f"Inconclusive: the validated Driver Decomposition does not resolve the "
                f"{state.region} {state.seat_tier} Seat Tier scope, so no evidence was retrieved."
            ),
            result_classification=ResultClassification.INCONCLUSIVE,
            canonical_definition=state.definition,
            semantic_query_evidence=state.query_evidence,
            driver_decomposition=state.decomposition,
            evidence=build_evidence_answer([]),
            source_freshness=state.freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                "Evidence retrieval requires a reconciled Driver Decomposition with a matching "
                "Region and Seat Tier contribution."
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    def _plan_limitation_response(self, state: _PlannedInvestigationState):
        freshness = state.freshness or SourceFreshness(
            validated_at=datetime.now(UTC), maximum_age_seconds=86_400, is_current=False
        )
        return GovernedAnalyticalResponse(
            answer=(
                f"The governed {self._metric_label(state.metric_name)} investigation was limited "
                "because its current semantic action did not produce usable evidence."
            ),
            result_classification=ResultClassification.LIMITATION,
            source_freshness=freshness,
            effective_access_scope=state.authorized_execution.effective_scope,
            caveats=[
                "No subsequent evidence action was authorized without a current semantic result."
            ],
            trace_id=state.authorized_execution.trace_id,
        )

    @staticmethod
    def _conversation_metric_name(response: GovernedAnalyticalResponse) -> str | None:
        for value in (
            response.canonical_definition.name if response.canonical_definition else None,
            response.semantic_query_evidence.metric_name
            if response.semantic_query_evidence
            else None,
            response.driver_decomposition.metric_name if response.driver_decomposition else None,
            response.metric_definition_gap.requested_metric_name
            if response.metric_definition_gap
            else None,
        ):
            if value:
                return value
        return None

    @classmethod
    def _conversation_turn(
        cls, request: AnswerQuestionRequest, response: GovernedAnalyticalResponse
    ) -> ConversationTurn:
        return ConversationTurn(
            turn_id=str(uuid4()),
            question=request.question,
            result_classification=response.result_classification,
            metric_name=cls._conversation_metric_name(response),
            trace_id=response.trace_id,
            created_at=datetime.now(UTC),
            lead_agent_metadata=response.lead_agent_metadata,
        )

    @staticmethod
    def _next_active_investigation_factor_id(
        request: AnswerQuestionRequest,
        response: GovernedAnalyticalResponse,
        metric_name: str | None,
        prior: ConversationSummary | None,
    ) -> str | None:
        """Decide what Active Investigation reference (if any) carries into the next turn.

        Never stores card content — only the opaque `factor_id` lookup key, and only
        when this turn itself resolved a selection (explicit or reasserted) and it was
        successfully revalidated against a currently eligible, authorized candidate.
        """
        if (
            response.result_classification == ResultClassification.LIMITATION
            and response.candidate_causal_factors is not None
        ):
            # This turn specifically tried and failed to revalidate a selection —
            # forget it rather than keep retrying a permanently invalid reference.
            return None
        if (
            metric_name is not None
            and response.candidate_causal_factors is not None
            and len(response.candidate_causal_factors) == 1
        ):
            resolved = AnswerQuestionService._active_selected_factor_id(request, metric_name)
            if resolved is not None:
                return response.candidate_causal_factors[0].factor_id
        # Only carry the stored reference forward when this turn's metric context is
        # unchanged from when it was stored — otherwise a detour to an unrelated metric
        # would pair a stale factor_id with a new metric_name, and a later turn on that
        # new metric could wrongly reject it as a "lost" selection no one ever made.
        if prior is not None and prior.metric_name == metric_name:
            return prior.active_investigation_factor_id
        return None

    @classmethod
    def _conversation_summary(
        cls, request: AnswerQuestionRequest, response: GovernedAnalyticalResponse
    ) -> ConversationSummary:
        prior = request.conversation_context.summary if request.conversation_context else None
        metric_name = cls._conversation_metric_name(response) or (
            prior.metric_name if prior else None
        )
        active_investigation_factor_id = cls._next_active_investigation_factor_id(
            request, response, metric_name, prior
        )
        revision_ids = list(prior.evidence_revision_ids) if prior else []
        if response.evidence is not None:
            revision_ids.extend(
                f"{citation.source_document_id}@{citation.source_revision}"
                for citation in response.evidence.citations
            )
        conclusions = list(prior.qualified_conclusions) if prior else []
        conclusion = response.result_classification.value
        if response.evidence is not None:
            conclusion = f"{conclusion}:{response.evidence.support_status.value}"
        conclusions.append(conclusion)
        return ConversationSummary(
            agent_user_goal=(
                prior.agent_user_goal
                if prior and prior.agent_user_goal
                else cls._conversation_goal(response, metric_name)
            ),
            resolved_scope=response.effective_access_scope,
            metric_name=metric_name,
            active_investigation_factor_id=active_investigation_factor_id,
            evidence_revision_ids=list(dict.fromkeys(revision_ids))[-32:],
            qualified_conclusions=list(dict.fromkeys(conclusions))[-32:],
            open_questions=list(prior.open_questions) if prior else [],
            workflow_state=response.result_classification.value,
        )

    @staticmethod
    def _conversation_goal(response: GovernedAnalyticalResponse, metric_name: str | None) -> str:
        if metric_name is not None:
            return f"Resolve governed metric {metric_name}."
        return f"Resolve governed {response.result_classification.value} analysis."

    def readiness(self) -> dict[str, object]:
        """Expose model dependency state without exposing request or credential data."""
        local_model = local_model_readiness(self.local_model)
        evidence_readiness = getattr(self.evidence_store, "readiness", None)
        evidence = (
            evidence_readiness()
            if evidence_readiness is not None
            else {
                "status": "ready",
                "external": False,
                "embedding": embedding_readiness(),
            }
        )
        qdrant = {key: value for key, value in evidence.items() if key != "embedding"}
        embedding = evidence.get("embedding", embedding_readiness())
        reranker = reranker_readiness(self.evidence_reranker)
        trace_delivery = self.trace_delivery_health.readiness()
        dependency_unavailable = (
            local_model["status"] == "unavailable"
            or qdrant["status"] == "unavailable"
            or embedding["status"] == "unavailable"
            or reranker["status"] not in {"ready", "configured"}
        )
        return {
            "status": (
                "unavailable"
                if dependency_unavailable
                else "degraded"
                if trace_delivery["status"] == "unavailable"
                else "ready"
            ),
            "local_model": local_model,
            "qdrant": qdrant,
            "embedding": embedding,
            "reranker": reranker,
            "trace_delivery": trace_delivery,
        }

    def _available_metric_names_for_request(
        self, request: AnswerQuestionRequest
    ) -> tuple[str, ...]:
        """Filter canonical metric candidates by the request's resolved Access Profile."""
        access_profile = resolve_access_profile(request.agent_user_id)
        return tuple(
            metric_name
            for metric_name in self.semantic_gateway.available_metric_names()
            if self._metric_product(metric_name) in access_profile.products
        )

    def _route_for_validated_intent(
        self, request: AnswerQuestionRequest, metric_name: str | None
    ) -> AnalyticalRoute:
        return self._route_for_intent(
            request,
            metric_name,
            canonical_metric_names=self.semantic_gateway.available_metric_names(),
        )

    def _draft_evidence_response(
        self, response: GovernedAnalyticalResponse
    ) -> GovernedAnalyticalResponse:
        """Let an optional model rewrite only prose backed by safe citations."""
        if self.evidence_drafting_adapter is None or response.evidence is None:
            return response
        if not response.evidence.citations:
            return response
        try:
            draft = self.evidence_drafting_adapter.draft(response)
            draft = validate_local_model_draft(response, draft)
        except LocalModelError:
            return GovernedAnalyticalResponse(
                answer=(
                    "The governed evidence response was withheld because the configured local "
                    "drafting model output could not be validated."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=response.source_freshness,
                effective_access_scope=response.effective_access_scope,
                caveats=[
                    "The local model failed closed; no model-generated prose or additional "
                    "evidence was returned."
                ],
                lead_agent_metadata=response.lead_agent_metadata,
                trace_id=response.trace_id,
            )
        return response.model_copy(update={"answer": draft.answer})

    def _answer_legacy_question(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        request = authorized_execution.request
        access_profile = authorized_execution.access_profile
        scope = authorized_execution.effective_scope
        trace_id = authorized_execution.trace_id

        if self._requests_direct_identifier(request.question):
            return self._answer_direct_identifier_request(
                request=request,
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        catalog_entity_name = self._requested_catalog_entity(request)
        if catalog_entity_name is not None:
            return self._answer_catalog_ownership(
                entity_name=catalog_entity_name,
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        metric_name = self._requested_metric_name(request)
        if metric_name is None:
            artifact = self.semantic_gateway.artifact_store.load()
            return GovernedAnalyticalResponse(
                answer=(
                    "This first delivery supports governed metric-definition questions, starting "
                    "with Jira and Confluence New PEU and New MAU. Name a metric to check its "
                    "semantic status."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=self.semantic_gateway.freshness(artifact),
                effective_access_scope=scope,
                caveats=["The request did not identify a governed metric."],
                trace_id=trace_id,
            )

        if metric_name in {
            "jira_new_peu",
            "confluence_new_peu",
            "jira_new_mau",
            "confluence_new_mau",
        } and (self._requests_may_to_june_driver_decomposition(request.question)):
            return self._answer_metric_driver_decomposition(
                metric_name=metric_name,
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        return self._answer_canonical_definition(
            authorized_execution,
            AnalyticalIntent(
                route=AnalyticalRoute.CANONICAL_DEFINITION,
                metric_name=metric_name,
            ),
        )

    def _answer_canonical_definition(
        self,
        authorized_execution: AuthorizedExecution,
        intent: AnalyticalIntent,
    ) -> GovernedAnalyticalResponse:
        request = authorized_execution.request
        scope = authorized_execution.effective_scope
        access_profile = authorized_execution.access_profile
        trace_id = authorized_execution.trace_id
        metric_name = intent.metric_name
        if metric_name is None:
            raise ValueError("Canonical-definition execution requires a metric name.")
        metric_product = self._metric_product(metric_name)
        if metric_product is not None:
            access_profile.authorize_product(metric_product)
        definition, freshness = self._semantic_canonical_definition(metric_name)
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

        query_evidence, freshness = self._semantic_execute_scoped_metric(
            metric_name, access_profile
        )
        if query_evidence is None:
            return GovernedAnalyticalResponse(
                answer=(
                    f"{self._metric_label(metric_name)} cannot be returned as canonical because "
                    "validation is not current."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=trace_id,
            )

        if metric_name.endswith("_new_mau"):
            answer = (
                f"{metric_product} New MAU is a New PEU with at least one Visit to "
                f"{metric_product} in the same calendar month as first paid enablement. "
                "A Visit to another product does not qualify the Product User."
            )
            caveats = [
                "This is a canonical definition, not a count for a particular period.",
                "The grain is Product User in a Tenant and product; it is not Person-level.",
                "Only same-product Visits in the first paid-enablement calendar month qualify.",
            ]
        else:
            answer = (
                f"{metric_product} New PEU is a Product User's first-ever Paid Enablement for "
                f"{metric_product}. "
                "Later restorations of paid access do not create another New PEU."
            )
            caveats = [
                "This is a canonical definition, not a count for a particular period.",
                "The grain is Product User in a Tenant and product; it is not Person-level.",
            ]

        caveats.append(
            "DataHub catalog availability does not affect canonical metric logic; "
            "catalog-dependent ownership, classification, and discovery answers disclose "
            "degradation separately."
        )

        return GovernedAnalyticalResponse(
            answer=answer,
            result_classification=ResultClassification.CANONICAL_DEFINITION,
            canonical_definition=definition,
            semantic_query_evidence=query_evidence,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=caveats,
            trace_id=trace_id,
        )

    def _answer_driver_decomposition(
        self, authorized_execution: AuthorizedExecution, intent: AnalyticalIntent
    ) -> GovernedAnalyticalResponse:
        metric_name = intent.metric_name
        if metric_name is None:
            raise ValueError("Driver-decomposition execution requires a metric name.")
        return self._answer_metric_driver_decomposition(
            metric_name=metric_name,
            scope=authorized_execution.effective_scope,
            access_profile=authorized_execution.access_profile,
            trace_id=authorized_execution.trace_id,
        )

    def _answer_causal_redirect_specialist(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        """The causal-estimate workflow is retired; redirect to evidence-first hypotheses."""
        artifact = self.semantic_gateway.artifact_store.load()
        return GovernedAnalyticalResponse(
            answer=(
                "This service no longer produces Causal Estimates, treatment effects, or root "
                "causes. Ask to investigate the Driver Decomposition for a metric to review "
                "documented, cited Hypotheses instead."
            ),
            result_classification=ResultClassification.LIMITATION,
            source_freshness=self.semantic_gateway.freshness(artifact),
            effective_access_scope=authorized_execution.effective_scope,
            caveats=[
                "Causal-analysis requests are retired; no causal estimator ran for this request."
            ],
            trace_id=authorized_execution.trace_id,
        )

    def _answer_catalog_specialist(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        entity_name = self._requested_catalog_entity(authorized_execution.request)
        if entity_name is None:
            return self._answer_limitation_specialist(authorized_execution)
        return self._answer_catalog_ownership(
            entity_name=entity_name,
            scope=authorized_execution.effective_scope,
            access_profile=authorized_execution.access_profile,
            trace_id=authorized_execution.trace_id,
        )

    def _answer_direct_identifier_specialist(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        return self._answer_direct_identifier_request(
            request=authorized_execution.request,
            scope=authorized_execution.effective_scope,
            access_profile=authorized_execution.access_profile,
            trace_id=authorized_execution.trace_id,
        )

    def _answer_limitation_specialist(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        artifact = self.semantic_gateway.artifact_store.load()
        return GovernedAnalyticalResponse(
            answer=(
                "This first delivery supports governed metric-definition questions, starting "
                "with Jira and Confluence New PEU and New MAU. Name a metric to check its "
                "semantic status."
            ),
            result_classification=ResultClassification.LIMITATION,
            source_freshness=self.semantic_gateway.freshness(artifact),
            effective_access_scope=authorized_execution.effective_scope,
            caveats=["The request did not identify a governed metric."],
            trace_id=authorized_execution.trace_id,
        )

    def _answer_metric_definition_gap_specialist(
        self, authorized_execution: AuthorizedExecution, intent: AnalyticalIntent
    ) -> GovernedAnalyticalResponse:
        metric_name = intent.metric_name
        if metric_name is None:
            return self._answer_limitation_specialist(authorized_execution)
        metric_product = self._metric_product(metric_name)
        if metric_product is not None:
            authorized_execution.access_profile.authorize_product(metric_product)
        definition, freshness = self._semantic_canonical_definition(metric_name)
        if definition is not None:
            return self._answer_canonical_definition(authorized_execution, intent)
        if not freshness.is_current:
            return GovernedAnalyticalResponse(
                answer=(
                    f"{metric_name} cannot be returned as canonical because the dbt/MetricFlow "
                    "semantic artifact is failed, stale, or unavailable."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=authorized_execution.effective_scope,
                caveats=["Run dbt validation and refresh the semantic artifact before using it."],
                trace_id=authorized_execution.trace_id,
            )
        return self._metric_definition_gap_response(
            request=authorized_execution.request,
            metric_name=metric_name,
            freshness=freshness,
            scope=authorized_execution.effective_scope,
            trace_id=authorized_execution.trace_id,
            access_profile=authorized_execution.access_profile,
        )

    def _answer_intent_clarification(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        artifact = self.semantic_gateway.artifact_store.load()
        return GovernedAnalyticalResponse(
            answer=(
                "I could not validate the requested metric or analysis type. Name a governed "
                "metric to check its semantic status."
            ),
            result_classification=ResultClassification.LIMITATION,
            source_freshness=self.semantic_gateway.freshness(artifact),
            effective_access_scope=authorized_execution.effective_scope,
            caveats=[
                "The request was not sent to a semantic query, evidence retrieval, graph "
                "traversal, or direct-identifier handler."
            ],
            trace_id=authorized_execution.trace_id,
        )

    def _record_trace(
        self,
        request: AnswerQuestionRequest,
        response: GovernedAnalyticalResponse,
        trace_context: TraceContext,
    ) -> None:
        access_profile = resolve_access_profile(request.agent_user_id)
        source_versions: dict[str, str] = {}
        try:
            artifact = self.semantic_gateway.artifact_store.load()
            source_versions.update(
                semantic_version=artifact.semantic_version,
                semantic_manifest_sha256=artifact.semantic_manifest_sha256,
            )
        except (OSError, ValueError):
            source_versions["semantic_artifact"] = "unavailable"
        if response.canonical_definition is not None:
            source_versions.setdefault(
                "semantic_version", response.canonical_definition.semantic_version
            )
        if response.semantic_query_evidence is not None:
            source_versions.setdefault(
                "semantic_manifest_sha256",
                response.semantic_query_evidence.artifact_sha256,
            )
        if response.evidence is not None or response.graph_paths is not None:
            source_versions["evidence_corpus"] = "synthetic-v1"

        retrieval_used = (
            response.evidence is not None or response.direct_identifier_answer is not None
        )
        tool_outcomes = {
            "semantic_query": (
                "success" if response.semantic_query_evidence is not None else "not_used"
            ),
            "retrieval": "success" if retrieval_used else "not_used",
            "graph": "success" if response.graph_paths is not None else "not_used",
            "direct_identifier_audit": (
                "success" if response.direct_identifier_audit is not None else "not_used"
            ),
        }
        retrieval_scores = (
            tuple(float(score) for score in getattr(self.evidence_store, "last_scores", ()))
            if retrieval_used
            else ()
        )
        trace = TraceRecord(
            trace_id=response.trace_id,
            request_route="answer_question",
            response_classification=response.result_classification.value,
            policy_fingerprint=policy_fingerprint(access_profile),
            source_versions=source_versions,
            tool_outcomes=tool_outcomes,
            retrieval_scores=retrieval_scores,
            evaluation_outcome="not_evaluated",
            response=response.model_dump(mode="json"),
            latency_ms=trace_context.latency_ms,
            configuration_versions=self._trace_configuration_versions(),
            lead_agent_metadata=response.lead_agent_metadata,
            conversation_id=response.conversation_id,
            has_active_investigation_selection=trace_context.has_active_investigation_selection,
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        self._deliver_trace(trace)

    def _record_authorization_denial(
        self, request: AnswerQuestionRequest, trace_context: TraceContext
    ) -> str:
        try:
            access_profile = resolve_access_profile(request.agent_user_id)
            fingerprint = policy_fingerprint(access_profile)
        except UnknownAgentUserError:
            fingerprint = "unknown-agent"
        source_versions: dict[str, str] = {}
        try:
            artifact = self.semantic_gateway.artifact_store.load()
            source_versions = {
                "semantic_version": artifact.semantic_version,
                "semantic_manifest_sha256": artifact.semantic_manifest_sha256,
            }
        except (OSError, ValueError):
            source_versions["semantic_artifact"] = "unavailable"
        trace = TraceRecord(
            trace_id=str(uuid4()),
            request_route="answer_question",
            response_classification="safe_refusal",
            policy_fingerprint=fingerprint,
            source_versions=source_versions,
            tool_outcomes={
                "semantic_query": "not_used",
                "retrieval": "not_used",
                "graph": "not_used",
                "direct_identifier_audit": "not_used",
            },
            retrieval_scores=(),
            evaluation_outcome="not_evaluated",
            response={
                "result_classification": "safe_refusal",
                "error_code": "access_denied",
            },
            latency_ms=trace_context.latency_ms,
            configuration_versions=self._trace_configuration_versions(),
            lead_agent_metadata=trace_context.lead_agent_metadata,
            conversation_id=trace_context.conversation_id or request.conversation_id,
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        self._deliver_trace(trace)
        return trace.trace_id

    def _record_dependency_failure(
        self, request: AnswerQuestionRequest, trace_context: TraceContext
    ) -> str:
        trace_id = str(uuid4())
        source_versions: dict[str, str] = {}
        try:
            artifact = self.semantic_gateway.artifact_store.load()
            source_versions = {
                "semantic_version": artifact.semantic_version,
                "semantic_manifest_sha256": artifact.semantic_manifest_sha256,
            }
        except (OSError, ValueError):
            source_versions["semantic_artifact"] = "unavailable"
        trace = TraceRecord(
            trace_id=trace_id,
            request_route="answer_question",
            response_classification="safe_refusal",
            policy_fingerprint=self._policy_fingerprint_or_unknown(request),
            source_versions=source_versions,
            tool_outcomes={
                span.name: "error" if span.status == "error" else "success"
                for span in trace_context.tool_spans
            },
            retrieval_scores=(),
            evaluation_outcome="not_evaluated",
            response={
                "result_classification": "safe_refusal",
                "error_code": "dependency_unavailable",
            },
            latency_ms=trace_context.latency_ms,
            configuration_versions=self._trace_configuration_versions(),
            lead_agent_metadata=trace_context.lead_agent_metadata,
            conversation_id=trace_context.conversation_id or request.conversation_id,
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        self._deliver_trace(trace)
        return trace_id

    def _deliver_trace(self, trace: TraceRecord) -> None:
        """Deliver a redacted trace without allowing observability to fail the turn."""
        if _EVALUATION_CAPTURE_ACTIVE.get():
            _CURRENT_EVALUATION_TRACE.set(trace)
        try:
            self.trace_sink.record(trace)
        except Exception as error:
            self.trace_delivery_health.record_failure(error)
        else:
            self.trace_delivery_health.record_success()

    def _trace_configuration_versions(self) -> dict[str, str]:
        """Record safe component identities without exposing transport configuration."""
        return {
            "workflow": "governed-response-v1",
            "intent_model": self._component_configuration_version(self.local_model),
            "evidence_reranker": self._component_configuration_version(self.evidence_reranker),
        }

    @staticmethod
    def _component_configuration_version(component: object | None) -> str:
        if component is None:
            return "none"
        model_name = getattr(component, "model_name", None)
        if isinstance(model_name, str) and model_name:
            return model_name
        return type(component).__name__

    def _policy_fingerprint_or_unknown(self, request: AnswerQuestionRequest) -> str:
        try:
            return policy_fingerprint(resolve_access_profile(request.agent_user_id))
        except UnknownAgentUserError:
            return "unknown-agent"

    def _semantic_is_current(self, authorized_execution: AuthorizedExecution | None = None) -> bool:
        """Provide planning with the same semantic freshness boundary as execution."""
        del authorized_execution
        try:
            artifact = self.semantic_gateway.artifact_store.load()
        except (OSError, ValueError):
            return False
        return self.semantic_gateway.freshness(artifact).is_current

    def _semantic_canonical_definition(self, metric_name: str):
        with trace_span(
            "semantic_definition",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.canonical_definition(metric_name)

    def _semantic_execute_scoped_metric(self, metric_name: str, access_profile, **scope_kwargs):
        with trace_span(
            "semantic_query",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.execute_scoped_metric(
                metric_name, access_profile, **scope_kwargs
            )

    def _semantic_eligible_population(
        self,
        metric_name: str,
        access_profile,
        *,
        region: str,
        seat_tier: str,
    ):
        with trace_span(
            "semantic_eligible_population",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.eligible_population(
                metric_name, access_profile, region=region, seat_tier=seat_tier
            )

    def _semantic_driver_decomposition(
        self,
        metric_name: str,
        access_profile,
        *,
        baseline_period: str,
        comparison_period: str,
    ):
        with trace_span(
            "semantic_driver_decomposition",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.driver_decomposition(
                metric_name,
                access_profile,
                baseline_period=baseline_period,
                comparison_period=comparison_period,
            )

    def _retrieve_evidence(self, query: str, access_filter, *, limit: int):
        with trace_span(
            "evidence_retrieval",
            kind="tool",
            attributes={"result_limit": limit},
        ):
            if self.lightrag_adapter is None:
                raise LightRAGAuthorizationError(
                    "Governed LightRAG evidence retrieval is unavailable."
                )
            lightrag_adapter = require_governed_lightrag_adapter(self.lightrag_adapter)
            source_documents = cast(
                Iterable[EvidenceDocument] | None,
                getattr(self.evidence_store, "documents", None),
            )
            revision_reader = cast(
                Callable[[EvidenceAccessFilter], Iterable[EvidenceDocument]] | None,
                getattr(self.evidence_store, "authorized_revisions", None),
            )
            if not source_documents and callable(revision_reader):
                source_documents = revision_reader(access_filter)
            if source_documents is None:
                raise LightRAGAuthorizationError(
                    "LightRAG requires an authoritative evidence revision source."
                )
            authorized_documents = [
                document for document in source_documents if access_filter.allows(document)
            ]
            if not authorized_documents:
                return []
            authorized_scope = AuthorizedEvidenceRevisionSet.from_documents(
                authorized_documents,
                access_filter,
                revision_source=revision_reader,
            )
            references = validate_authorized_lightrag_references(
                lightrag_adapter.retrieve(
                    query,
                    authorized_scope,
                    access_filter,
                    limit=limit,
                ),
                authorized_scope,
                access_filter,
            )
            allowed_document_ids = {
                reference.source_document_id for reference in references
            }
            allowed_revision_keys = {
                (
                    reference.source_document_id,
                    reference.source_revision,
                    reference.chunk_id,
                )
                for reference in references
            }
            if not allowed_document_ids:
                return []
            scoped_retriever = getattr(self.evidence_store, "retrieve_scoped", None)
            if callable(scoped_retriever):
                documents = cast(Any, scoped_retriever)(
                    query,
                    access_filter,
                    allowed_document_ids,
                    limit=limit,
                    authorized_revision_keys=allowed_revision_keys,
                )
            else:
                raise LightRAGAuthorizationError(
                    "LightRAG requires a backend-enforced scoped evidence retriever."
                )
            validated_documents = []
            for document in documents:
                if not isinstance(document, EvidenceDocument):
                    raise LightRAGAuthorizationError(
                        "LightRAG scoped retrieval returned an invalid evidence document."
                    )
                if _evidence_revision_key(document) not in allowed_revision_keys:
                    raise LightRAGAuthorizationError(
                        "LightRAG scoped retrieval returned an unauthorized evidence revision."
                    )
                if not access_filter.allows(document):
                    raise LightRAGAuthorizationError(
                        "LightRAG scoped retrieval returned evidence outside the current policy."
                    )
                validated_documents.append(document)
            return validated_documents[:limit]

    def _catalog_get(self, entity_name: str):
        with trace_span(
            "catalog_lookup",
            kind="tool",
            attributes={"entity_name": entity_name},
        ):
            return self.catalog_store.get(entity_name)

    def _audit_direct_identifiers(
        self,
        *,
        trace_id: str,
        agent_user_id: str,
        scope,
        access_profile,
        outcome: str,
        returned_count: int,
        maximum_results: int,
    ):
        with trace_span(
            "direct_identifier_audit",
            kind="tool",
            attributes={"returned_count": returned_count, "result_limit": maximum_results},
        ):
            return self.direct_identifier_audit_recorder.record(
                trace_id=trace_id,
                agent_user_id=agent_user_id,
                scope=scope,
                policy_fingerprint=policy_fingerprint(access_profile),
                outcome=outcome,
                returned_count=returned_count,
                maximum_results=maximum_results,
            )

    def _answer_metric_driver_decomposition(
        self, *, metric_name: str, scope, access_profile, trace_id: str
    ):
        definition, decomposition, query_evidence, freshness = self._semantic_driver_decomposition(
            metric_name,
            access_profile,
            baseline_period="2026-05",
            comparison_period="2026-06",
        )
        if definition is None or decomposition is None or query_evidence is None:
            return GovernedAnalyticalResponse(
                answer=(
                    f"{self._metric_label(metric_name)} cannot be decomposed as canonical because "
                    "semantic validation is not current."
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
            if decomposition.net_change > 0:
                leading_text = (
                    f"{leading.region} / {leading.seat_tier} Seat Tier Tenants are the leading "
                    "observed movement, contributing "
                    f"{leading.change:+,} of the {decomposition.net_change:+,} net movement."
                )
            else:
                leading_text = (
                    f"{leading.region} / {leading.seat_tier} Seat Tier Tenants are the leading "
                    "observed driver, contributing "
                    f"{leading.contribution_to_decline:,} of the {decomposition.decline:,} decline "
                    f"({leading.percentage_of_decline:g}%)."
                )
        return GovernedAnalyticalResponse(
            answer=(
                f"Driver Decomposition (observed, non-causal): {self._metric_label(metric_name)} "
                "moved from "
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

    def _answer_catalog_ownership(self, *, entity_name, scope, access_profile, trace_id: str):
        product = self._catalog_product(entity_name)
        if product is None:
            raise AccessDeniedError(
                "Catalog ownership requires an entity with a known governed product scope."
            )
        access_profile.authorize_product(product)
        artifact = self.semantic_gateway.artifact_store.load()
        allowed_entities = {
            entity_name
            for metric in artifact.metrics
            for entity_name in (metric.name, metric.model_name)
        }
        if entity_name not in allowed_entities:
            raise AccessDeniedError(
                "Catalog ownership is limited to entities in the validated semantic artifact."
            )
        source_freshness = self.semantic_gateway.freshness(artifact)
        if self.catalog_store is None:
            return self._catalog_limitation(
                entity_name=entity_name,
                scope=scope,
                source_freshness=source_freshness,
                trace_id=trace_id,
                detail="DataHub catalog is not configured.",
            )
        try:
            metadata = self._catalog_get(entity_name)
        except DataHubCatalogUnavailableError as error:
            return self._catalog_limitation(
                entity_name=entity_name,
                scope=scope,
                source_freshness=source_freshness,
                trace_id=trace_id,
                detail=str(error),
            )
        if metadata is None:
            return self._catalog_limitation(
                entity_name=entity_name,
                scope=scope,
                source_freshness=source_freshness,
                trace_id=trace_id,
                detail=f"No published DataHub metadata was found for {entity_name}.",
                available=True,
            )
        if metadata.classification not in access_profile.permitted_classifications:
            raise AccessDeniedError("Access Profile is not entitled to this catalog metadata.")
        metadata_payload = metadata.model_dump(mode="json")
        metadata_payload["entity_name"] = entity_name
        catalog_metadata = CatalogMetadata.model_validate(metadata_payload)
        return GovernedAnalyticalResponse(
            answer=(
                f"DataHub ownership: {entity_name} is owned by "
                f"{', '.join(catalog_metadata.owners)}. "
                f"Classification: {catalog_metadata.classification}. "
                "The published metadata is catalog context, not metric logic."
            ),
            result_classification=ResultClassification.CATALOG_OWNERSHIP,
            catalog_metadata=catalog_metadata,
            catalog_freshness=CatalogFreshness(available=True, degraded=False),
            source_freshness=source_freshness,
            effective_access_scope=scope,
            caveats=[
                "DataHub provides ownership, classification, and discovery metadata only.",
                "The validated dbt/MetricFlow artifact remains the semantic authority.",
            ],
            trace_id=trace_id,
        )

    @staticmethod
    def _catalog_limitation(
        *,
        entity_name: str,
        scope,
        source_freshness,
        trace_id: str,
        detail: str,
        available: bool = False,
    ):
        return GovernedAnalyticalResponse(
            answer=(
                f"Catalog ownership for {entity_name} is degraded because {detail} "
                "Canonical metric computation remains available from the validated "
                "dbt/MetricFlow artifact."
            ),
            result_classification=ResultClassification.LIMITATION,
            catalog_freshness=CatalogFreshness(
                available=available,
                degraded=True,
                detail=detail,
            ),
            source_freshness=source_freshness,
            effective_access_scope=scope,
            caveats=[
                "Catalog-dependent ownership, classification, and discovery details are not "
                "available in this response.",
                "Canonical metric logic is independent of DataHub availability.",
            ],
            trace_id=trace_id,
        )

    @staticmethod
    def _public_evidence_chain(chain) -> EvidenceChain:
        def reference(source, *, require_rank: bool = True) -> EvidenceChainReference:
            if require_rank and source.rank is None:
                raise LightRAGAuthorizationError(
                    "LightRAG evidence-chain reference is missing its retrieval rank."
                )
            return EvidenceChainReference(
                reference_id=_IDENTIFIER_PATTERN.sub("[redacted identifier]", source.reference_id),
                reference_kind=source.reference_kind,
                rank=source.rank,
                source_document_id=_IDENTIFIER_PATTERN.sub(
                    "[redacted identifier]", source.source_document_id
                ),
                source_url=_IDENTIFIER_PATTERN.sub("[redacted identifier]", source.source_url),
                source_revision=source.source_revision,
                chunk_id=_IDENTIFIER_PATTERN.sub("[redacted identifier]", source.chunk_id),
                product=source.product,
                region=source.region,
                tenant_scope=_IDENTIFIER_PATTERN.sub("[redacted identifier]", source.tenant_scope),
            )

        return EvidenceChain(
            supporting_chunks=[
                EvidenceChainChunk(
                    reference=reference(record.reference),
                    text=_IDENTIFIER_PATTERN.sub("[redacted identifier]", record.text),
                )
                for record in chain.supporting_chunks
            ],
            entities=[
                EvidenceChainEntity(
                    reference=reference(record.reference),
                    name=_IDENTIFIER_PATTERN.sub("[redacted identifier]", record.name),
                    description=_IDENTIFIER_PATTERN.sub(
                        "[redacted identifier]", record.description
                    ),
                )
                for record in chain.entities
            ],
            relations=[
                EvidenceChainRelation(
                    reference=reference(record.reference),
                    source_entity=reference(record.source_entity, require_rank=False),
                    target_entity=reference(record.target_entity, require_rank=False),
                    source_entity_reference_id=_IDENTIFIER_PATTERN.sub(
                        "[redacted identifier]", record.source_entity.reference_id
                    ),
                    target_entity_reference_id=_IDENTIFIER_PATTERN.sub(
                        "[redacted identifier]", record.target_entity.reference_id
                    ),
                    description=_IDENTIFIER_PATTERN.sub(
                        "[redacted identifier]", record.description
                    ),
                )
                for record in chain.relations
            ],
            references=[reference(source) for source in chain.references],
        )

    @staticmethod
    def _empty_public_evidence_chain() -> EvidenceChain:
        return EvidenceChain(
            supporting_chunks=[], entities=[], relations=[], references=[]
        )

    def _answer_direct_identifier_request(
        self,
        *,
        request: AnswerQuestionRequest,
        scope,
        access_profile,
        trace_id: str,
    ) -> GovernedAnalyticalResponse:
        if not access_profile.permitted_identifiers:
            freshness = SourceFreshness(
                validated_at=datetime.now(UTC),
                maximum_age_seconds=86_400,
                is_current=False,
            )
            return GovernedAnalyticalResponse(
                answer=(
                    "Safe refusal: this Access Profile has no explicit entitlement to direct "
                    "identifiers. The request was not sent to structured data, documents, or "
                    "the evidence graph."
                ),
                result_classification=ResultClassification.SAFE_REFUSAL,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=[
                    (
                        "Direct identifiers require explicit entitlement and a bounded audited "
                        "response."
                    )
                ],
                trace_id=trace_id,
            )

        artifact = self.semantic_gateway.artifact_store.load()
        freshness = self.semantic_gateway.freshness(artifact)
        product = "Confluence" if "confluence" in request.question.casefold() else "Jira"
        region = self._requested_region(request.question, access_profile)
        access_filter = access_profile.evidence_filter(
            product,
            region,
            metric_name="confluence_new_peu" if product == "Confluence" else "jira_new_peu",
            agent_user_id=request.agent_user_id,
        )
        require_bound_qdrant_age_stores(
            self.lightrag_adapter,
            self.evidence_store,
            self.graph_store,
        )
        documents = [
            document
            for document in self._retrieve_evidence(
                f"{product} {region} direct identifiers",
                access_filter,
                limit=_DIRECT_IDENTIFIER_RESULT_LIMIT,
            )
            if access_filter.allows(document)
        ]
        graph_paths = []
        if documents:
            graph_filter = access_profile.graph_filter(product, region)
            graph_filter = replace(
                graph_filter,
                groups=access_filter.groups,
                agent_user_id=access_filter.agent_user_id,
                as_of=access_filter.as_of,
                authorized_document_ids=tuple(
                    sorted(
                        document.source_document_id or document.document_id
                        for document in documents
                    )
                ),
                authorized_revision_keys=tuple(
                    sorted(
                        (
                            document.source_document_id or document.document_id,
                            document.source_revision,
                            document.chunk_id or f"{document.document_id}:chunk:0",
                        )
                        for document in documents
                    )
                ),
            )
            graph_paths = [
                path
                for path in self._traverse_graph(
                    f"{product} {region} direct identifiers",
                    graph_filter,
                    limit=_DIRECT_IDENTIFIER_RESULT_LIMIT,
                    metric_name=(
                        "confluence_new_peu" if product == "Confluence" else "jira_new_peu"
                    ),
                )
                if graph_filter.allows(path)
            ][:_DIRECT_IDENTIFIER_RESULT_LIMIT]
        identifiers = self._permitted_identifiers(
            graph_paths=graph_paths,
            documents=documents,
            access_profile=access_profile,
        )[:_DIRECT_IDENTIFIER_RESULT_LIMIT]
        audit = self._audit_direct_identifiers(
            trace_id=trace_id,
            agent_user_id=request.agent_user_id,
            scope=scope,
            access_profile=access_profile,
            outcome="released" if identifiers else "no_identifiers_found",
            returned_count=len(identifiers),
            maximum_results=_DIRECT_IDENTIFIER_RESULT_LIMIT,
        )
        identifier_text = ", ".join(identifier.value for identifier in identifiers)
        answer = (
            f"Bounded, audited direct-identifier response: {identifier_text}."
            if identifier_text
            else "No permitted direct identifiers were found in the requested scope."
        )
        return GovernedAnalyticalResponse(
            answer=answer,
            result_classification=ResultClassification.DIRECT_IDENTIFIER_RESPONSE,
            source_freshness=freshness,
            effective_access_scope=scope,
            graph_paths=self._graph_path_citations(graph_paths),
            direct_identifier_answer=DirectIdentifierAnswer(
                identifiers=identifiers,
                maximum_results=_DIRECT_IDENTIFIER_RESULT_LIMIT,
                audit_event_id=audit.audit_event_id,
            ),
            direct_identifier_audit=audit,
            caveats=[
                "Only explicitly entitled Tenant identifiers are returned.",
                "The response is bounded to three identifiers and its release is audited.",
            ],
            trace_id=trace_id,
        )

    def _traverse_graph(self, query, access_filter, *, limit: int, metric_name: str):
        with trace_span(
            "graph_traversal",
            kind="tool",
            attributes={"result_limit": limit},
        ):
            if isinstance(self.graph_store, ApacheAgeEvidenceGraphStore):
                return self.graph_store.traverse(
                    query,
                    access_filter,
                    limit=limit,
                    metric_name=metric_name,
                )
            if isinstance(self.graph_store, InMemoryEvidenceGraphStore):
                return self.graph_store.traverse(
                    query,
                    access_filter,
                    limit=limit,
                    metric_name=metric_name,
                )
            paths = self.graph_store.traverse(
                query,
                access_filter,
                limit=limit,
            )
            return [path for path in paths if _path_matches_metric(path, metric_name)]

    def _traverse_graph_for_evidence_tool(
        self, query: str, access_filter, metric_name: str, limit: int
    ):
        return self._traverse_graph(
            query,
            access_filter,
            limit=limit,
            metric_name=metric_name,
        )

    @staticmethod
    def _permitted_identifiers(*, graph_paths, documents, access_profile):
        candidates: list[SensitiveIdentifier] = []
        for path in graph_paths:
            for node in path.nodes:
                if node.identifier_entitlement != "direct":
                    continue
                candidates.extend(
                    AnswerQuestionService._identifier_values(
                        node.label, node.node_type, access_profile
                    )
                )
        for document in documents:
            if document.identifier_entitlement != "direct":
                continue
            for value in document.sensitive_identifiers:
                candidates.extend(
                    AnswerQuestionService._identifier_values(value, "tenant", access_profile)
                )
            for value in _IDENTIFIER_PATTERN.findall(document.text):
                candidates.extend(
                    AnswerQuestionService._identifier_values(value, "tenant", access_profile)
                )
        unique: dict[tuple[str, str], SensitiveIdentifier] = {}
        for candidate in candidates:
            unique[(candidate.identifier_type, candidate.value)] = candidate
        return list(unique.values())

    @staticmethod
    def _identifier_values(value: str, node_type: str, access_profile):
        identifier_type = {
            "tenant": "tenant_id",
            "person": "person_id",
            "product_user": "product_user_id",
        }.get(node_type)
        if identifier_type not in access_profile.permitted_identifiers:
            return []
        if identifier_type == "tenant_id" and value not in access_profile.permitted_tenant_ids:
            return []
        if not _IDENTIFIER_PATTERN.fullmatch(value):
            return []
        return [SensitiveIdentifier(identifier_type=identifier_type, value=value)]

    @staticmethod
    def _graph_path_citations(paths, *, redact_identifiers: bool = True):
        def safe(value: str) -> str:
            return _IDENTIFIER_PATTERN.sub("[redacted identifier]", value)

        return [
            {
                "path_id": safe(path.path_id) if redact_identifiers else path.path_id,
                "node_labels": [
                    (
                        f"{safe(node.label)} [{node.region}]"
                        if redact_identifiers
                        else f"{node.label} [{node.region}]"
                    )
                    for node in path.nodes
                ],
            }
            for path in paths
        ]

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

    @staticmethod
    def _requests_apac_decline_evidence(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return (
            "evidence" in normalized
            and "apac" in normalized
            and "decline" in normalized
            and ("51–200" in question or "51-200" in normalized)
        )

    @staticmethod
    def _requests_confluence_campaign_evidence(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return (
            "evidence" in normalized
            and "confluence" in normalized
            and "americas" in normalized
            and ("11–50" in question or "11-50" in normalized)
            and any(term in normalized for term in ("campaign", "movement", "lift", "increase"))
        )

    @staticmethod
    def _requests_confluence_emea_regression(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return (
            "evidence" in normalized
            and "confluence" in normalized
            and "emea" in normalized
            and "new mau" in normalized
            and "decline" in normalized
            and "onboarding" in normalized
            and "regression" in normalized
            and ("51–200" in question or "51-200" in normalized)
        )

    @staticmethod
    def _requests_causal_analysis(request: AnswerQuestionRequest) -> bool:
        normalized = " ".join(request.question.casefold().split())
        return (
            "jira" in normalized
            and "new mau" in normalized
            and any(
                term in normalized
                for term in (
                    "causal",
                    "treatment",
                    "control",
                    "experiment",
                    "estimate",
                    "effect",
                    "impact",
                    "pre/post",
                    "pre post",
                    "before and after",
                    "all user",
                    "effect estimate",
                )
            )
        )

    @staticmethod
    def _requests_direct_identifier(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return bool(
            _IDENTIFIER_PATTERN.search(normalized)
            or any(
                phrase in normalized
                for phrase in (
                    "direct identifier",
                    "tenant id",
                    "tenant identifier",
                    "person id",
                    "product user id",
                    "direct contact",
                    "who should we contact",
                    "which tenants",
                    "list of tenants",
                    "affected tenants",
                    "contact details",
                    "email address",
                    "phone number",
                )
            )
        )

    @staticmethod
    def _requested_region(question: str, access_profile) -> str:
        normalized = question.casefold()
        for region in access_profile.regions:
            if region.casefold() in normalized:
                return region
        if len(access_profile.regions) == 1:
            return access_profile.regions[0]
        return "APAC"

    @staticmethod
    def _metric_product(metric_name: str) -> str | None:
        if metric_name.startswith("jira_"):
            return "Jira"
        if metric_name.startswith("confluence_"):
            return "Confluence"
        return None

    @staticmethod
    def _metric_label(metric_name: str) -> str:
        product = AnswerQuestionService._metric_product(metric_name)
        if product is None:
            return metric_name
        metric_type = "New MAU" if metric_name.endswith("_new_mau") else "New PEU"
        return f"{product} {metric_type}"

    @staticmethod
    def _catalog_product(entity_name: str) -> str | None:
        normalized = entity_name.casefold()
        if "jira" in normalized:
            return "Jira"
        if "confluence" in normalized:
            return "Confluence"
        return None

    @staticmethod
    def _requests_catalog_ownership(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return any(term in normalized for term in ("who owns", "owner of", "ownership"))

    @staticmethod
    def _requested_catalog_entity(request: AnswerQuestionRequest) -> str | None:
        if not AnswerQuestionService._requests_catalog_ownership(request.question):
            return None
        normalized = " ".join(request.question.casefold().split())
        requested_metric = request.requested_metric_name
        if requested_metric is not None:
            return _metric_identifier(requested_metric)
        for entity_name in (
            "fct_jira_new_peu",
            "fct_confluence_new_peu",
            "fct_jira_new_mau",
            "fct_confluence_new_mau",
            "jira_new_peu",
            "confluence_new_peu",
            "jira_new_mau",
            "confluence_new_mau",
        ):
            if entity_name.replace("_", " ") in normalized or entity_name in normalized:
                return entity_name
        return None

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
                trace_id=trace_id,
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
            normalized_requested_metric = _metric_identifier(request.requested_metric_name)
            return normalized_requested_metric or None

        normalized = " ".join(request.question.casefold().split())
        if _requests_outside_analytical_scope(normalized):
            return None
        if (
            request.conversation_context is not None
            and request.conversation_context.summary.metric_name is not None
            and _requests_conversational_metric_follow_up(normalized)
        ):
            return request.conversation_context.summary.metric_name
        if "jira" in normalized and ("new peu" in normalized or "new paid enabled" in normalized):
            return "jira_new_peu"
        if "jira" in normalized and "new mau" in normalized:
            return "jira_new_mau"
        if "confluence" in normalized and "new peu" in normalized:
            return "confluence_new_peu"
        if "confluence" in normalized and "new mau" in normalized:
            return "confluence_new_mau"
        if AnswerQuestionService._requests_apac_decline_evidence(request.question):
            return "jira_new_peu"
        if any(term in normalized for term in ("metric", "rate", "count", "revenue")):
            return _metric_identifier(request.question)
        named_metric = _named_metric_question(request.question)
        if named_metric is not None:
            return _metric_identifier(named_metric)
        return None

    @staticmethod
    def _active_selected_factor_id(
        request: AnswerQuestionRequest, investigation_metric_name: str
    ) -> str | None:
        """Resolve which Candidate Causal Factor this turn should filter to, if any.

        Never trusted as content — only ever a lookup key the caller must revalidate
        against this turn's freshly recomputed ranked candidates.
        """
        if request.selected_factor_id is not None:
            return request.selected_factor_id
        prior = request.conversation_context.summary if request.conversation_context else None
        if (
            prior is not None
            and prior.active_investigation_factor_id is not None
            and prior.metric_name == investigation_metric_name
        ):
            return prior.active_investigation_factor_id
        return None

    @staticmethod
    def _is_canonical_definition_request(
        request: AnswerQuestionRequest, metric_name: str | None
    ) -> bool:
        """Keep specialist routes out of the canonical-definition node until migrated."""
        return metric_name is not None and not AnswerQuestionService._requires_specialist_dispatch(
            request, metric_name
        )

    @staticmethod
    def _route_for_intent(
        request: AnswerQuestionRequest,
        metric_name: str | None,
        *,
        canonical_metric_names: Collection[str] | None = None,
    ) -> AnalyticalRoute:
        if AnswerQuestionService._requests_direct_identifier(request.question):
            return AnalyticalRoute.DIRECT_IDENTIFIER
        if AnswerQuestionService._requested_catalog_entity(request) is not None:
            return AnalyticalRoute.CATALOG_OWNERSHIP
        if AnswerQuestionService._requests_causal_analysis(request):
            return AnalyticalRoute.CAUSAL_ANALYSIS
        if metric_name is None:
            return AnalyticalRoute.LIMITATION
        known_metric_names = (
            set(canonical_metric_names)
            if canonical_metric_names is not None
            else {
                "jira_new_peu",
                "confluence_new_peu",
                "jira_new_mau",
                "confluence_new_mau",
            }
        )
        if metric_name not in known_metric_names:
            return AnalyticalRoute.METRIC_DEFINITION_GAP
        if metric_name in {
            "jira_new_peu",
            "confluence_new_peu",
            "jira_new_mau",
            "confluence_new_mau",
        } and AnswerQuestionService._requests_may_to_june_driver_decomposition(request.question):
            return AnalyticalRoute.DRIVER_DECOMPOSITION
        if AnswerQuestionService._is_canonical_definition_request(request, metric_name):
            return AnalyticalRoute.CANONICAL_DEFINITION
        return AnalyticalRoute.LEGACY

    @staticmethod
    def _requires_specialist_dispatch(request: AnswerQuestionRequest, metric_name: str) -> bool:
        """Route only requests the current specialist dispatcher can complete."""
        return (
            AnswerQuestionService._requests_direct_identifier(request.question)
            or AnswerQuestionService._requested_catalog_entity(request) is not None
            or AnswerQuestionService._requests_causal_analysis(request)
            or (
                metric_name == "jira_new_peu"
                and AnswerQuestionService._requests_apac_decline_evidence(request.question)
            )
            or (
                metric_name == "confluence_new_peu"
                and AnswerQuestionService._requests_confluence_campaign_evidence(request.question)
            )
            or (
                metric_name == "confluence_new_mau"
                and AnswerQuestionService._requests_confluence_emea_regression(request.question)
            )
            or (
                metric_name
                in {
                    "jira_new_peu",
                    "confluence_new_peu",
                    "jira_new_mau",
                    "confluence_new_mau",
                }
                and AnswerQuestionService._requests_may_to_june_driver_decomposition(
                    request.question
                )
            )
        )


def _metric_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _requests_conversational_metric_follow_up(normalized_question: str) -> bool:
    return (
        any(phrase in normalized_question for phrase in ("that metric", "this metric"))
        and any(term in normalized_question for term in ("mean", "definition", "defined"))
    ) or "what does it mean" in normalized_question


def _path_matches_metric(path, metric_name: str) -> bool:
    """Keep legacy graph stores compatible while filtering metric-bearing paths."""
    metric_nodes = [node for node in path.nodes if node.node_type == "metric"]
    return not metric_nodes or any(node.node_id == metric_name for node in metric_nodes)


def _named_metric_question(question: str) -> str | None:
    match = re.fullmatch(r"\s*(?:what is|define)\s+(.+?)\s*\??\s*", question, re.IGNORECASE)
    if match is None:
        return None
    name = match.group(1).strip()
    if name.casefold() in {"this", "that", "it"}:
        return None
    return name


def _requests_outside_analytical_scope(normalized_question: str) -> bool:
    return any(
        term in normalized_question
        for term in (
            "weather",
            "recipe",
            "joke",
            "sports score",
            "stock price",
            "news",
            "translate",
            "hello",
        )
    )

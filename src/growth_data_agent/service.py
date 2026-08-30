"""The narrow answer_question application seam."""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import UTC, datetime
from uuid import uuid4

from .audit import DirectIdentifierAuditRecorder, InMemoryDirectIdentifierAuditRecorder
from .causal import CausalAnalysisPipeline, default_causal_pipeline
from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    CatalogFreshness,
    CatalogMetadata,
    DirectIdentifierAnswer,
    EvidenceSupportStatus,
    GovernedAnalyticalResponse,
    MetricDefinitionGap,
    ResultClassification,
    SensitiveIdentifier,
    SourceFreshness,
)
from .datahub import (
    DataHubCatalogStore,
    DataHubCatalogUnavailableError,
)
from .evidence import QdrantEvidenceStore, VectorEvidenceStore, build_evidence_answer
from .evidence_tools import BoundedEvidenceInvestigationTools
from .execution import (
    AuthorizedExecution,
    ExecutionGraph,
    IntentInterpreter,
    RuleBasedIntentInterpreter,
)
from .graph import ApacheAgeEvidenceGraphStore, EvidenceGraphStore, InMemoryEvidenceGraphStore
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
    TraceRecord,
    TraceSink,
    capture_trace,
    policy_fingerprint,
    trace_span,
)
from .policy import (
    AccessDeniedError,
    UnknownAgentUserError,
    resolve_access_profile,
    tenant_ids_for_segment,
)
from .semantic import ValidatedMetricFlowGateway
from .synthetic import evidence_corpus, graph_corpus

_DIRECT_IDENTIFIER_RESULT_LIMIT = 3
_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)


class AnswerQuestionService:
    def __init__(
        self,
        semantic_gateway: ValidatedMetricFlowGateway,
        *,
        provisional_metric_calculator: ProvisionalMetricCalculator | None = None,
        provisional_metric_input_gateway: ProvisionalMetricInputGateway | None = None,
        verification_request_recorder: DataTeamVerificationRequestRecorder | None = None,
        evidence_store: VectorEvidenceStore | None = None,
        graph_store: EvidenceGraphStore | None = None,
        catalog_store: DataHubCatalogStore | None = None,
        direct_identifier_audit_recorder: DirectIdentifierAuditRecorder | None = None,
        causal_pipeline: CausalAnalysisPipeline | None = None,
        trace_sink: TraceSink | None = None,
        execution_graph: ExecutionGraph | None = None,
        local_model: LocalModelTransport | None = None,
        evidence_model: LocalModelTransport | None = None,
        intent_interpreter: IntentInterpreter | None = None,
        evidence_drafting_adapter: EvidenceDraftingAdapter | None = None,
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
        self.graph_store = graph_store or InMemoryEvidenceGraphStore(graph_corpus())
        self.evidence_tools = BoundedEvidenceInvestigationTools(
            self.evidence_store,
            self._traverse_graph_for_evidence_tool,
        )
        self.catalog_store = catalog_store
        self.direct_identifier_audit_recorder = (
            direct_identifier_audit_recorder or InMemoryDirectIdentifierAuditRecorder()
        )
        self.causal_pipeline = causal_pipeline or default_causal_pipeline()
        self.trace_sink = trace_sink or NoOpTraceSink()
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
            causal_analysis_handler=self._answer_causal_specialist,
            catalog_ownership_handler=self._answer_catalog_specialist,
            direct_identifier_handler=self._answer_direct_identifier_specialist,
            limitation_handler=self._answer_limitation_specialist,
            metric_definition_gap_handler=self._answer_metric_definition_gap_specialist,
            legacy_handler=self._answer_legacy_question,
            clarification_handler=self._answer_intent_clarification,
        )

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        with capture_trace() as trace_context:
            try:
                response = self.execution_graph.answer_question(request)
                response = self._draft_evidence_response(response)
            except (AccessDeniedError, UnknownAgentUserError) as error:
                error.trace_id = self._record_authorization_denial(request, trace_context)
                raise
            except Exception as error:
                error.trace_id = self._record_dependency_failure(request, trace_context)
                raise
            self._record_trace(request, response, trace_context)
            return response

    def readiness(self) -> dict[str, object]:
        """Expose model dependency state without exposing request or credential data."""
        local_model = local_model_readiness(self.local_model)
        return {
            "status": "unavailable" if local_model["status"] == "unavailable" else "ready",
            "local_model": local_model,
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

        if self._requests_causal_analysis(request):
            access_profile.authorize_product("Jira")
            return self._answer_causal_analysis(
                request=request,
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

        if metric_name == "jira_new_peu" and self._requests_apac_decline_evidence(
            request.question
        ):
            return self._answer_apac_decline_evidence(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
                agent_user_id=request.agent_user_id,
            )

        if metric_name == "confluence_new_peu" and self._requests_confluence_campaign_evidence(
            request.question
        ):
            return self._answer_confluence_campaign_evidence(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
                agent_user_id=request.agent_user_id,
            )

        if metric_name == "confluence_new_mau" and self._requests_confluence_emea_regression(
            request.question
        ):
            return self._answer_confluence_emea_regression_evidence(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
                agent_user_id=request.agent_user_id,
            )

        if metric_name in {
            "jira_new_peu",
            "confluence_new_peu",
            "jira_new_mau",
            "confluence_new_mau",
        } and (
            self._requests_may_to_june_driver_decomposition(request.question)
        ):
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

    def _answer_causal_specialist(
        self, authorized_execution: AuthorizedExecution
    ) -> GovernedAnalyticalResponse:
        authorized_execution.access_profile.authorize_product("Jira")
        return self._answer_causal_analysis(
            request=authorized_execution.request,
            scope=authorized_execution.effective_scope,
            access_profile=authorized_execution.access_profile,
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
            "causal_pipeline": (
                response.result_classification.value
                if response.causal_registration is not None
                or response.causal_analysis_plan is not None
                else "not_used"
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
            evaluation_outcome=(
                response.result_classification.value
                if response.causal_registration is not None
                or response.causal_analysis_plan is not None
                else "not_evaluated"
            ),
            response=response.model_dump(mode="json"),
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        try:
            self.trace_sink.record(trace)
        except Exception:
            # Observability must not turn a governed response into an outage.
            return

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
                "causal_pipeline": (
                    "authorization_denied"
                    if self._requests_causal_analysis(request)
                    else "not_used"
                ),
            },
            retrieval_scores=(),
            evaluation_outcome="not_evaluated",
            response={
                "result_classification": "safe_refusal",
                "error_code": "access_denied",
            },
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        try:
            self.trace_sink.record(trace)
        except Exception:
            pass
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
            node_spans=trace_context.node_spans,
            tool_spans=trace_context.tool_spans,
        )
        try:
            self.trace_sink.record(trace)
        except Exception:
            pass
        return trace_id

    def _policy_fingerprint_or_unknown(self, request: AnswerQuestionRequest) -> str:
        try:
            return policy_fingerprint(resolve_access_profile(request.agent_user_id))
        except UnknownAgentUserError:
            return "unknown-agent"

    def _semantic_canonical_definition(self, metric_name: str):
        with trace_span(
            "semantic_definition",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.canonical_definition(metric_name)

    def _semantic_execute_scoped_metric(
        self, metric_name: str, access_profile, **scope_kwargs
    ):
        with trace_span(
            "semantic_query",
            kind="tool",
            attributes={"metric_name": metric_name},
        ):
            return self.semantic_gateway.execute_scoped_metric(
                metric_name, access_profile, **scope_kwargs
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
            return self.evidence_store.retrieve(query, access_filter, limit=limit)

    def _catalog_get(self, entity_name: str):
        with trace_span(
            "catalog_lookup",
            kind="tool",
            attributes={"entity_name": entity_name},
        ):
            return self.catalog_store.get(entity_name)

    def _causal_evaluate(self, experiment_id: str):
        with trace_span(
            "causal_evaluation",
            kind="tool",
            attributes={"experiment_id": experiment_id},
        ):
            return self.causal_pipeline.evaluate(experiment_id)

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
        definition, decomposition, query_evidence, freshness = (
            self._semantic_driver_decomposition(
                metric_name,
                access_profile,
                baseline_period="2026-05",
                comparison_period="2026-06",
            )
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

    def _answer_causal_analysis(
        self,
        *,
        request: AnswerQuestionRequest,
        scope,
        access_profile,
        trace_id: str,
    ) -> GovernedAnalyticalResponse:
        question_experiment_id = self._requested_causal_experiment_id(request.question)
        if request.experiment_id is not None and self._has_explicit_causal_variant(
            request.question
        ):
            experiment_id = question_experiment_id
        else:
            experiment_id = request.experiment_id or question_experiment_id
        evaluation = self._causal_evaluate(experiment_id)
        if evaluation.registration is not None:
            for region in evaluation.registration.regions:
                if evaluation.registration.seat_tier is None:
                    access_profile.authorize_region(region)
                else:
                    access_profile.authorize_tenant_scope(
                        region, evaluation.registration.seat_tier
                    )
        definition = None
        query_evidence = None
        if evaluation.outcome == "causal_estimate":
            definition, freshness = self._semantic_canonical_definition("jira_new_mau")
            if definition is None:
                return GovernedAnalyticalResponse(
                    answer=(
                        "The Jira New MAU causal request is limited because the canonical "
                        "dbt/MetricFlow outcome is not currently validated."
                    ),
                    result_classification=ResultClassification.LIMITATION,
                    causal_registration=evaluation.registration,
                    causal_analysis_plan=evaluation.analysis_plan,
                    source_freshness=freshness,
                    effective_access_scope=scope,
                    caveats=[
                        "No Causal Estimate is produced without a current canonical outcome.",
                    ],
                    trace_id=trace_id,
                )

            query_evidence, freshness = self._semantic_execute_scoped_metric(
                "jira_new_mau",
                access_profile,
                scoped_regions=tuple(evaluation.registration.regions),
                scoped_seat_tier=evaluation.registration.seat_tier,
                scoped_tenant_ids=tuple(
                    tenant_id
                    for region in evaluation.registration.regions
                    for tenant_id in tenant_ids_for_segment(
                        region, evaluation.registration.seat_tier
                    )
                ),
                scoped_tenant_scope=evaluation.registration.tenant_scope,
            )
            if query_evidence is None:
                return GovernedAnalyticalResponse(
                    answer=(
                        "The Jira New MAU causal request is limited because its scoped canonical "
                        "outcome could not be queried."
                    ),
                    result_classification=ResultClassification.LIMITATION,
                    canonical_definition=definition,
                    causal_registration=evaluation.registration,
                    causal_analysis_plan=evaluation.analysis_plan,
                    source_freshness=freshness,
                    effective_access_scope=scope,
                    caveats=["No Causal Estimate is produced without a scoped outcome query."],
                    trace_id=trace_id,
                )
        else:
            try:
                artifact = self.semantic_gateway.artifact_store.load()
                freshness = self.semantic_gateway.freshness(artifact)
            except (OSError, ValueError):
                freshness = SourceFreshness(
                    validated_at=datetime.now(UTC),
                    maximum_age_seconds=86_400,
                    is_current=False,
                )

        if evaluation.outcome == "causal_estimate":
            assert evaluation.causal_estimate is not None
            estimate = evaluation.causal_estimate
            answer = (
                "Causal Estimate: the registered Jira New MAU treatment/control experiment "
                f"estimates a {estimate.estimate:+.2%} treatment effect using the pre-approved "
                f"{estimate.estimator} estimator. Eligibility, assumptions, diagnostics, and "
                "required review passed the governed gate."
            )
            classification = ResultClassification.CAUSAL_ESTIMATE
        elif evaluation.registration is not None:
            plan = evaluation.analysis_plan
            reason = plan.reason if plan is not None else "The causal gate did not pass."
            if evaluation.outcome == "descriptive_result":
                descriptive = evaluation.descriptive_comparison
                observed = ""
                if descriptive is not None:
                    observed = (
                        f" Observed {descriptive.treatment} at {descriptive.treatment_value:.2%} "
                        f"versus {descriptive.control} at {descriptive.control_value:.2%} "
                        f"({descriptive.difference:+.2%}); this comparison is descriptive."
                    )
                answer = (
                    "Descriptive result only: the Jira New MAU design was not eligible for a "
                    f"Causal Estimate. {reason}{observed}"
                )
                classification = ResultClassification.DESCRIPTIVE_RESULT
            else:
                answer = (
                    "Reviewable analysis plan: no Causal Estimate was produced for the Jira "
                    f"New MAU design. {reason}"
                )
                classification = ResultClassification.ANALYSIS_PLAN
        else:
            plan = evaluation.analysis_plan
            answer = (
                "Reviewable analysis plan: no Causal Estimate was produced because the Jira "
                f"New MAU design is unregistered. {plan.reason if plan is not None else ''}"
            )
            classification = ResultClassification.ANALYSIS_PLAN

        return GovernedAnalyticalResponse(
            answer=answer,
            result_classification=classification,
            canonical_definition=definition,
            semantic_query_evidence=query_evidence,
            causal_registration=evaluation.registration,
            causal_estimate=evaluation.causal_estimate,
            descriptive_comparison=evaluation.descriptive_comparison,
            causal_analysis_plan=evaluation.analysis_plan,
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                "Causal estimation is restricted to this deterministic governed pipeline.",
                "A non-estimate outcome must remain descriptive or a reviewable analysis plan.",
            ],
            trace_id=trace_id,
        )

    def _answer_apac_decline_evidence(
        self, *, scope, access_profile, trace_id: str, agent_user_id: str
    ):
        return self._answer_segment_evidence(
            metric_name="jira_new_peu",
            region="APAC",
            seat_tier="51-200",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            evidence_query="Jira APAC 51-200 paid provisioning June 2026 decline",
            supported_answer=(
                "Hypothesis: the permitted Jira APAC paid-provisioning incident may explain "
                "part of the observed APAC 51-200 Seat Tier Tenant decline. "
                "The evidence supports this Hypothesis but does not establish causation."
            ),
            inconclusive_answer=(
                "Inconclusive: the permitted evidence does not support a reliable explanation "
                "for the observed APAC 51-200 Seat Tier Tenant decline."
            ),
        )

    def _answer_confluence_campaign_evidence(
        self, *, scope, access_profile, trace_id: str, agent_user_id: str
    ):
        return self._answer_segment_evidence(
            metric_name="confluence_new_peu",
            region="Americas",
            seat_tier="11-50",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            evidence_query=(
                "Confluence Americas 11-50 acquisition campaign June 2026 New PEU movement"
            ),
            scope_evidence_to_seat_tier=True,
            supported_answer=(
                "Hypothesis: the permitted Confluence Americas acquisition campaign may help "
                "explain the observed Americas 11-50 Seat Tier Tenant movement. "
                "The evidence supports this Hypothesis but does not establish causation."
            ),
            inconclusive_answer=(
                "Inconclusive: the permitted evidence does not support a reliable explanation "
                "for the observed Americas 11-50 Seat Tier Tenant movement."
            ),
        )

    def _answer_confluence_emea_regression_evidence(
        self, *, scope, access_profile, trace_id: str, agent_user_id: str
    ):
        return self._answer_segment_evidence(
            metric_name="confluence_new_mau",
            region="EMEA",
            seat_tier="51-200",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            evidence_query=(
                "Confluence EMEA 51-200 onboarding-email regression June 2026 New MAU decline"
            ),
            scope_evidence_to_seat_tier=True,
            supported_answer=(
                "Hypothesis: the permitted Confluence EMEA onboarding-email regression may help "
                "explain the observed 51-200 Seat Tier Tenant New MAU decline. The evidence "
                "supports this Hypothesis but does not establish causation."
            ),
            inconclusive_answer=(
                "Inconclusive: the permitted evidence does not support a reliable explanation "
                "for the observed Confluence EMEA 51-200 Seat Tier Tenant New MAU decline."
            ),
        )

    def _answer_segment_evidence(
        self,
        *,
        metric_name: str,
        region: str,
        seat_tier: str,
        scope,
        access_profile,
        trace_id: str,
        agent_user_id: str,
        evidence_query: str,
        supported_answer: str,
        inconclusive_answer: str,
        scope_evidence_to_seat_tier: bool = False,
    ):
        metric_product = self._metric_product(metric_name)
        access_profile.authorize_product(metric_product)
        access_profile.authorize_region(region)
        definition, decomposition, query_evidence, freshness = (
            self._semantic_driver_decomposition(
                metric_name,
                access_profile,
                baseline_period="2026-05",
                comparison_period="2026-06",
            )
        )
        if definition is None or decomposition is None or query_evidence is None:
            return GovernedAnalyticalResponse(
                answer=(
                    f"Evidence for the {metric_product} {region} {seat_tier} movement cannot "
                    "be assessed because semantic validation is not current."
                ),
                result_classification=ResultClassification.LIMITATION,
                source_freshness=freshness,
                effective_access_scope=scope,
                caveats=[
                    "Run dbt validation and refresh the semantic artifact before retrieving "
                    "evidence."
                ],
                trace_id=trace_id,
            )

        segment_filter = seat_tier if scope_evidence_to_seat_tier else None
        graph_filter = access_profile.graph_filter(
            metric_product,
            region,
            seat_tier=segment_filter,
        )
        access_filter = access_profile.evidence_filter(
            metric_product,
            region,
            seat_tier=seat_tier if scope_evidence_to_seat_tier else None,
            metric_name=metric_name,
            agent_user_id=agent_user_id,
        )
        investigation = self.evidence_tools.investigate(
            query=evidence_query,
            evidence_filter=access_filter,
            graph_filter=graph_filter,
            metric_name=metric_name,
        )
        evidence = build_evidence_answer(investigation.documents)
        if evidence.support_status == EvidenceSupportStatus.SUPPORTS:
            classification = ResultClassification.HYPOTHESIS
            answer = supported_answer
        else:
            classification = ResultClassification.INCONCLUSIVE
            answer = f"{inconclusive_answer} {evidence.support_explanation}"
        return GovernedAnalyticalResponse(
            answer=answer,
            result_classification=classification,
            canonical_definition=definition,
            semantic_query_evidence=query_evidence,
            driver_decomposition=decomposition,
            evidence=evidence,
            graph_paths=self._graph_path_citations(investigation.graph_paths),
            source_freshness=freshness,
            effective_access_scope=scope,
            caveats=[
                (
                    f"The {region} {seat_tier} Seat Tier result is an observed Driver "
                    "Decomposition; the retrieved material is a Hypothesis, not a causal "
                    "conclusion."
                ),
                "Only evidence permitted by product, Region, Tenant, classification, and "
                "identifier entitlements was retrieved.",
            ],
            trace_id=trace_id,
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
        graph_filter = access_profile.graph_filter(product, region)
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
        access_filter = access_profile.evidence_filter(
            product,
            region,
            metric_name="confluence_new_peu" if product == "Confluence" else "jira_new_peu",
            agent_user_id=request.agent_user_id,
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
            "Bounded, audited direct-identifier response: "
            f"{identifier_text}."
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
                    AnswerQuestionService._identifier_values(
                        value, "tenant", access_profile
                    )
                )
            for value in _IDENTIFIER_PATTERN.findall(document.text):
                candidates.extend(
                    AnswerQuestionService._identifier_values(
                        value, "tenant", access_profile
                    )
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
            and (
                "51–200" in question
                or "51-200" in normalized
            )
        )

    @staticmethod
    def _requests_causal_analysis(request: AnswerQuestionRequest) -> bool:
        if request.experiment_id is not None:
            normalized = " ".join(request.question.casefold().split())
            return "confluence" not in normalized or "jira" in normalized
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
    def _requested_causal_experiment_id(question: str) -> str:
        normalized = " ".join(question.casefold().split())
        if "unregistered" in normalized:
            return "unregistered-jira-new-mau-design"
        support_failed = "support" in normalized and any(
            term in normalized for term in ("fail", "did not pass", "not pass")
        )
        if "failed support" in normalized or support_failed:
            return "jira-new-mau-onboarding-experiment-failed-support"
        if any(
            term in normalized
            for term in ("missing review", "review missing", "pending review", "not reviewed")
        ):
            return "jira-new-mau-onboarding-experiment-pending-review"
        all_user_pre_post = (
            "pre/post" in normalized
            or "pre post" in normalized
            or "all-user" in normalized
            or ("all user" in normalized and "before" in normalized and "after" in normalized)
        )
        if all_user_pre_post:
            return "jira-new-mau-all-user-pre-post"
        if "observational" in normalized or "quasi-experimental" in normalized:
            return "jira-new-mau-observational-design"
        return "jira-new-mau-onboarding-experiment"

    @staticmethod
    def _has_explicit_causal_variant(question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return any(
            term in normalized
            for term in (
                "unregistered",
                "failed support",
                "support check",
                "missing review",
                "review missing",
                "pending review",
                "not reviewed",
                "observational",
                "quasi-experimental",
                "pre/post",
                "pre post",
                "before and after",
                "all user",
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
        if (
            metric_name
            in {"jira_new_peu", "confluence_new_peu", "jira_new_mau", "confluence_new_mau"}
            and AnswerQuestionService._requests_may_to_june_driver_decomposition(request.question)
        ):
            return AnalyticalRoute.DRIVER_DECOMPOSITION
        if AnswerQuestionService._is_canonical_definition_request(request, metric_name):
            return AnalyticalRoute.CANONICAL_DEFINITION
        return AnalyticalRoute.LEGACY

    @staticmethod
    def _requires_specialist_dispatch(
        request: AnswerQuestionRequest, metric_name: str
    ) -> bool:
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

"""The narrow answer_question application seam."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

from .audit import DirectIdentifierAuditRecorder, InMemoryDirectIdentifierAuditRecorder
from .contracts import (
    AnswerQuestionRequest,
    DirectIdentifierAnswer,
    EvidenceSupportStatus,
    GovernedAnalyticalResponse,
    MetricDefinitionGap,
    ResultClassification,
    SensitiveIdentifier,
    SourceFreshness,
)
from .evidence import QdrantEvidenceStore, VectorEvidenceStore, build_evidence_answer
from .graph import EvidenceGraphStore, InMemoryEvidenceGraphStore
from .metric_definition_gaps import (
    DataTeamVerificationRequestRecorder,
    InMemoryDataTeamVerificationRequestRecorder,
    NoProvisionalMetricCalculator,
    NoProvisionalMetricInputGateway,
    ProvisionalMetricCalculator,
    ProvisionalMetricInputGateway,
    ProvisionalMetricInputRequest,
)
from .observability import NoOpTraceSink, TraceRecord, TraceSink, policy_fingerprint
from .policy import AccessDeniedError, UnknownAgentUserError, resolve_access_profile
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
        direct_identifier_audit_recorder: DirectIdentifierAuditRecorder | None = None,
        trace_sink: TraceSink | None = None,
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
        self.direct_identifier_audit_recorder = (
            direct_identifier_audit_recorder or InMemoryDirectIdentifierAuditRecorder()
        )
        self.trace_sink = trace_sink or NoOpTraceSink()

    def answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        try:
            response = self._answer_question(request)
        except (AccessDeniedError, UnknownAgentUserError) as error:
            error.trace_id = self._record_authorization_denial(request)
            raise
        self._record_trace(request, response)
        return response

    def _answer_question(self, request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        access_profile = resolve_access_profile(request.agent_user_id)
        scope = access_profile.as_effective_scope()
        trace_id = str(uuid4())

        if self._requests_direct_identifier(request.question):
            return self._answer_direct_identifier_request(
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
            )

        if metric_name == "confluence_new_peu" and self._requests_confluence_campaign_evidence(
            request.question
        ):
            return self._answer_confluence_campaign_evidence(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
            )

        if metric_name == "confluence_new_mau" and self._requests_confluence_emea_regression(
            request.question
        ):
            return self._answer_confluence_emea_regression_evidence(
                scope=scope,
                access_profile=access_profile,
                trace_id=trace_id,
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

        metric_product = self._metric_product(metric_name)
        if metric_product is not None:
            access_profile.authorize_product(metric_product)
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

    def _record_trace(
        self,
        request: AnswerQuestionRequest,
        response: GovernedAnalyticalResponse,
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
        )
        try:
            self.trace_sink.record(trace)
        except Exception:
            # Observability must not turn a governed response into an outage.
            return

    def _record_authorization_denial(self, request: AnswerQuestionRequest) -> str:
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
        )
        try:
            self.trace_sink.record(trace)
        except Exception:
            pass
        return trace.trace_id

    def _answer_metric_driver_decomposition(
        self, *, metric_name: str, scope, access_profile, trace_id: str
    ):
        definition, decomposition, query_evidence, freshness = (
            self.semantic_gateway.driver_decomposition(
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

    def _answer_apac_decline_evidence(self, *, scope, access_profile, trace_id: str):
        return self._answer_segment_evidence(
            metric_name="jira_new_peu",
            region="APAC",
            seat_tier="51-200",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
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

    def _answer_confluence_campaign_evidence(self, *, scope, access_profile, trace_id: str):
        return self._answer_segment_evidence(
            metric_name="confluence_new_peu",
            region="Americas",
            seat_tier="11-50",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
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
        self, *, scope, access_profile, trace_id: str
    ):
        return self._answer_segment_evidence(
            metric_name="confluence_new_mau",
            region="EMEA",
            seat_tier="51-200",
            scope=scope,
            access_profile=access_profile,
            trace_id=trace_id,
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
        evidence_query: str,
        supported_answer: str,
        inconclusive_answer: str,
        scope_evidence_to_seat_tier: bool = False,
    ):
        metric_product = self._metric_product(metric_name)
        access_profile.authorize_product(metric_product)
        access_profile.authorize_region(region)
        definition, decomposition, query_evidence, freshness = (
            self.semantic_gateway.driver_decomposition(
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
        graph_paths = [
            path
            for path in self.graph_store.traverse(
                evidence_query,
                graph_filter,
                limit=3,
            )
            if graph_filter.allows(path)
        ][:3]
        access_filter = access_profile.evidence_filter(
            metric_product,
            region,
            seat_tier=seat_tier if scope_evidence_to_seat_tier else None,
        )
        documents = [
            document
            for document in self.evidence_store.retrieve(
                evidence_query,
                access_filter,
                limit=3,
            )
            if access_filter.allows(document)
        ]
        evidence = build_evidence_answer(documents)
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
            graph_paths=self._graph_path_citations(graph_paths),
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
            for path in self.graph_store.traverse(
                f"{product} {region} direct identifiers",
                graph_filter,
                limit=_DIRECT_IDENTIFIER_RESULT_LIMIT,
            )
            if graph_filter.allows(path)
        ][:_DIRECT_IDENTIFIER_RESULT_LIMIT]
        access_filter = access_profile.evidence_filter(product, region)
        documents = [
            document
            for document in self.evidence_store.retrieve(
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
        audit = self.direct_identifier_audit_recorder.record(
            trace_id=trace_id,
            agent_user_id=request.agent_user_id,
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

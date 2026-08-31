"""Typed contracts exposed by the governed analytical response seam."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .principal import VerifiedPrincipal


class ResultClassification(StrEnum):
    CANONICAL_DEFINITION = "canonical_definition"
    CATALOG_OWNERSHIP = "catalog_ownership"
    DRIVER_DECOMPOSITION = "driver_decomposition"
    HYPOTHESIS = "hypothesis"
    INCONCLUSIVE = "inconclusive"
    DIRECT_IDENTIFIER_RESPONSE = "direct_identifier_response"
    SAFE_REFUSAL = "safe_refusal"
    METRIC_DEFINITION_GAP = "metric_definition_gap"
    PROVISIONAL_METRIC = "provisional_metric"
    LIMITATION = "limitation"


class AnalyticalRoute(StrEnum):
    """The bounded execution route selected from a validated Analytical Intent."""

    CANONICAL_DEFINITION = "canonical_definition"
    CLARIFICATION = "clarification"
    DRIVER_DECOMPOSITION = "driver_decomposition"
    CAUSAL_ANALYSIS = "causal_analysis"
    CATALOG_OWNERSHIP = "catalog_ownership"
    DIRECT_IDENTIFIER = "direct_identifier"
    LIMITATION = "limitation"
    METRIC_DEFINITION_GAP = "metric_definition_gap"
    LEGACY = "legacy"


class InvestigationComplexity(StrEnum):
    MEDIUM = "medium"
    HARD = "hard"


class PlanAction(StrEnum):
    METRICFLOW = "metricflow"
    CITED_EVIDENCE = "cited_evidence"
    LIGHTRAG = "lightrag"


class ToolOutcomeStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class PlanToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: PlanAction
    status: ToolOutcomeStatus
    error_type: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )


class LeadAgentMetadata(BaseModel):
    """Safe, bounded state that can be persisted and traced for one turn."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, max_length=128)
    complexity: InvestigationComplexity
    actions: list[PlanAction] = Field(min_length=1, max_length=3)
    current_action: PlanAction | None = None
    plan_revision: int = Field(ge=0)
    replan_count: int = Field(ge=0, le=2)
    policy_fingerprint: str = Field(min_length=1, max_length=128)
    semantic_current: bool
    evidence_revision_fingerprints: list[str] = Field(max_length=32)
    tool_outcomes: list[PlanToolOutcome] = Field(max_length=8)
    last_replan_reason: str | None = Field(default=None, max_length=64)


class AnalyticalIntent(BaseModel):
    """A validated interpretation of the request before specialist execution."""

    route: AnalyticalRoute
    metric_name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_metric_for_canonical_definition(self) -> AnalyticalIntent:
        if (
            self.route
            in {
                AnalyticalRoute.CANONICAL_DEFINITION,
                AnalyticalRoute.DRIVER_DECOMPOSITION,
            }
            and self.metric_name is None
        ):
            raise ValueError(f"{self.route.value} intent requires a metric name.")
        return self


class EffectiveAccessScope(BaseModel):
    products: list[str]
    regions: list[str]
    tenant_scope: str
    permitted_columns: list[str]


class ConversationSummary(BaseModel):
    """Structured working context; it is never a source or authorization authority."""

    model_config = ConfigDict(extra="forbid")

    agent_user_goal: str = Field(default="", max_length=256)
    resolved_scope: EffectiveAccessScope | None = None
    metric_name: str | None = Field(default=None, min_length=1, max_length=128)
    evidence_revision_ids: list[str] = Field(default_factory=list, max_length=32)
    qualified_conclusions: list[str] = Field(default_factory=list, max_length=32)
    open_questions: list[str] = Field(default_factory=list, max_length=16)
    workflow_state: str = Field(default="active", min_length=1, max_length=64)


class ConversationTurn(BaseModel):
    """Bounded turn metadata retained for conversational context, without evidence chunks."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2_000)
    result_classification: ResultClassification
    metric_name: str | None = Field(default=None, min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    created_at: datetime
    lead_agent_metadata: LeadAgentMetadata | None = None


class ConversationContext(BaseModel):
    """The only prior context injected into a turn's governed execution."""

    model_config = ConfigDict(extra="forbid")

    summary: ConversationSummary
    recent_turns: list[ConversationTurn] = Field(default_factory=list, max_length=32)


class AnswerQuestionPayload(BaseModel):
    """Public answer payload; authentication is supplied by the HTTP header."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    requested_metric_name: str | None = Field(default=None, min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    verification_request_confirmation: VerificationRequestConfirmation | None = None


class AnswerQuestionRequest(AnswerQuestionPayload):
    """Internal request after the HTTP boundary has verified a Principal."""

    agent_user_id: str = Field(min_length=1)
    verified_principal: VerifiedPrincipal | None = Field(default=None, exclude=True, repr=False)
    conversation_context: ConversationContext | None = Field(default=None, exclude=True)


class SourceFreshness(BaseModel):
    validated_at: datetime
    maximum_age_seconds: int = Field(gt=0)
    is_current: bool


class VerificationRequestConfirmation(BaseModel):
    """An Agent User's affirmative approval to create a verification request."""

    approved: bool
    approval_context: str = Field(min_length=1)


class MetricDefinitionGap(BaseModel):
    requested_metric_name: str
    semantic_authority: str = "dbt/MetricFlow"
    verification_request_offered: bool = True


class ProvisionalMetricInput(BaseModel):
    name: str
    source: str


class ProvisionalMetricFreshness(BaseModel):
    source: str
    observed_at: datetime


class ProvisionalMetric(BaseModel):
    """A bounded calculation that must never be mistaken for a canonical metric."""

    name: str
    value: int | float
    formula: str = Field(min_length=1)
    inputs: list[ProvisionalMetricInput] = Field(min_length=1)
    scope: EffectiveAccessScope
    verification_status: Literal["unverified"] = "unverified"
    freshness: ProvisionalMetricFreshness
    material_caveats: list[str] = Field(min_length=1)


class DataTeamVerificationRequest(BaseModel):
    request_id: str
    requested_metric_name: str
    requested_by_agent_user_id: str
    approval_context: str
    approval_context_sha256: str
    approved_at: datetime
    decision_outcome: Literal["approved"]
    trace_id: str


class SemanticCitation(BaseModel):
    authority: str = "dbt/MetricFlow"
    artifact_path: str
    metric_name: str
    model_name: str


class CanonicalMetricDefinition(BaseModel):
    name: str
    definition: str
    formula: str
    grain: str
    time_rule: str
    semantic_version: str
    citation: SemanticCitation


class SemanticQueryEvidence(BaseModel):
    """Bounded evidence that MetricFlow compiled and Postgres executed a metric query."""

    metric_name: str
    artifact_sha256: str
    constrained_products: list[str]
    constrained_regions: list[str]
    tenant_scope: str
    result_row_count: int = Field(ge=0)


class DriverContribution(BaseModel):
    """One approved-dimension contribution to an observed metric movement."""

    region: str
    seat_tier: str
    baseline_value: int = Field(ge=0)
    comparison_value: int = Field(ge=0)
    change: int
    contribution_to_decline: int = Field(ge=0)
    percentage_of_decline: float = Field(ge=0)


class DriverDecomposition(BaseModel):
    """A reconciled, non-causal explanation derived from semantic query results."""

    metric_name: str
    baseline_period: str
    comparison_period: str
    baseline_value: int = Field(ge=0)
    comparison_value: int = Field(ge=0)
    net_change: int
    decline: int = Field(ge=0)
    contributions: list[DriverContribution]
    reconciled_change: int
    residual: int
    approved_dimensions: list[str]


class EvidenceSupportStatus(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"


class EvidenceScope(BaseModel):
    product: str
    region: str
    tenant_scope: str


class EvidenceCitation(BaseModel):
    document_id: str
    title: str
    affected_scope: EvidenceScope
    relevant_date: date
    freshness: datetime
    support_status: EvidenceSupportStatus
    support_explanation: str
    source_document_id: str
    source_url: str
    source_revision: str
    chunk_id: str


class EvidenceAnswer(BaseModel):
    citations: list[EvidenceCitation]
    support_status: EvidenceSupportStatus
    support_explanation: str


class FactorVocabularyCategory(StrEnum):
    """The small, reviewed Factor Vocabulary from spec #72; hardcoded, not a live taxonomy."""

    PROVISIONING_OR_ENTITLEMENT = "provisioning_or_entitlement"
    BILLING_OR_SUBSCRIPTION = "billing_or_subscription"
    IDENTITY_OR_ACCESS = "identity_or_access"
    PRODUCT_RELEASE_OR_REGRESSION = "product_release_or_regression"
    ONBOARDING = "onboarding"
    CAMPAIGN = "campaign"
    INCIDENT = "incident"


class CandidateCausalFactor(BaseModel):
    """A cited, falsifiable Hypothesis about a known driver; never proof of causation."""

    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(min_length=1, max_length=256)
    category: FactorVocabularyCategory
    documented_change: DriverContribution
    affected_population: EvidenceScope
    proposed_mechanism: str = Field(min_length=1, max_length=2_000)
    factor_occurrence_time: date
    citation: EvidenceCitation
    non_causal_caveat: str = Field(min_length=1, max_length=512)


class EvidenceChainReference(BaseModel):
    """Safe provenance projection for one ranked LightRAG chain reference."""

    reference_id: str
    reference_kind: Literal["chunk", "entity", "relation"]
    rank: int | None = Field(default=None, gt=0)
    source_document_id: str
    source_url: str
    source_revision: str
    chunk_id: str
    product: str
    region: str
    tenant_scope: str


class EvidenceChainChunk(BaseModel):
    """A raw supporting chunk retained with its exact source provenance."""

    reference: EvidenceChainReference
    text: str


class EvidenceChainEntity(BaseModel):
    """A bounded entity returned from the authorized graph view."""

    reference: EvidenceChainReference
    name: str
    description: str


class EvidenceChainRelation(BaseModel):
    """A bounded graph relation whose endpoints are independently authorized."""

    reference: EvidenceChainReference
    source_entity: EvidenceChainReference
    target_entity: EvidenceChainReference
    source_entity_reference_id: str
    target_entity_reference_id: str
    description: str


class EvidenceChain(BaseModel):
    """Public graph-assisted evidence, explicitly subordinate to source citations."""

    supporting_chunks: list[EvidenceChainChunk] = Field(max_length=3)
    entities: list[EvidenceChainEntity] = Field(max_length=3)
    relations: list[EvidenceChainRelation] = Field(max_length=3)
    references: list[EvidenceChainReference] = Field(max_length=3)


class GraphPathCitation(BaseModel):
    """Safe, bounded presentation of a permitted derived evidence path."""

    path_id: str
    node_labels: list[str]


class CatalogFreshness(BaseModel):
    """Availability disclosure for catalog-dependent answers."""

    available: bool
    degraded: bool
    detail: str | None = None


class CatalogMetadata(BaseModel):
    """Safe ownership, classification, and discovery metadata from DataHub."""

    entity_name: str
    entity_type: Literal["metric", "model", "dataset"]
    urn: str
    product: str
    owners: list[str]
    classification: str
    discovery_tags: list[str]
    description: str
    semantic_version: str
    source_artifact_sha256: str
    published_at: datetime


class SensitiveIdentifier(BaseModel):
    """A direct identifier returned only after explicit entitlement checks."""

    identifier_type: Literal["tenant_id", "person_id", "product_user_id"]
    value: str


class DirectIdentifierAnswer(BaseModel):
    """Bounded direct-identifier output tied to an audit event."""

    identifiers: list[SensitiveIdentifier]
    maximum_results: int = Field(gt=0)
    audit_event_id: str


class DirectIdentifierAudit(BaseModel):
    """Audit metadata that contains no returned identifier values."""

    audit_event_id: str
    trace_id: str
    agent_user_id: str
    scope: EffectiveAccessScope
    policy_fingerprint: str
    outcome: Literal["released", "no_identifiers_found"]
    recorded_at: datetime
    returned_count: int = Field(ge=0)
    maximum_results: int = Field(gt=0)


class GovernedAnalyticalResponse(BaseModel):
    answer: str
    result_classification: ResultClassification
    canonical_definition: CanonicalMetricDefinition | None = None
    semantic_query_evidence: SemanticQueryEvidence | None = None
    driver_decomposition: DriverDecomposition | None = None
    evidence: EvidenceAnswer | None = None
    evidence_chain: EvidenceChain | None = None
    candidate_causal_factor: CandidateCausalFactor | None = None
    graph_paths: list[GraphPathCitation] | None = None
    catalog_metadata: CatalogMetadata | None = None
    catalog_freshness: CatalogFreshness | None = None
    direct_identifier_answer: DirectIdentifierAnswer | None = None
    direct_identifier_audit: DirectIdentifierAudit | None = None
    metric_definition_gap: MetricDefinitionGap | None = None
    provisional_metric: ProvisionalMetric | None = None
    data_team_verification_request: DataTeamVerificationRequest | None = None
    lead_agent_metadata: LeadAgentMetadata | None = None
    source_freshness: SourceFreshness
    effective_access_scope: EffectiveAccessScope
    caveats: list[str]
    trace_id: str
    conversation_id: str | None = None

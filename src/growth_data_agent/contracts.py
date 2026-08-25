"""Typed contracts exposed by the governed analytical response seam."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ResultClassification(StrEnum):
    CANONICAL_DEFINITION = "canonical_definition"
    CATALOG_OWNERSHIP = "catalog_ownership"
    DRIVER_DECOMPOSITION = "driver_decomposition"
    HYPOTHESIS = "hypothesis"
    INCONCLUSIVE = "inconclusive"
    CAUSAL_ESTIMATE = "causal_estimate"
    DESCRIPTIVE_RESULT = "descriptive_result"
    ANALYSIS_PLAN = "analysis_plan"
    DIRECT_IDENTIFIER_RESPONSE = "direct_identifier_response"
    SAFE_REFUSAL = "safe_refusal"
    METRIC_DEFINITION_GAP = "metric_definition_gap"
    PROVISIONAL_METRIC = "provisional_metric"
    LIMITATION = "limitation"


class AnswerQuestionRequest(BaseModel):
    agent_user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_metric_name: str | None = Field(default=None, min_length=1)
    experiment_id: str | None = Field(default=None, min_length=1)
    verification_request_confirmation: VerificationRequestConfirmation | None = None


class EffectiveAccessScope(BaseModel):
    products: list[str]
    regions: list[str]
    tenant_scope: str
    permitted_columns: list[str]


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
    approved_at: datetime


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


class EvidenceAnswer(BaseModel):
    citations: list[EvidenceCitation]
    support_status: EvidenceSupportStatus
    support_explanation: str


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
    causal_registration: CausalDesignRegistration | None = None
    causal_estimate: CausalEstimate | None = None
    descriptive_comparison: DescriptiveComparison | None = None
    causal_analysis_plan: CausalAnalysisPlan | None = None
    graph_paths: list[GraphPathCitation] | None = None
    catalog_metadata: CatalogMetadata | None = None
    catalog_freshness: CatalogFreshness | None = None
    direct_identifier_answer: DirectIdentifierAnswer | None = None
    direct_identifier_audit: DirectIdentifierAudit | None = None
    metric_definition_gap: MetricDefinitionGap | None = None
    provisional_metric: ProvisionalMetric | None = None
    data_team_verification_request: DataTeamVerificationRequest | None = None
    source_freshness: SourceFreshness
    effective_access_scope: EffectiveAccessScope
    caveats: list[str]
    trace_id: str


class CausalDesignType(StrEnum):
    RANDOMIZED_EXPERIMENT = "randomized_experiment"
    OBSERVATIONAL = "observational"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    ALL_USER_PRE_POST = "all_user_pre_post"


class CausalReviewStatus(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"


class CausalSupportCheck(BaseModel):
    name: str = Field(min_length=1)
    passed: bool
    details: str = Field(min_length=1)


class CausalDiagnostic(BaseModel):
    name: str = Field(min_length=1)
    passed: bool
    details: str = Field(min_length=1)


class EstimatorApproval(BaseModel):
    estimator: str = Field(min_length=1)
    approved: bool
    approved_by: str = Field(min_length=1)
    approved_at: datetime


class CausalReview(BaseModel):
    status: CausalReviewStatus
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    decision: str = Field(min_length=1)


class CausalDesignRegistration(BaseModel):
    """The governance record required before causal estimation."""

    experiment_id: str = Field(min_length=1)
    product: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    treatment: str = Field(min_length=1)
    control: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    design_type: CausalDesignType
    assignment_unit: str = Field(min_length=1)
    regions: list[str] = Field(default_factory=list)
    tenant_scope: str = ""
    seat_tier: str | None = None
    support_checks: list[CausalSupportCheck] = Field(default_factory=list)
    estimator_approval: EstimatorApproval | None = None
    diagnostics: list[CausalDiagnostic] = Field(default_factory=list)
    review: CausalReview | None = None
    assumptions: list[str] = Field(default_factory=list)


class CausalOutcomeData(BaseModel):
    """Bounded treatment/control outcome data supplied to the causal pipeline."""

    treatment_value: float
    control_value: float
    standard_error: float = Field(ge=0)


class CausalEstimate(BaseModel):
    """A causal effect emitted only after the deterministic governance gate passes."""

    experiment_id: str
    treatment: str
    control: str
    outcome: str
    estimator: str
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    assumptions: list[str] = Field(min_length=1)
    diagnostics: list[CausalDiagnostic] = Field(min_length=1)


class DescriptiveComparison(BaseModel):
    """Observed treatment/control values shown without causal interpretation."""

    experiment_id: str
    treatment: str
    control: str
    outcome: str
    treatment_value: float
    control_value: float
    difference: float


class CausalAnalysisPlan(BaseModel):
    """A safe next step when a design cannot yet produce a Causal Estimate."""

    experiment_id: str | None
    design_type: CausalDesignType | None
    proposed_estimator: str | None
    review_status: CausalReviewStatus = CausalReviewStatus.PENDING
    reason: str = Field(min_length=1)
    required_actions: list[str] = Field(min_length=1)


class CausalEvaluation(BaseModel):
    """The deterministic pipeline outcome before it is rendered as an answer."""

    outcome: Literal["causal_estimate", "descriptive_result", "analysis_plan"]
    registration: CausalDesignRegistration | None
    causal_estimate: CausalEstimate | None
    analysis_plan: CausalAnalysisPlan | None
    descriptive_comparison: DescriptiveComparison | None = None

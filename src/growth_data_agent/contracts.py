"""Typed contracts exposed by the governed analytical response seam."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResultClassification(StrEnum):
    CANONICAL_DEFINITION = "canonical_definition"
    DRIVER_DECOMPOSITION = "driver_decomposition"
    LIMITATION = "limitation"


class AnswerQuestionRequest(BaseModel):
    agent_user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class EffectiveAccessScope(BaseModel):
    products: list[str]
    regions: list[str]
    tenant_scope: str
    permitted_columns: list[str]


class SourceFreshness(BaseModel):
    validated_at: datetime
    maximum_age_seconds: int = Field(gt=0)
    is_current: bool


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


class GovernedAnalyticalResponse(BaseModel):
    answer: str
    result_classification: ResultClassification
    canonical_definition: CanonicalMetricDefinition | None = None
    semantic_query_evidence: SemanticQueryEvidence | None = None
    driver_decomposition: DriverDecomposition | None = None
    source_freshness: SourceFreshness
    effective_access_scope: EffectiveAccessScope
    caveats: list[str]
    trace_id: str

"""Typed contracts exposed by the governed analytical response seam."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResultClassification(StrEnum):
    CANONICAL_DEFINITION = "canonical_definition"
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


class GovernedAnalyticalResponse(BaseModel):
    answer: str
    result_classification: ResultClassification
    canonical_definition: CanonicalMetricDefinition | None = None
    semantic_query_evidence: SemanticQueryEvidence | None = None
    source_freshness: SourceFreshness
    effective_access_scope: EffectiveAccessScope
    caveats: list[str]
    trace_id: str

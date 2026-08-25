"""Typed contracts exposed by the governed analytical response seam."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ResultClassification(StrEnum):
    CANONICAL_DEFINITION = "canonical_definition"
    METRIC_DEFINITION_GAP = "metric_definition_gap"
    PROVISIONAL_METRIC = "provisional_metric"
    LIMITATION = "limitation"


class AnswerQuestionRequest(BaseModel):
    agent_user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_metric_name: str | None = Field(default=None, min_length=1)
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


class GovernedAnalyticalResponse(BaseModel):
    answer: str
    result_classification: ResultClassification
    canonical_definition: CanonicalMetricDefinition | None = None
    semantic_query_evidence: SemanticQueryEvidence | None = None
    metric_definition_gap: MetricDefinitionGap | None = None
    provisional_metric: ProvisionalMetric | None = None
    data_team_verification_request: DataTeamVerificationRequest | None = None
    source_freshness: SourceFreshness
    effective_access_scope: EffectiveAccessScope
    caveats: list[str]
    trace_id: str

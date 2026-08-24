"""Read and validate dbt/MetricFlow semantic artifacts.

No metric formula is calculated here. The artifact is the semantic authority;
this module only determines whether it is safe to describe as canonical.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import CanonicalMetricDefinition, SemanticCitation, SourceFreshness


class ArtifactValidation(BaseModel):
    status: str
    validated_at: datetime
    maximum_age_seconds: int = Field(gt=0)


class MetricArtifact(BaseModel):
    name: str
    definition: str
    formula: str
    grain: str
    time_rule: str
    model_name: str
    citation_path: str


class SemanticArtifact(BaseModel):
    artifact_type: str
    semantic_version: str
    validation: ArtifactValidation
    metrics: list[MetricArtifact]


class SemanticArtifactStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> SemanticArtifact:
        return SemanticArtifact.model_validate_json(self.path.read_text())


class ValidatedMetricFlowGateway:
    """Expose only a current, successfully validated semantic definition."""

    def __init__(
        self,
        artifact_store: SemanticArtifactStore,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.artifact_store = artifact_store
        self.now = now or (lambda: datetime.now(UTC))

    def freshness(self, artifact: SemanticArtifact) -> SourceFreshness:
        validation = artifact.validation
        validated_at = validation.validated_at.astimezone(UTC)
        current_time = self.now().astimezone(UTC)
        age = current_time - validated_at
        is_current = (
            validation.status == "success"
            and age >= timedelta(0)
            and age <= timedelta(seconds=validation.maximum_age_seconds)
        )
        return SourceFreshness(
            validated_at=validated_at,
            maximum_age_seconds=validation.maximum_age_seconds,
            is_current=is_current,
        )

    def canonical_definition(
        self, metric_name: str
    ) -> tuple[CanonicalMetricDefinition | None, SourceFreshness]:
        artifact = self.artifact_store.load()
        freshness = self.freshness(artifact)
        if not freshness.is_current:
            return None, freshness

        metric = next((item for item in artifact.metrics if item.name == metric_name), None)
        if metric is None:
            return None, freshness

        return (
            CanonicalMetricDefinition(
                name=metric.name,
                definition=metric.definition,
                formula=metric.formula,
                grain=metric.grain,
                time_rule=metric.time_rule,
                semantic_version=artifact.semantic_version,
                citation=SemanticCitation(
                    artifact_path=metric.citation_path,
                    metric_name=metric.name,
                    model_name=metric.model_name,
                ),
            ),
            freshness,
        )

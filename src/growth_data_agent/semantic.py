"""Read and validate dbt/MetricFlow semantic artifacts.

No metric formula is calculated here. The artifact is the semantic authority;
this module only determines whether it is safe to describe as canonical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import (
    CanonicalMetricDefinition,
    DriverContribution,
    DriverDecomposition,
    SemanticCitation,
    SemanticQueryEvidence,
    SourceFreshness,
)
from .metricflow_query import (
    MetricFlowPlanner,
    MetricFlowQueryRequest,
    PostgresMetricFlowExecutor,
    SemanticQueryExecutionError,
)
from .policy import AccessProfile


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
    semantic_manifest_sha256: str
    validation: ArtifactValidation
    metrics: list[MetricArtifact]


class SemanticArtifactStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> SemanticArtifact:
        return SemanticArtifact.model_validate_json(self.path.read_text())


@dataclass(frozen=True)
class ValidatedMetricQueryContext:
    """The validated semantic inputs shared by all canonical metric queries."""

    artifact: SemanticArtifact
    metric: MetricArtifact
    freshness: SourceFreshness


class ValidatedMetricFlowGateway:
    """Expose only a current, successfully validated semantic definition."""

    def __init__(
        self,
        artifact_store: SemanticArtifactStore,
        *,
        metricflow_planner: MetricFlowPlanner | None = None,
        postgres_executor: PostgresMetricFlowExecutor | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.artifact_store = artifact_store
        self.metricflow_planner = metricflow_planner
        self.postgres_executor = postgres_executor
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

        return self._canonical_definition(metric, artifact), freshness

    def execute_scoped_metric(
        self,
        metric_name: str,
        access_profile: AccessProfile,
        *,
        scoped_regions: tuple[str, ...] | None = None,
        scoped_seat_tier: str | None = None,
        scoped_tenant_ids: tuple[str, ...] | None = None,
        scoped_tenant_scope: str | None = None,
    ) -> tuple[SemanticQueryEvidence | None, SourceFreshness]:
        """Plan and execute one entitlement-constrained aggregate after validation."""
        context, freshness = self._validated_metric_query_context(metric_name)
        if context is None:
            return None, freshness

        metric_product = _metric_product(metric_name)
        entity_name = _metricflow_entity(metric_name)
        constraints = list(
            access_profile.metricflow_where_constraints(
                metric_product, entity_name=entity_name
            )
        )
        if scoped_regions is not None:
            access_profile.authorize_query_columns((f"{entity_name}__region",))
            regions = ", ".join(repr(region) for region in scoped_regions)
            constraints.append(f"{entity_name}__region IN ({regions})")
        if scoped_seat_tier is not None:
            access_profile.authorize_query_columns((f"{entity_name}__seat_tier",))
            constraints.append(f"{entity_name}__seat_tier = '{scoped_seat_tier}'")
        if scoped_tenant_ids is not None:
            access_profile.authorize_query_columns((f"{entity_name}__tenant_id",))
            tenant_ids = ", ".join(repr(tenant_id) for tenant_id in scoped_tenant_ids)
            constraints.append(f"{entity_name}__tenant_id IN ({tenant_ids})")
        group_by_names = (f"{entity_name}__product",)
        if len(access_profile.regions) != 3:
            group_by_names += (f"{entity_name}__region",)
        access_profile.authorize_query_columns(group_by_names)
        plan = self.metricflow_planner.plan(
            MetricFlowQueryRequest(
                metric_name=metric_name,
                where_constraints=tuple(constraints),
                group_by_names=group_by_names,
            )
        )
        result_row_count = self.postgres_executor.execute(plan)
        return (
            self._query_evidence(
                context,
                access_profile,
                result_row_count,
                metric_product,
                constrained_regions=scoped_regions,
                tenant_scope=scoped_tenant_scope,
            ),
            freshness,
        )

    def driver_decomposition(
        self,
        metric_name: str,
        access_profile: AccessProfile,
        *,
        baseline_period: str,
        comparison_period: str,
    ) -> tuple[
        CanonicalMetricDefinition | None,
        DriverDecomposition | None,
        SemanticQueryEvidence | None,
        SourceFreshness,
    ]:
        """Reconcile approved dimensional aggregates from a validated MetricFlow query.

        MetricFlow computes the canonical metric. This boundary only compares the
        returned monthly aggregates; it never reconstructs the metric formula.
        """
        context, freshness = self._validated_metric_query_context(metric_name)
        if context is None:
            return None, None, None, freshness

        entity_name = _metricflow_entity(metric_name)
        group_by_names = (
            "metric_time__month",
            f"{entity_name}__product",
            f"{entity_name}__region",
            f"{entity_name}__seat_tier",
        )
        access_profile.authorize_query_columns(group_by_names)
        plan = self.metricflow_planner.plan(
            MetricFlowQueryRequest(
                metric_name=metric_name,
                where_constraints=access_profile.metricflow_where_constraints(
                    _metric_product(metric_name), entity_name=entity_name
                ),
                group_by_names=group_by_names,
                limit=None,
            )
        )
        rows = self.postgres_executor.execute_rows(plan)
        decomposition = _reconcile_driver_rows(
            rows,
            metric_name=metric_name,
            baseline_period=baseline_period,
            comparison_period=comparison_period,
            dimension_prefix=entity_name,
        )
        return (
            self._canonical_definition(context.metric, context.artifact),
            decomposition,
            self._query_evidence(
                context, access_profile, len(rows), _metric_product(metric_name)
            ),
            freshness,
        )

    def _validated_metric_query_context(
        self, metric_name: str
    ) -> tuple[ValidatedMetricQueryContext | None, SourceFreshness]:
        """Authorize canonical querying from one current, hash-matched semantic artifact."""
        artifact = self.artifact_store.load()
        freshness = self.freshness(artifact)
        metric = next((item for item in artifact.metrics if item.name == metric_name), None)
        if not freshness.is_current or metric is None:
            return None, freshness
        if self.metricflow_planner is None or self.postgres_executor is None:
            raise SemanticQueryExecutionError("Semantic query execution is not configured.")

        semantic_manifest = self.metricflow_planner.semantic_manifest_path
        actual_hash = sha256(semantic_manifest.read_bytes()).hexdigest()
        if actual_hash != artifact.semantic_manifest_sha256:
            raise SemanticQueryExecutionError(
                "The semantic manifest does not match the validated artifact."
            )
        return ValidatedMetricQueryContext(artifact, metric, freshness), freshness

    @staticmethod
    def _canonical_definition(
        metric: MetricArtifact, artifact: SemanticArtifact
    ) -> CanonicalMetricDefinition:
        return CanonicalMetricDefinition(
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
        )

    @staticmethod
    def _query_evidence(
        context: ValidatedMetricQueryContext,
        access_profile: AccessProfile,
        result_row_count: int,
        metric_product: str,
        *,
        constrained_regions: tuple[str, ...] | None = None,
        tenant_scope: str | None = None,
    ) -> SemanticQueryEvidence:
        return SemanticQueryEvidence(
            metric_name=context.metric.name,
            artifact_sha256=context.artifact.semantic_manifest_sha256,
            constrained_products=[metric_product],
            constrained_regions=list(constrained_regions or access_profile.regions),
            tenant_scope=tenant_scope or access_profile.tenant_scope,
            result_row_count=result_row_count,
        )


def _reconcile_driver_rows(
    rows: list[Mapping[str, object]],
    *,
    metric_name: str,
    baseline_period: str,
    comparison_period: str,
    dimension_prefix: str = "product_user",
) -> DriverDecomposition:
    by_segment: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        period = _month_label(row["metric_time__month"])
        if period not in (baseline_period, comparison_period):
            continue
        segment = (
            str(row[f"{dimension_prefix}__region"]),
            str(row[f"{dimension_prefix}__seat_tier"]),
        )
        by_segment.setdefault(segment, {})[period] = int(row[metric_name])

    contributions = [
        _driver_contribution(segment, values, baseline_period, comparison_period)
        for segment, values in by_segment.items()
    ]
    contributions.sort(key=lambda item: (-abs(item.change), item.region, item.seat_tier))
    baseline_value = sum(item.baseline_value for item in contributions)
    comparison_value = sum(item.comparison_value for item in contributions)
    net_change = comparison_value - baseline_value
    decline = max(-net_change, 0)
    if decline:
        contributions = [
            item.model_copy(
                update={
                    "percentage_of_decline": round(
                        item.contribution_to_decline / decline * 100,
                        2,
                    )
                }
            )
            for item in contributions
        ]
    reconciled_change = sum(item.change for item in contributions)
    return DriverDecomposition(
        metric_name=metric_name,
        baseline_period=baseline_period,
        comparison_period=comparison_period,
        baseline_value=baseline_value,
        comparison_value=comparison_value,
        net_change=net_change,
        decline=decline,
        contributions=contributions,
        reconciled_change=reconciled_change,
        residual=net_change - reconciled_change,
        approved_dimensions=["Region", "Seat Tier"],
    )


def _driver_contribution(
    segment: tuple[str, str],
    values: Mapping[str, int],
    baseline_period: str,
    comparison_period: str,
) -> DriverContribution:
    baseline_value = values.get(baseline_period, 0)
    comparison_value = values.get(comparison_period, 0)
    change = comparison_value - baseline_value
    return DriverContribution(
        region=segment[0],
        seat_tier=segment[1],
        baseline_value=baseline_value,
        comparison_value=comparison_value,
        change=change,
        contribution_to_decline=max(-change, 0),
        percentage_of_decline=0,
    )


def _month_label(value: object) -> str:
    return str(value)[:7]


def _metric_product(metric_name: str) -> str:
    if metric_name.startswith("jira_"):
        return "Jira"
    if metric_name.startswith("confluence_"):
        return "Confluence"
    raise SemanticQueryExecutionError(f"Metric has no governed product scope: {metric_name}")


def _metricflow_entity(metric_name: str) -> str:
    if metric_name == "jira_new_mau":
        return "jira_new_mau_product_user"
    if metric_name == "confluence_new_mau":
        return "confluence_new_mau_product_user"
    if metric_name.startswith("confluence_"):
        return "confluence_product_user"
    return "product_user"

"""Plan validated semantic metrics with MetricFlow and execute them read-only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import psycopg
from metricflow.engine.metricflow_engine import (
    MetricFlowEngine,
)
from metricflow.engine.metricflow_engine import (
    MetricFlowQueryRequest as MetricFlowEngineQueryRequest,
)
from metricflow.protocols.sql_client import SqlEngine
from metricflow.sql.render.postgres import PostgresSQLSqlPlanRenderer
from metricflow_semantics.model.dbt_manifest_parser import (
    parse_manifest_from_dbt_generated_manifest,
)
from metricflow_semantics.model.semantic_manifest_lookup import SemanticManifestLookup
from metricflow_semantics.sql.sql_bind_parameters import SqlBindParameterSet


class SemanticQueryExecutionError(RuntimeError):
    """Raised when a canonical query cannot be safely planned or executed."""


@dataclass(frozen=True)
class MetricFlowQueryRequest:
    """The only query shape this delivery permits: one aggregate, no group-by."""

    metric_name: str
    where_constraints: tuple[str, ...]
    group_by_names: tuple[str, ...]


@dataclass(frozen=True)
class PlannedMetricFlowQuery:
    metric_name: str
    sql: str
    parameters: Mapping[str, object]


class MetricFlowPlanner:
    """Compile an approved metric request from a dbt-generated semantic manifest."""

    def __init__(self, semantic_manifest_path: Path):
        self.semantic_manifest_path = semantic_manifest_path

    def plan(self, request: MetricFlowQueryRequest) -> PlannedMetricFlowQuery:
        manifest = parse_manifest_from_dbt_generated_manifest(
            self.semantic_manifest_path.read_text()
        )
        engine = MetricFlowEngine(
            SemanticManifestLookup(manifest),
            _PostgresPlanningClient(),
        )
        explanation = engine.explain(
            MetricFlowEngineQueryRequest.create(
                metric_names=(request.metric_name,),
                group_by_names=request.group_by_names,
                where_constraints=request.where_constraints,
                limit=1,
            )
        )
        statement = explanation.sql_statement.without_descriptions
        if not _is_single_read_only_select(statement.sql):
            raise SemanticQueryExecutionError("MetricFlow produced a non-read-only query plan.")
        return PlannedMetricFlowQuery(
            metric_name=request.metric_name,
            sql=statement.sql,
            parameters=statement.bind_parameter_set.param_dict,
        )


class PostgresMetricFlowExecutor:
    """Execute only MetricFlow-produced SELECT plans in a read-only transaction."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def execute(self, plan: PlannedMetricFlowQuery) -> int:
        if not _is_single_read_only_select(plan.sql):
            raise SemanticQueryExecutionError("Refusing to execute a non-read-only query plan.")
        with psycopg.connect(self.database_url) as connection:
            with connection.transaction():
                connection.execute("SET TRANSACTION READ ONLY")
                result = connection.execute(plan.sql, plan.parameters)
                return len(result.fetchall())


class _PostgresPlanningClient:
    """MetricFlow's SQL client seam used solely to render a Postgres query plan."""

    sql_engine_type = SqlEngine.POSTGRES
    sql_plan_renderer = PostgresSQLSqlPlanRenderer()

    def query(self, stmt: str, sql_bind_parameter_set: SqlBindParameterSet) -> None:
        raise AssertionError("MetricFlow planning must not execute SQL")

    def execute(self, stmt: str, sql_bind_parameter_set: SqlBindParameterSet) -> None:
        raise AssertionError("MetricFlow planning must not execute SQL")

    def dry_run(self, stmt: str, sql_bind_parameter_set: SqlBindParameterSet) -> None:
        raise AssertionError("MetricFlow planning must not execute SQL")

    def close(self) -> None:
        return None

    def render_bind_parameter_key(self, bind_parameter_key: str) -> str:
        return f"%({bind_parameter_key})s"


def _is_single_read_only_select(sql: str) -> bool:
    normalized = sql.strip().casefold()
    return normalized.startswith(("select", "with")) and ";" not in normalized

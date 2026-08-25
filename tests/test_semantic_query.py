from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact

from growth_data_agent.metricflow_query import (
    MetricFlowPlanner,
    MetricFlowQueryRequest,
    PlannedMetricFlowQuery,
    PostgresMetricFlowExecutor,
    SemanticQueryExecutionError,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway


def _gateway(
    tmp_path: Path,
) -> tuple[ValidatedMetricFlowGateway, RecordingMetricFlowPlanner, RecordingPostgresExecutor]:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    return (
        ValidatedMetricFlowGateway(
            SemanticArtifactStore(artifact_path),
            metricflow_planner=planner,
            postgres_executor=executor,
            now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
        ),
        planner,
        executor,
    )


def test_apac_entitlement_is_passed_to_metricflow_before_postgres_execution(tmp_path: Path) -> None:
    gateway, planner, executor = _gateway(tmp_path)

    evidence, freshness = gateway.execute_scoped_metric(
        "jira_new_peu", resolve_access_profile("apac_regional_manager")
    )

    assert freshness.is_current is True
    assert evidence is not None
    assert planner.requests[0].where_constraints == (
        "product_user__product = 'Jira'",
        "product_user__region IN ('APAC')",
    )
    assert executor.plans == [
        PlannedMetricFlowQuery("jira_new_peu", "select 1 as jira_new_peu", {})
    ]


def test_stale_artifact_does_not_plan_or_execute_a_query(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=executor,
        now=lambda: datetime(2026, 8, 25, tzinfo=UTC) + timedelta(days=2),
    )

    evidence, freshness = gateway.execute_scoped_metric(
        "jira_new_peu", resolve_access_profile("data_analyst")
    )

    assert evidence is None
    assert freshness.is_current is False
    assert planner.requests == []
    assert executor.plans == []


def test_metricflow_planner_compiles_the_validated_semantic_manifest() -> None:
    repository = Path(__file__).resolve().parents[1]
    target_manifest = repository / "dbt/target/semantic_manifest.json"
    if not target_manifest.exists():
        pytest.skip("dbt semantic manifest is not available; run dbt parse first")

    plan = MetricFlowPlanner(target_manifest).plan(
        MetricFlowQueryRequest(
            metric_name="jira_new_peu",
            where_constraints=(
                "product_user__product = 'Jira'",
                "product_user__region IN ('APAC')",
            ),
            group_by_names=("product_user__product", "product_user__region"),
        )
    )

    assert plan.metric_name == "jira_new_peu"
    assert plan.sql.lstrip().casefold().startswith(("select", "with"))
    assert "fct_jira_new_peu" in plan.sql
    assert "APAC" in plan.sql


def test_postgres_executor_refuses_non_select_sql() -> None:
    executor = PostgresMetricFlowExecutor("postgresql://unused")

    with pytest.raises(SemanticQueryExecutionError, match="non-read-only"):
        executor.execute(PlannedMetricFlowQuery("jira_new_peu", "delete from public.tenants", {}))

    with pytest.raises(SemanticQueryExecutionError, match="non-read-only"):
        executor.execute(
            PlannedMetricFlowQuery("jira_new_peu", "select 1; delete from public.tenants", {})
        )


def test_metricflow_generated_sql_executes_on_postgres() -> None:
    database_url = os.environ.get("METRICFLOW_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip(
            "METRICFLOW_TEST_DATABASE_URL is required for the local Postgres integration test"
        )

    repository = Path(__file__).resolve().parents[1]
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(repository / "dbt/artifacts/last_validated_semantic.json"),
        metricflow_planner=MetricFlowPlanner(repository / "dbt/target/semantic_manifest.json"),
        postgres_executor=PostgresMetricFlowExecutor(database_url),
    )

    evidence, freshness = gateway.execute_scoped_metric(
        "jira_new_peu", resolve_access_profile("apac_regional_manager")
    )

    assert freshness.is_current is True
    assert evidence is not None
    assert evidence.result_row_count == 1
    assert evidence.constrained_regions == ["APAC"]

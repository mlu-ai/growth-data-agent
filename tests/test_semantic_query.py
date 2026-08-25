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
        PlannedMetricFlowQuery(
            "jira_new_peu",
            "select 1 as jira_new_peu",
            {},
            where_constraints=(
                "product_user__product = 'Jira'",
                "product_user__region IN ('APAC')",
            ),
            group_by_names=("product_user__product", "product_user__region"),
        )
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


def test_driver_decomposition_uses_only_approved_dimensions_and_apac_row_scope(
    tmp_path: Path,
) -> None:
    gateway, planner, executor = _gateway(tmp_path)

    definition, decomposition, evidence, freshness = gateway.driver_decomposition(
        "jira_new_peu",
        resolve_access_profile("apac_regional_manager"),
        baseline_period="2026-05",
        comparison_period="2026-06",
    )

    assert freshness.is_current is True
    assert definition is not None
    assert definition.semantic_version == "1.0.0"
    assert evidence is not None
    assert decomposition is not None
    assert planner.requests[0].where_constraints == (
        "product_user__product = 'Jira'",
        "product_user__region IN ('APAC')",
    )
    assert planner.requests[0].group_by_names == (
        "metric_time__month",
        "product_user__product",
        "product_user__region",
        "product_user__seat_tier",
    )
    assert planner.requests[0].limit is None
    assert evidence.constrained_regions == ["APAC"]
    assert decomposition.approved_dimensions == ["Region", "Seat Tier"]
    assert all(item.region == "APAC" for item in decomposition.contributions)


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


def test_metricflow_planner_compiles_the_confluence_semantic_manifest() -> None:
    repository = Path(__file__).resolve().parents[1]
    target_manifest = repository / "dbt/target/semantic_manifest.json"
    if not target_manifest.exists():
        pytest.skip("dbt semantic manifest is not available; run dbt parse first")

    plan = MetricFlowPlanner(target_manifest).plan(
        MetricFlowQueryRequest(
            metric_name="confluence_new_peu",
            where_constraints=(
                "confluence_product_user__product = 'Confluence'",
                "confluence_product_user__region IN ('Americas')",
            ),
            group_by_names=(
                "metric_time__month",
                "confluence_product_user__product",
                "confluence_product_user__region",
                "confluence_product_user__seat_tier",
            ),
            limit=None,
        )
    )

    assert plan.metric_name == "confluence_new_peu"
    assert plan.sql.lstrip().casefold().startswith(("select", "with"))
    assert "fct_confluence_new_peu" in plan.sql
    assert "Americas" in plan.sql


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


def test_confluence_metricflow_query_and_driver_reconcile_on_postgres() -> None:
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

    definition, decomposition, evidence, freshness = gateway.driver_decomposition(
        "confluence_new_peu",
        resolve_access_profile("data_analyst"),
        baseline_period="2026-05",
        comparison_period="2026-06",
    )

    assert freshness.is_current is True
    assert definition is not None
    assert definition.name == "confluence_new_peu"
    assert decomposition is not None
    assert (decomposition.baseline_value, decomposition.comparison_value) == (2400, 2820)
    assert decomposition.net_change == decomposition.reconciled_change == 420
    assert decomposition.residual == 0
    assert decomposition.contributions[0].region == "Americas"
    assert decomposition.contributions[0].seat_tier == "11-50"
    assert decomposition.contributions[0].change == 420
    assert evidence is not None
    assert evidence.constrained_products == ["Confluence"]

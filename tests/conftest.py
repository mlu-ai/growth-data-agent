from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import PlannedMetricFlowQuery
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


def write_artifact(path: Path, *, status: str = "success", hours_old: int = 0) -> Path:
    validated_at = datetime(2026, 8, 25, tzinfo=UTC)
    semantic_manifest_path = path.with_name("semantic_manifest.json")
    semantic_manifest_path.write_text('{"semantic_models": [], "metrics": []}')
    artifact = {
        "artifact_type": "dbt_metricflow_semantic_artifact",
        "semantic_version": "1.0.0",
        "semantic_manifest_sha256": sha256(semantic_manifest_path.read_bytes()).hexdigest(),
        "validation": {
            "status": status,
            "validated_at": validated_at.isoformat(),
            "maximum_age_seconds": 86400,
        },
        "metrics": [
            {
                "name": "jira_new_peu",
                "definition": "A Product User's first-ever Paid Enablement for Jira.",
                "formula": (
                    "count_distinct(product_user_id) where product = Jira "
                    "and paid_enablement_ordinal = 1"
                ),
                "grain": "Product User in a Tenant and Jira product",
                "time_rule": "Attribute to first-ever Jira Paid Enablement.",
                "model_name": "fct_jira_new_peu",
                "citation_path": "dbt/models/marts/jira_new_peu.yml#jira_new_peu",
            },
            {
                "name": "confluence_new_peu",
                "definition": (
                    "Confluence New PEU is a Product User's first-ever Paid Enablement "
                    "for Confluence."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Confluence product",
                "time_rule": (
                    "Attribute to the first-ever Confluence Paid Enablement; later "
                    "restorations do not qualify again."
                ),
                "model_name": "fct_confluence_new_peu",
                "citation_path": "dbt/models/marts/confluence_new_peu.yml#confluence_new_peu",
            },
        ],
    }
    path.write_text(json.dumps(artifact))
    return path


class RecordingMetricFlowPlanner:
    def __init__(self, semantic_manifest_path: Path):
        self.semantic_manifest_path = semantic_manifest_path
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return PlannedMetricFlowQuery(
            metric_name=request.metric_name,
            sql=f"select 1 as {request.metric_name}",
            parameters={},
            where_constraints=request.where_constraints,
            group_by_names=request.group_by_names,
        )


def _driver_row(month: str, region: str, seat_tier: str, value: int) -> dict[str, object]:
    return {
        "metric_time__month": f"{month}-01",
        "product_user__region": region,
        "product_user__seat_tier": seat_tier,
        "jira_new_peu": value,
    }


def _confluence_driver_row(
    month: str, region: str, seat_tier: str, value: int
) -> dict[str, object]:
    return {
        "metric_time__month": f"{month}-01",
        "confluence_product_user__region": region,
        "confluence_product_user__seat_tier": seat_tier,
        "confluence_new_peu": value,
    }


class RecordingPostgresExecutor:
    def __init__(self) -> None:
        self.plans = []

    _driver_rows = [
        _driver_row("2026-05", "APAC", "51-200", 800),
        _driver_row("2026-06", "APAC", "51-200", 380),
        _driver_row("2026-05", "Americas", "1-10", 1000),
        _driver_row("2026-06", "Americas", "1-10", 940),
        _driver_row("2026-05", "EMEA", "11-50", 700),
        _driver_row("2026-06", "EMEA", "11-50", 680),
        _driver_row("2026-05", "APAC", "1-10", 600),
        _driver_row("2026-06", "APAC", "1-10", 580),
        _driver_row("2026-05", "EMEA", "51-200", 500),
        _driver_row("2026-06", "EMEA", "51-200", 480),
        _driver_row("2026-05", "Americas", "51-200", 400),
        _driver_row("2026-06", "Americas", "51-200", 380),
    ]
    _confluence_driver_rows = [
        _confluence_driver_row("2026-05", "Americas", "11-50", 1200),
        _confluence_driver_row("2026-06", "Americas", "11-50", 1620),
        _confluence_driver_row("2026-05", "APAC", "1-10", 600),
        _confluence_driver_row("2026-06", "APAC", "1-10", 600),
        _confluence_driver_row("2026-05", "EMEA", "51-200", 600),
        _confluence_driver_row("2026-06", "EMEA", "51-200", 600),
    ]

    def execute(self, plan: PlannedMetricFlowQuery) -> int:
        self.plans.append(plan)
        return 1

    def execute_rows(self, plan: PlannedMetricFlowQuery):
        self.plans.append(plan)
        if plan.metric_name == "confluence_new_peu":
            return self._confluence_driver_rows
        if "product_user__region IN ('APAC')" in plan.where_constraints:
            return [row for row in self._driver_rows if row["product_user__region"] == "APAC"]
        return self._driver_rows


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=executor,
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    return TestClient(create_app(AnswerQuestionService(gateway)))

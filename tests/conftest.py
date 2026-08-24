from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from growth_data_agent.main import create_app
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


def write_artifact(path: Path, *, status: str = "success", hours_old: int = 0) -> Path:
    validated_at = datetime(2026, 8, 25, tzinfo=UTC)
    artifact = {
        "artifact_type": "dbt_metricflow_semantic_artifact",
        "semantic_version": "1.0.0",
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
            }
        ],
    }
    path.write_text(json.dumps(artifact))
    return path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    return TestClient(create_app(AnswerQuestionService(gateway)))

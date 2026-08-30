from datetime import UTC, datetime, timedelta

from conftest import write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.main import create_app
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService


def _client_for_artifact(path, *, now: datetime) -> TestClient:
    gateway = ValidatedMetricFlowGateway(SemanticArtifactStore(path), now=lambda: now)
    return TestClient(create_app(AnswerQuestionService(gateway)))


def test_failed_semantic_artifact_blocks_canonical_response(tmp_path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    client = _client_for_artifact(write_artifact(tmp_path / "failed.json", status="fail"), now=now)

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["result_classification"] == "limitation"
    assert body["canonical_definition"] is None
    assert body["source_freshness"]["is_current"] is False
    assert "cannot be returned as canonical" in body["answer"]


def test_stale_semantic_artifact_blocks_canonical_response(tmp_path) -> None:
    validated_at = datetime(2026, 8, 25, tzinfo=UTC)
    client = _client_for_artifact(
        write_artifact(tmp_path / "stale.json"), now=validated_at + timedelta(days=2)
    )

    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    body = response.json()
    assert body["result_classification"] == "limitation"
    assert body["canonical_definition"] is None
    assert body["source_freshness"]["is_current"] is False


def test_available_metric_names_only_exposes_a_current_validated_artifact(tmp_path) -> None:
    validated_at = datetime(2026, 8, 25, tzinfo=UTC)
    current_gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(write_artifact(tmp_path / "current.json")),
        now=lambda: validated_at,
    )
    stale_gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(write_artifact(tmp_path / "stale-names.json")),
        now=lambda: validated_at + timedelta(days=2),
    )

    assert current_gateway.available_metric_names() == (
        "jira_new_peu",
        "jira_new_mau",
        "confluence_new_peu",
        "confluence_new_mau",
    )
    assert stale_gateway.available_metric_names() == ()

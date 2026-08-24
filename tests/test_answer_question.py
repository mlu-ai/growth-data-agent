from fastapi.testclient import TestClient


def test_data_analyst_receives_typed_canonical_definition(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"] == {
        "name": "jira_new_peu",
        "definition": "A Product User's first-ever Paid Enablement for Jira.",
        "formula": (
            "count_distinct(product_user_id) where product = Jira "
            "and paid_enablement_ordinal = 1"
        ),
        "grain": "Product User in a Tenant and Jira product",
        "time_rule": "Attribute to first-ever Jira Paid Enablement.",
        "semantic_version": "1.0.0",
        "citation": {
            "authority": "dbt/MetricFlow",
            "artifact_path": "dbt/models/marts/jira_new_peu.yml#jira_new_peu",
            "metric_name": "jira_new_peu",
            "model_name": "fct_jira_new_peu",
        },
    }
    assert body["source_freshness"]["is_current"] is True
    assert body["effective_access_scope"]["regions"] == ["Americas", "APAC", "EMEA"]
    assert body["trace_id"]


def test_apac_manager_receives_only_apac_effective_scope(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "Define Jira New Paid Enabled User",
        },
    )

    assert response.status_code == 200
    assert response.json()["effective_access_scope"]["regions"] == ["APAC"]


def test_unknown_agent_user_is_refused(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "unknown", "question": "What is Jira New PEU?"},
    )

    assert response.status_code == 403

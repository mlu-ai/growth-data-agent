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
    body = response.json()
    assert body["effective_access_scope"]["regions"] == ["APAC"]
    assert body["semantic_query_evidence"] == {
        "metric_name": "jira_new_peu",
        "artifact_sha256": body["semantic_query_evidence"]["artifact_sha256"],
        "constrained_products": ["Jira"],
        "constrained_regions": ["APAC"],
        "tenant_scope": "APAC Tenants only",
        "result_row_count": 1,
    }


def test_body_only_agent_user_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "unknown", "question": "What is Jira New PEU?"},
    )

    assert response.status_code == 401


def test_data_analyst_receives_scoped_apac_evidence_hypothesis(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    assert body["driver_decomposition"]["decline"] == 560
    assert body["driver_decomposition"]["contributions"][0] == {
        "region": "APAC",
        "seat_tier": "51-200",
        "baseline_value": 800,
        "comparison_value": 380,
        "change": -420,
        "contribution_to_decline": 420,
        "percentage_of_decline": 75.0,
    }
    assert body["evidence"]["citations"][0] == {
        "document_id": "jira-apac-paid-provisioning-incident",
        "title": "Jira APAC paid provisioning incident",
        "affected_scope": {
            "product": "Jira",
            "region": "APAC",
            "tenant_scope": "APAC 51-200 Seat Tier Tenants",
        },
        "relevant_date": "2026-06-12",
        "freshness": "2026-06-13T00:00:00Z",
        "support_status": "supports",
            "support_explanation": (
                "The incident overlaps the APAC 51-200 Seat Tier Tenant scope and the June 2026 "
                "decline period."
            ),
            "source_document_id": "jira-apac-paid-provisioning-incident",
            "source_url": "https://evidence.local/synthetic/jira-apac-paid-provisioning-incident",
            "source_revision": "synthetic-v1",
            "chunk_id": "jira-apac-paid-provisioning-incident:chunk:0",
        }
    citation_ids = [citation["document_id"] for citation in body["evidence"]["citations"]]
    assert citation_ids[0] == "jira-apac-paid-provisioning-incident"
    assert citation_ids[1:] == []
    assert all(
        citation["document_id"] != "jira-apac-paid-provisioning-incident-restricted"
        for citation in body["evidence"]["citations"]
    )
    assert "does not establish causation" in body["answer"]
    assert body["effective_access_scope"]["regions"] == ["Americas", "APAC", "EMEA"]

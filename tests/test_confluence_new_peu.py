from fastapi.testclient import TestClient


def test_data_analyst_receives_confluence_canonical_definition(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Confluence New PEU?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"] == {
        "name": "confluence_new_peu",
        "definition": (
            "Confluence New PEU is a Product User's first-ever Paid Enablement "
            "for Confluence."
        ),
        "formula": "count_distinct(product_user_id)",
        "grain": "Product User in a Tenant and Confluence product",
        "time_rule": (
            "Attribute to the first-ever Confluence Paid Enablement; later restorations "
            "do not qualify again."
        ),
        "semantic_version": "1.0.0",
        "citation": {
            "authority": "dbt/MetricFlow",
            "artifact_path": "dbt/models/marts/confluence_new_peu.yml#confluence_new_peu",
            "metric_name": "confluence_new_peu",
            "model_name": "fct_confluence_new_peu",
        },
    }
    assert body["semantic_query_evidence"]["constrained_products"] == ["Confluence"]
    assert body["source_freshness"]["is_current"] is True
    assert "first-ever" in body["answer"]
    assert "restorations" in body["answer"]


def test_data_analyst_receives_reconciled_confluence_campaign_movement(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "Why did Confluence New PEU move from May to June after the Americas "
                "11–50 Seat Tier acquisition campaign?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    decomposition = body["driver_decomposition"]
    assert body["result_classification"] == "driver_decomposition"
    assert (decomposition["baseline_value"], decomposition["comparison_value"]) == (2400, 2820)
    assert (decomposition["net_change"], decomposition["reconciled_change"]) == (420, 420)
    assert decomposition["residual"] == 0
    assert decomposition["contributions"][0] == {
        "region": "Americas",
        "seat_tier": "11-50",
        "baseline_value": 1200,
        "comparison_value": 1620,
        "change": 420,
        "contribution_to_decline": 0,
        "percentage_of_decline": 0.0,
    }
    assert "observed" in body["answer"]
    assert "does not establish causation" in body["answer"]


def test_data_analyst_receives_scoped_confluence_campaign_evidence(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What evidence may explain the Americas 11–50-seat Confluence New PEU "
                "movement after the acquisition campaign?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    assert body["driver_decomposition"]["residual"] == 0
    citation = body["evidence"]["citations"][0]
    assert citation["document_id"] == "confluence-americas-acquisition-campaign"
    assert citation["affected_scope"] == {
        "product": "Confluence",
        "region": "Americas",
        "tenant_scope": "Americas 11-50 Seat Tier Tenants",
    }
    assert citation["relevant_date"] == "2026-06-15"
    assert citation["freshness"] == "2026-06-16T00:00:00Z"
    assert citation["support_status"] == "supports"
    assert body["evidence"]["support_status"] == "supports"
    assert "campaign" in body["evidence"]["support_explanation"]
    assert "increase" in body["evidence"]["support_explanation"]
    assert "incident" not in body["evidence"]["support_explanation"]
    assert "decline" not in body["evidence"]["support_explanation"]
    assert "does not establish that it caused" in body["answer"]
    assert body["source_freshness"]["is_current"] is True
    factor = body["candidate_causal_factor"]
    assert factor["category"] == "campaign"
    assert factor["factor_occurrence_time"] == "2026-06-15"
    assert factor["citation"]["source_document_id"] == "confluence-americas-acquisition-campaign"
    assert all(
        item["affected_scope"]["product"] == "Confluence"
        and item["affected_scope"]["region"] == "Americas"
        and item["affected_scope"]["tenant_scope"]
        == "Americas 11-50 Seat Tier Tenants"
        for item in body["evidence"]["citations"]
    )
    assert all(
        "restricted" not in item["document_id"]
        for item in body["evidence"]["citations"]
    )


def test_apac_manager_cannot_request_americas_confluence_campaign_evidence(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": (
                "What evidence may explain the Americas 11–50-seat Confluence New PEU "
                "movement after the acquisition campaign?"
            ),
        },
    )

    assert response.status_code == 403
    assert "Americas" in response.json()["detail"]

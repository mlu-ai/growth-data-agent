from fastapi.testclient import TestClient


def test_data_analyst_receives_reconciled_ranked_may_to_june_driver_decomposition(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Why did Jira New PEU fall from May to June?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    decomposition = body["driver_decomposition"]
    assert body["result_classification"] == "driver_decomposition"
    assert (decomposition["baseline_period"], decomposition["comparison_period"]) == (
        "2026-05",
        "2026-06",
    )
    assert (decomposition["baseline_value"], decomposition["comparison_value"]) == (4000, 3440)
    assert (decomposition["net_change"], decomposition["decline"]) == (-560, 560)
    assert decomposition["contributions"][0] == {
        "region": "APAC",
        "seat_tier": "51-200",
        "baseline_value": 800,
        "comparison_value": 380,
        "change": -420,
        "contribution_to_decline": 420,
        "percentage_of_decline": 75.0,
    }
    assert decomposition["reconciled_change"] == -560
    assert decomposition["residual"] == 0
    assert decomposition["approved_dimensions"] == ["Region", "Seat Tier"]
    assert "Driver Decomposition" in body["answer"]
    assert "leading observed driver" in body["answer"]
    assert "does not establish causation" in body["answer"]
    assert "Causal Estimate" not in body["answer"]
    assert body["canonical_definition"]["semantic_version"] == "1.0.0"
    assert body["canonical_definition"]["definition"] == (
        "A Product User's first-ever Paid Enablement for Jira."
    )


def test_apac_manager_receives_only_apac_decomposition_without_cross_region_values(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "Why did Jira New PEU fall from May to June?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    decomposition = body["driver_decomposition"]
    assert body["effective_access_scope"]["regions"] == ["APAC"]
    assert body["semantic_query_evidence"]["constrained_regions"] == ["APAC"]
    assert (decomposition["baseline_value"], decomposition["comparison_value"]) == (1400, 960)
    assert all(item["region"] == "APAC" for item in decomposition["contributions"])
    assert "Americas" not in body["answer"]
    assert "EMEA" not in body["answer"]

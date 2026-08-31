from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_data_analyst_receives_canonical_confluence_new_mau_definition(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Confluence New MAU?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    definition = body["canonical_definition"]
    assert definition["name"] == "confluence_new_mau"
    assert "New PEU" in definition["definition"]
    assert "same calendar month" in definition["definition"]
    assert "same-product Visit" in definition["time_rule"]
    assert definition["citation"]["model_name"] == "fct_confluence_new_mau"
    assert body["semantic_query_evidence"]["constrained_products"] == ["Confluence"]
    assert "same calendar month" in body["answer"]


def test_new_mau_models_require_same_product_and_calendar_month() -> None:
    repository = Path(__file__).resolve().parents[1]
    for product in ("jira", "confluence"):
        model = (repository / f"dbt/models/marts/fct_{product}_new_mau.sql").read_text()
        product_name = product.title()
        assert f"visits.product = '{product_name}'" in model
        assert "date_trunc('month', visits.visited_at)" in model
        assert "date_trunc('month', new_peu.paid_enabled_at)" in model


def test_data_analyst_receives_emea_confluence_new_mau_regression_hypothesis(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What evidence may explain the Confluence EMEA 51–200-seat New MAU "
                "decline after the onboarding-email regression?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    decomposition = body["driver_decomposition"]
    assert (decomposition["baseline_value"], decomposition["comparison_value"]) == (698, 422)
    assert decomposition["reconciled_change"] == decomposition["net_change"] == -276
    assert decomposition["residual"] == 0
    assert decomposition["contributions"][0]["region"] == "EMEA"
    assert decomposition["contributions"][0]["seat_tier"] == "51-200"
    assert decomposition["contributions"][0]["contribution_to_decline"] == 300
    assert body["evidence"]["citations"][0]["document_id"] == (
        "confluence-emea-onboarding-email-regression"
    )
    assert "Hypothesis" in body["answer"]
    assert "onboarding-email regression" in body["answer"]
    assert "does not establish that it caused" in body["answer"]
    factors = body["candidate_causal_factors"]
    assert len(factors) == 1
    factor = factors[0]
    assert factor["category"] == "onboarding"
    assert factor["status"] == "supported"
    assert factor["factor_occurrence_time"] == "2026-06-20"
    assert [c["source_document_id"] for c in factor["citations"]] == [
        "confluence-emea-onboarding-email-regression"
    ]


def test_apac_regional_manager_receives_only_apac_jira_new_mau_rows(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "Why did Jira New MAU fall from May to June?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "driver_decomposition"
    assert body["semantic_query_evidence"]["constrained_regions"] == ["APAC"]
    assert all(item["region"] == "APAC" for item in body["driver_decomposition"]["contributions"])
    assert "EMEA" not in response.text


def test_apac_regional_manager_receives_only_apac_confluence_new_mau_rows(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "Why did Confluence New MAU change from May to June?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "driver_decomposition"
    assert body["semantic_query_evidence"]["constrained_regions"] == ["APAC"]
    assert all(
        item["region"] == "APAC" for item in body["driver_decomposition"]["contributions"]
    )
    assert "EMEA" not in response.text


def test_apac_regional_manager_cannot_request_emea_confluence_new_mau_evidence(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": (
                "What evidence may explain the Confluence EMEA 51–200-seat New MAU decline "
                "after the onboarding-email regression?"
            ),
        },
    )

    assert response.status_code == 403
    assert "EMEA" in response.json()["detail"]

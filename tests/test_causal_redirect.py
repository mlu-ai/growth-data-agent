from __future__ import annotations

from fastapi.testclient import TestClient


def test_causal_phrased_jira_new_mau_question_is_redirected(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What is the causal estimate for the registered Jira New MAU "
                "onboarding treatment/control experiment?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "limitation"
    assert "Causal Estimate" in body["answer"]
    assert "no longer produces" in body["answer"]
    assert any("retired" in caveat for caveat in body["caveats"])
    assert "causal_registration" not in body
    assert "causal_estimate" not in body
    assert "descriptive_comparison" not in body
    assert "causal_analysis_plan" not in body


def test_causal_redirect_does_not_run_a_causal_estimator_for_any_phrasing_variant(
    client: TestClient,
) -> None:
    causal_variant_questions = [
        "What is the estimate for an unregistered Jira New MAU design?",
        "Estimate the causal effect of an observational Jira New MAU design.",
        "Compare Jira New MAU for all users before and after the rollout causally.",
        (
            "What is the causal estimate for the registered Jira New MAU onboarding "
            "treatment/control experiment with missing review?"
        ),
    ]

    for question in causal_variant_questions:
        response = client.post(
            "/answer_question",
            json={"agent_user_id": "data_analyst", "question": question},
        )

        assert response.status_code == 200, question
        body = response.json()
        assert body["result_classification"] == "limitation", question
        assert "causal_estimate" not in body, question
        assert "descriptive_comparison" not in body, question
        assert "causal_analysis_plan" not in body, question


def test_causal_redirect_does_not_vary_by_access_profile(client: TestClient) -> None:
    """There is no scoped registration left to leak across Access Profiles."""
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": (
                "What is the causal estimate for the registered Jira New MAU "
                "onboarding treatment/control experiment?"
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"


def test_confluence_metric_question_still_resolves_to_canonical_definition(
    client: TestClient,
) -> None:
    """A Confluence question must not be hijacked by causal-phrase detection."""
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What is Confluence New MAU?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"]["name"] == "confluence_new_mau"

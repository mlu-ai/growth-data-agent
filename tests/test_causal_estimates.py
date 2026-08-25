from __future__ import annotations

from fastapi.testclient import TestClient


def test_reviewed_registered_jira_new_mau_experiment_returns_causal_estimate(
    client: TestClient,
) -> None:
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
    assert body["result_classification"] == "causal_estimate"

    registration = body["causal_registration"]
    assert registration["treatment"] == "onboarding_email_v2"
    assert registration["control"] == "no_onboarding_email"
    assert registration["outcome"] == "jira_new_mau"
    assert registration["tenant_scope"] == "Americas 1-10 Seat Tier Tenants"
    assert registration["seat_tier"] == "1-10"
    assert registration["support_checks"]
    assert all(check["passed"] for check in registration["support_checks"])
    assert registration["estimator_approval"]["estimator"] == "difference_in_means"
    assert registration["estimator_approval"]["approved"] is True
    assert registration["diagnostics"]
    assert all(diagnostic["passed"] for diagnostic in registration["diagnostics"])
    assert registration["review"]["status"] == "approved"

    estimate = body["causal_estimate"]
    assert estimate["estimator"] == "difference_in_means"
    assert estimate["estimate"] == 0.06
    assert estimate["assumptions"]
    assert estimate["diagnostics"] == registration["diagnostics"]
    assert body["semantic_query_evidence"]["metric_name"] == "jira_new_mau"
    assert body["semantic_query_evidence"]["constrained_regions"] == ["Americas"]
    assert "Causal Estimate" in body["answer"]


def test_failed_support_check_returns_descriptive_result_without_estimate(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What is the causal estimate for the registered Jira New MAU "
                "onboarding treatment/control experiment with a support check that did not pass?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "descriptive_result"
    assert body["causal_estimate"] is None
    assert any(not check["passed"] for check in body["causal_registration"]["support_checks"])
    assert "Descriptive result only" in body["answer"]
    assert "Causal Estimate" in body["answer"]


def test_unregistered_design_returns_reviewable_analysis_plan(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What is the estimate for an unregistered Jira New MAU design?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "analysis_plan"
    assert body["causal_registration"] is None
    assert body["causal_estimate"] is None
    assert body["causal_analysis_plan"]["experiment_id"] == "unregistered-jira-new-mau-design"
    assert body["causal_analysis_plan"]["required_actions"]
    assert "unregistered" in body["causal_analysis_plan"]["reason"]
    assert "Reviewable analysis plan" in body["answer"]


def test_missing_review_returns_analysis_plan_without_estimate(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": (
                "What is the causal estimate for the registered Jira New MAU "
                "onboarding treatment/control experiment with missing review?"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "analysis_plan"
    assert body["causal_estimate"] is None
    assert body["causal_registration"]["review"]["status"] == "pending"
    assert "human review" in body["causal_analysis_plan"]["reason"]
    assert "Reviewable analysis plan" in body["answer"]


def test_all_user_pre_post_comparison_is_explicitly_descriptive(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Compare Jira New MAU for all users before and after the rollout causally.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "descriptive_result"
    assert body["causal_estimate"] is None
    assert body["causal_registration"]["design_type"] == "all_user_pre_post"
    assert body["descriptive_comparison"]["difference"] == 0.06
    assert "All-user pre/post comparisons are descriptive" in body["causal_analysis_plan"]["reason"]
    assert "descriptive" in body["answer"].casefold()
    assert "Causal Estimate" in body["answer"]


def test_regional_profile_cannot_receive_out_of_scope_causal_estimate(
    client: TestClient,
) -> None:
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

    assert response.status_code == 403
    assert "Americas" in response.json()["detail"]


def test_observational_design_returns_reviewable_plan(client: TestClient) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Estimate the causal effect of an observational Jira New MAU design.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "analysis_plan"
    assert body["causal_estimate"] is None
    assert body["causal_registration"]["design_type"] == "observational"
    assert "analysis plan" in body["causal_analysis_plan"]["reason"]


def test_question_design_variant_cannot_be_overridden_by_passing_experiment_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "experiment_id": "jira-new-mau-onboarding-experiment",
            "question": "Estimate an observational Jira New MAU design.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "analysis_plan"
    assert body["causal_registration"]["design_type"] == "observational"
    assert body["causal_estimate"] is None


def test_jira_experiment_id_does_not_hijack_confluence_metric_question(
    client: TestClient,
) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "experiment_id": "jira-new-mau-onboarding-experiment",
            "question": "What is Confluence New MAU?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"]["name"] == "confluence_new_mau"
    assert body["causal_estimate"] is None

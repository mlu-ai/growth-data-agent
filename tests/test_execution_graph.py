from __future__ import annotations

import pytest
from pydantic import ValidationError

from growth_data_agent.contracts import AnalyticalIntent, AnalyticalRoute, AnswerQuestionRequest
from growth_data_agent.execution import ExecutionGraph, RuleBasedIntentInterpreter
from growth_data_agent.policy import UnknownAgentUserError
from growth_data_agent.service import AnswerQuestionService


def test_canonical_definition_runs_through_validated_intent(client) -> None:
    response = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "Define Jira New PEU"},
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "canonical_definition"


def test_unknown_user_is_denied_before_question_interpretation() -> None:
    interpreter_called = False

    class Interpreter:
        def interpret(self, request):
            nonlocal interpreter_called
            interpreter_called = True
            return AnalyticalIntent(
                route=AnalyticalRoute.CANONICAL_DEFINITION,
                metric_name="jira_new_peu",
            )

    graph = ExecutionGraph(
        intent_interpreter=Interpreter(),
        canonical_definition_handler=lambda *_: None,
        driver_decomposition_handler=lambda *_: None,
        causal_analysis_handler=lambda _: None,
        catalog_ownership_handler=lambda _: None,
        direct_identifier_handler=lambda _: None,
        limitation_handler=lambda _: None,
        metric_definition_gap_handler=lambda *_: None,
        legacy_handler=lambda _: None,
        clarification_handler=lambda _: None,
    )

    with pytest.raises(UnknownAgentUserError):
        graph.answer_question(
            AnswerQuestionRequest(
                agent_user_id="unknown", question="Define Jira New PEU"
            )
        )

    assert interpreter_called is False


def test_canonical_intent_requires_a_metric_name() -> None:
    with pytest.raises(ValidationError):
        AnalyticalIntent(route=AnalyticalRoute.CANONICAL_DEFINITION)


def test_driver_intent_requires_a_metric_name() -> None:
    with pytest.raises(ValidationError):
        AnalyticalIntent(route=AnalyticalRoute.DRIVER_DECOMPOSITION)


def test_blank_requested_metric_returns_a_governed_limitation(client) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Define a metric",
            "requested_metric_name": " ",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"
    assert "metric" in response.json()["answer"].casefold()


def test_malformed_interpreter_output_uses_the_clarification_handler() -> None:
    clarification_called = False

    class MalformedInterpreter:
        def interpret(self, request):
            return {"route": "not-a-route"}

    graph = ExecutionGraph(
        intent_interpreter=MalformedInterpreter(),
        canonical_definition_handler=lambda *_: pytest.fail("canonical handler must not run"),
        driver_decomposition_handler=lambda *_: pytest.fail("driver handler must not run"),
        causal_analysis_handler=lambda _: pytest.fail("causal handler must not run"),
        catalog_ownership_handler=lambda _: pytest.fail("catalog handler must not run"),
        direct_identifier_handler=lambda _: pytest.fail("identifier handler must not run"),
        limitation_handler=lambda _: pytest.fail("limitation handler must not run"),
        metric_definition_gap_handler=lambda *_: pytest.fail("gap handler must not run"),
        legacy_handler=lambda _: pytest.fail("legacy handler must not run"),
        clarification_handler=lambda _: _mark_clarification(),
    )

    def _mark_clarification():
        nonlocal clarification_called
        clarification_called = True
        return None

    graph.answer_question(
        AnswerQuestionRequest(agent_user_id="data_analyst", question="Define Jira New PEU")
    )

    assert clarification_called is True


def test_constructed_invalid_intent_uses_the_clarification_handler() -> None:
    clarification_called = False

    class ConstructedInvalidInterpreter:
        def interpret(self, request):
            return AnalyticalIntent.model_construct(route="not-a-route", metric_name=None)

    graph = ExecutionGraph(
        intent_interpreter=ConstructedInvalidInterpreter(),
        canonical_definition_handler=lambda *_: pytest.fail("canonical handler must not run"),
        driver_decomposition_handler=lambda *_: pytest.fail("driver handler must not run"),
        causal_analysis_handler=lambda _: pytest.fail("causal handler must not run"),
        catalog_ownership_handler=lambda _: pytest.fail("catalog handler must not run"),
        direct_identifier_handler=lambda _: pytest.fail("identifier handler must not run"),
        limitation_handler=lambda _: pytest.fail("limitation handler must not run"),
        metric_definition_gap_handler=lambda *_: pytest.fail("gap handler must not run"),
        legacy_handler=lambda _: pytest.fail("legacy handler must not run"),
        clarification_handler=lambda _: _mark_clarification(),
    )

    def _mark_clarification():
        nonlocal clarification_called
        clarification_called = True
        return None

    graph.answer_question(
        AnswerQuestionRequest(agent_user_id="data_analyst", question="Define Jira New PEU")
    )

    assert clarification_called is True


def test_unhandled_specialist_phrase_keeps_unknown_metric_on_gap_route() -> None:
    interpreter = RuleBasedIntentInterpreter(
        metric_name_resolver=AnswerQuestionService._requested_metric_name,
        route_resolver=AnswerQuestionService._route_for_intent,
    )

    intent = interpreter.interpret(
        AnswerQuestionRequest(
            agent_user_id="data_analyst",
            question="Why did the metric fall from May to June?",
            requested_metric_name="new_trials",
        )
    )

    assert intent.route is AnalyticalRoute.METRIC_DEFINITION_GAP


def test_driver_and_causal_requests_have_explicit_specialist_routes() -> None:
    driver_request = AnswerQuestionRequest(
        agent_user_id="data_analyst",
        question="Why did Jira New PEU fall from May to June?",
    )
    causal_request = AnswerQuestionRequest(
        agent_user_id="data_analyst",
        question="Estimate the causal effect of the Jira New MAU experiment.",
    )
    explicit_experiment_request = AnswerQuestionRequest(
        agent_user_id="data_analyst",
        question="Estimate the causal effect.",
        experiment_id="jira-new-mau-onboarding-experiment",
    )

    assert (
        AnswerQuestionService._route_for_intent(
            driver_request, AnswerQuestionService._requested_metric_name(driver_request)
        )
        is AnalyticalRoute.DRIVER_DECOMPOSITION
    )
    assert (
        AnswerQuestionService._route_for_intent(
            causal_request, AnswerQuestionService._requested_metric_name(causal_request)
        )
        is AnalyticalRoute.CAUSAL_ANALYSIS
    )
    assert (
        AnswerQuestionService._route_for_intent(
            explicit_experiment_request,
            AnswerQuestionService._requested_metric_name(explicit_experiment_request),
        )
        is AnalyticalRoute.CAUSAL_ANALYSIS
    )


def test_catalog_identifier_and_unsupported_requests_have_explicit_routes() -> None:
    catalog_request = AnswerQuestionRequest(
        agent_user_id="data_analyst", question="Who owns the Jira New PEU metric?"
    )
    identifier_request = AnswerQuestionRequest(
        agent_user_id="data_analyst", question="List affected tenant IDs"
    )
    unsupported_request = AnswerQuestionRequest(
        agent_user_id="data_analyst", question="Tell me a joke"
    )

    assert (
        AnswerQuestionService._route_for_intent(
            catalog_request, AnswerQuestionService._requested_metric_name(catalog_request)
        )
        is AnalyticalRoute.CATALOG_OWNERSHIP
    )
    assert (
        AnswerQuestionService._route_for_intent(
            identifier_request, AnswerQuestionService._requested_metric_name(identifier_request)
        )
        is AnalyticalRoute.DIRECT_IDENTIFIER
    )
    assert (
        AnswerQuestionService._route_for_intent(
            unsupported_request, AnswerQuestionService._requested_metric_name(unsupported_request)
        )
        is AnalyticalRoute.LIMITATION
    )


def test_unknown_metric_uses_the_metric_definition_gap_route() -> None:
    request = AnswerQuestionRequest(
        agent_user_id="data_analyst",
        question="Define New Trials",
        requested_metric_name="new_trials",
    )

    assert (
        AnswerQuestionService._route_for_intent(
            request, AnswerQuestionService._requested_metric_name(request)
        )
        is AnalyticalRoute.METRIC_DEFINITION_GAP
    )


def test_route_for_intent_accepts_a_metric_declared_by_the_semantic_artifact() -> None:
    request = AnswerQuestionRequest(
        agent_user_id="data_analyst", question="How is the custom metric defined?"
    )

    assert (
        AnswerQuestionService._route_for_intent(
            request,
            "jira_custom_metric",
            canonical_metric_names=("jira_custom_metric",),
        )
        is AnalyticalRoute.CANONICAL_DEFINITION
    )


def test_product_scoped_user_cannot_request_an_unknown_metric_for_another_product(client) -> None:
    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "confluence_product_manager",
            "question": "Define Jira New Trials",
            "requested_metric_name": "jira_new_trials",
        },
    )

    assert response.status_code == 403

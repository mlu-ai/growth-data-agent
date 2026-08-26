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


def test_unhandled_specialist_phrase_keeps_its_metric_on_the_canonical_route() -> None:
    interpreter = RuleBasedIntentInterpreter(
        metric_name_resolver=AnswerQuestionService._requested_metric_name,
        is_canonical_definition_request=AnswerQuestionService._is_canonical_definition_request,
    )

    intent = interpreter.interpret(
        AnswerQuestionRequest(
            agent_user_id="data_analyst",
            question="Why did the metric fall from May to June?",
            requested_metric_name="new_trials",
        )
    )

    assert intent.route is AnalyticalRoute.CANONICAL_DEFINITION

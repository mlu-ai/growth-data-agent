from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pytest

from growth_data_agent.adversarial_evaluation import (
    AdversarialDimension,
    AdversarialObservation,
    PromptfooCase,
    PromptfooMatrixStore,
    evaluate_adversarial_case,
    run_promptfoo_matrix,
)
from growth_data_agent.evaluation_dataset import EvaluationSplit
from growth_data_agent.principal import development_token_environment_variable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MATRIX_PATH = _REPOSITORY_ROOT / "evaluations/promptfoo/matrix.json"
_PROMPTFOO_CONFIG = _REPOSITORY_ROOT / "evaluations/promptfoo/promptfooconfig.yaml"
_PROMPTFOO_TESTS = _REPOSITORY_ROOT / "evaluations/promptfoo/tests.py"
_PROMPTFOO_PROVIDER = _REPOSITORY_ROOT / "evaluations/promptfoo/provider.py"


def _observation(
    *,
    status_code: int = 200,
    classification: str = "safe_refusal",
    regions: tuple[str, ...] = ("APAC",),
    tools: tuple[str, ...] = (),
) -> AdversarialObservation:
    return AdversarialObservation(
        status_code=status_code,
        response={
            "answer": "This request cannot be fulfilled within the governed scope.",
            "result_classification": classification,
            "effective_access_scope": {"regions": list(regions)},
            "evidence": None,
            "candidate_causal_factors": None,
            "direct_identifier_answer": None,
        },
        tool_names=tools,
    )


def test_promptfoo_matrix_covers_all_adversarial_dimensions_and_splits() -> None:
    matrix = PromptfooMatrixStore(_MATRIX_PATH).load()

    assert {case.dimension for case in matrix.cases} == set(AdversarialDimension)
    assert {case.split for case in matrix.cases} == set(EvaluationSplit)
    assert len(matrix.cases) >= 8
    assert all(case.allowed_tools == () for case in matrix.cases)


def test_valid_prompt_injection_is_denied_without_tool_or_scope_expansion() -> None:
    case = PromptfooMatrixStore(_MATRIX_PATH).load().cases[0]
    result = evaluate_adversarial_case(case, _observation())

    assert result.passed


def test_adversarial_evaluation_rejects_permission_and_tool_expansion() -> None:
    matrix = PromptfooMatrixStore(_MATRIX_PATH).load()
    case = next(
        case for case in matrix.cases if case.dimension is AdversarialDimension.PERMISSION_EXPANSION
    )
    result = evaluate_adversarial_case(
        case,
        _observation(
            classification="canonical_definition",
            regions=("APAC", "EMEA"),
            tools=("sql",),
        ),
    )

    assert not result.passed
    assert any(
        "region" in finding.detail or "tool" in finding.detail
        for finding in result.findings
    )


def test_adversarial_evaluation_rejects_an_unsuccessful_target_response() -> None:
    case = PromptfooMatrixStore(_MATRIX_PATH).load().cases[0]

    result = evaluate_adversarial_case(case, _observation(status_code=503))

    assert not result.passed
    assert any(finding.check == "governed_response_status" for finding in result.findings)


def test_promptfoo_evaluation_reports_adversarial_results_separately() -> None:
    matrix = PromptfooMatrixStore(_MATRIX_PATH).load()

    scorecard = run_promptfoo_matrix(
        matrix,
        lambda case: _observation(classification=case.expected_result_classification),
    )

    assert scorecard.name == "adversarial"
    assert scorecard.total == len(matrix.cases)
    assert scorecard.failed == 0


def test_promptfoo_matrix_rejects_missing_dimension(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(
        '{"artifact_type":"promptfoo_adversarial_matrix","dataset_version":"1.0.0",'
        '"published_at":"2026-09-01","cases":[]}'
    )

    with pytest.raises(ValueError):
        PromptfooMatrixStore(path).load()


def test_promptfoo_matrix_rejects_case_without_explicit_tool_boundary() -> None:
    with pytest.raises(ValueError, match="allowed_tools"):
        PromptfooCase.model_validate(
            {
                "case_id": "case-1",
                "dimension": "prompt_injection",
                "split": "development",
                "agent_user_id": "data_analyst",
                "question": "no tools",
                "expected_result_classification": "safe_refusal",
                "allowed_regions": ["APAC"],
            }
        )


def test_promptfoo_config_runs_the_checked_in_matrix_provider_and_assertions() -> None:
    config = _PROMPTFOO_CONFIG.read_text()

    assert "file://provider.py" in config
    assert "file://tests.py:generate_tests" in config
    assert "PROMPTFOO_TARGET_URL" in config


def test_promptfoo_generates_one_executable_test_per_versioned_matrix_case() -> None:
    matrix = PromptfooMatrixStore(_MATRIX_PATH).load()
    module = runpy.run_path(str(_PROMPTFOO_TESTS))
    generated = module["generate_tests"]()

    assert [test["description"] for test in generated] == [case.case_id for case in matrix.cases]
    assert all(test["vars"]["question"] for test in generated)


def test_promptfoo_provider_consumes_only_the_safe_evaluation_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path(str(_PROMPTFOO_PROVIDER))
    monkeypatch.setenv("GROWTH_DATA_AGENT_DEV_TOKEN_DATA_ANALYST", "test-token")
    monkeypatch.setenv("GROWTH_DATA_AGENT_EVALUATION_TOKEN", "evaluation-test-token")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "response": {
                        "has_answer": True,
                        "result_classification": "safe_refusal",
                        "trace_id": "trace-1",
                        "effective_access_scope": {"regions": ["APAC"]},
                        "source_freshness": {"is_current": True},
                        "has_evidence": False,
                        "evidence_regions": [],
                        "has_candidate_causal_factors": False,
                        "has_direct_identifier_answer": False,
                    },
                    "executed_tools": [{"name": "evidence_retrieval", "status": "success"}],
                }
            ).encode()

    monkeypatch.setitem(
        module["call_api"].__globals__,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    result = module["call_api"](
        "adversarial request",
        {"config": {"url": "http://private-service"}},
        {"vars": {"agent_user_id": "data_analyst"}},
    )
    output = json.loads(result["output"])

    assert output == {
        "status_code": 200,
        "response": {
            "has_answer": True,
            "result_classification": "safe_refusal",
            "trace_id": "trace-1",
            "effective_access_scope": {"regions": ["APAC"]},
            "source_freshness": {"is_current": True},
            "has_evidence": False,
            "evidence_regions": [],
            "has_candidate_causal_factors": False,
            "has_direct_identifier_answer": False,
        },
        "selected_tools": ["evidence_retrieval"],
    }


def test_evaluation_endpoint_requires_a_separate_evaluator_capability(client) -> None:
    token = os.environ[development_token_environment_variable("data_analyst")]

    response = client.post(
        "/evaluation/answer_question",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "agent_user_id": "data_analyst",
            "question": "What is the definition of Jira New PEU?",
        },
    )

    assert response.status_code == 403


def test_evaluation_endpoint_rejects_a_mismatched_evaluator_capability(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = os.environ[development_token_environment_variable("data_analyst")]
    monkeypatch.setenv("GROWTH_DATA_AGENT_EVALUATION_TOKEN", "expected-evaluator-token")

    response = client.post(
        "/evaluation/answer_question",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Evaluation-Token": "wrong-evaluator-token",
        },
        json={
            "agent_user_id": "data_analyst",
            "question": "What is the definition of Jira New PEU?",
        },
    )

    assert response.status_code == 403


def test_promptfoo_assertion_fails_closed_when_the_projection_has_unknown_regions() -> None:
    generated = runpy.run_path(str(_PROMPTFOO_TESTS))["generate_tests"]()

    assert all(
        "!response.unknown_region_observed" in test["assert"][0]["value"]
        for test in generated
    )


def test_evaluation_endpoint_returns_only_safe_response_and_execution_metadata(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = os.environ[development_token_environment_variable("data_analyst")]
    monkeypatch.setenv("GROWTH_DATA_AGENT_EVALUATION_TOKEN", "evaluation-test-token")

    response = client.post(
        "/evaluation/answer_question",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Evaluation-Token": "evaluation-test-token",
        },
        json={
            "agent_user_id": "data_analyst",
            "question": "What is the definition of Jira New PEU?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"response", "executed_tools"}
    assert "answer" not in body["response"]
    assert "evidence" not in body["response"]
    assert "lead_agent_metadata" not in body["response"]
    assert body["executed_tools"] == [
        {"name": "semantic_definition", "status": "success"},
        {"name": "semantic_query", "status": "success"},
    ]


def test_adversarial_evaluation_fails_closed_when_a_unknown_region_was_observed() -> None:
    case = PromptfooMatrixStore(_MATRIX_PATH).load().cases[0]
    result = evaluate_adversarial_case(
        case,
        AdversarialObservation(
            status_code=200,
            response={
                "result_classification": "safe_refusal",
                "effective_access_scope": {"regions": []},
                "unknown_region_observed": True,
            },
        ),
    )

    assert not result.passed
    assert any(finding.check == "permission_boundary" for finding in result.findings)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from growth_data_agent.evaluation_dataset import EvaluationSplit
from growth_data_agent.observability import TraceRecord, TraceSpan
from growth_data_agent.trajectory_evaluation import (
    DeepEvalCaseDimension,
    DeepEvalDatasetStore,
    ToolCallObservation,
    TrajectoryExpectation,
    TrajectoryObservation,
    TrajectoryStage,
    evaluate_deepeval_tool_correctness,
    evaluate_trajectory,
    evaluate_turn_sequence,
    observation_from_trace,
    run_deepeval_dataset,
    trajectory_scorecard_passed,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/deepeval/v1/cases.json"


def _observation(
    *,
    trace_id: str = "trace-1",
    conversation_id: str | None = "conversation-1",
    response: dict | None = None,
    selected_tools: tuple[str, ...] = (),
    tool_calls=(),
) -> TrajectoryObservation:
    return TrajectoryObservation(
        trace_id=trace_id,
        conversation_id=conversation_id,
        response=response
        or {
            "answer": "A governed answer.",
            "result_classification": "canonical_definition",
            "trace_id": trace_id,
            "effective_access_scope": {"products": ["Jira"], "regions": ["APAC"]},
            "source_freshness": {"is_current": True},
            "has_active_investigation_selection": False,
        },
        selected_tools=selected_tools,
        tool_calls=tool_calls,
    )


def test_deepeval_dataset_covers_required_turn_dimensions() -> None:
    dataset = DeepEvalDatasetStore(_DATASET_PATH).load()

    assert dataset.dataset_version == "1.0.0"
    represented = {dimension for case in dataset.cases for dimension in case.dimensions}
    assert represented == set(DeepEvalCaseDimension)
    assert {case.split for case in dataset.cases} == set(EvaluationSplit)
    assert all(case.source_case_id for case in dataset.cases)


def test_deepeval_dataset_declares_tool_arguments_and_state_boundaries() -> None:
    dataset = DeepEvalDatasetStore(_DATASET_PATH).load()
    summary_boundary = next(
        case for case in dataset.cases if "summary-boundary" in case.case_id
    )
    reauthorization = next(
        case for case in dataset.cases if "reauthorization" in case.case_id
    )

    assert summary_boundary.summary_boundary_turn == 3
    assert reauthorization.reauthorization_turns == (2, 3)
    assert reauthorization.expected_tool_argument_keys[0] == {
        "semantic_driver_decomposition": ("metric_name",),
        "lightrag_retrieval": ("result_limit",),
        "evidence_retrieval": ("result_limit",),
    }


def test_valid_trajectory_measures_all_six_stages() -> None:
    result = evaluate_trajectory(
        _observation(),
        TrajectoryExpectation(
            expected_result_classification="canonical_definition",
            expected_tool_selection=(),
            allowed_tools=(),
        ),
    )

    assert result.passed
    assert {finding.stage for finding in result.findings} == set(TrajectoryStage)


def test_deepeval_tool_correctness_evaluates_observed_tools() -> None:
    assert evaluate_deepeval_tool_correctness(
        observed_tools=("metricflow",), expected_tools=("metricflow",)
    )
    assert not evaluate_deepeval_tool_correctness(
        observed_tools=("sql",), expected_tools=("metricflow",)
    )


def test_denied_trajectory_is_valid_when_it_has_no_tool_or_output_leak() -> None:
    result = evaluate_trajectory(
        _observation(
            response={
                "answer": "This request cannot be fulfilled.",
                "result_classification": "safe_refusal",
                "trace_id": "trace-denied",
                "effective_access_scope": {"products": ["Jira"], "regions": ["APAC"]},
                "source_freshness": {"is_current": False},
            },
            trace_id="trace-denied",
            selected_tools=(),
        ),
        TrajectoryExpectation(
            expected_result_classification="safe_refusal",
            denied=True,
        ),
    )

    assert result.passed


def test_stale_trajectory_requires_limitation_and_current_evidence_is_not_reused() -> None:
    result = evaluate_trajectory(
        _observation(
            response={
                "answer": "The semantic artifact is stale.",
                "result_classification": "limitation",
                "trace_id": "trace-stale",
                "effective_access_scope": {"products": ["Jira"], "regions": ["APAC"]},
                "source_freshness": {"is_current": False},
            },
            trace_id="trace-stale",
        ),
        TrajectoryExpectation(
            expected_result_classification="limitation",
            expected_freshness="stale",
        ),
    )

    assert result.passed


def test_malformed_trajectory_fails_the_argument_and_execution_stages() -> None:
    result = evaluate_trajectory(
        _observation(
            selected_tools=("sql",),
            tool_calls=(
                {"name": "sql", "status": "unknown", "argument_keys": ("raw_sql",)},
            ),
        ),
        TrajectoryExpectation(
            expected_result_classification="canonical_definition",
            allowed_tools=("metricflow",),
        ),
    )

    assert not result.passed
    failed_stages = {finding.stage for finding in result.findings if not finding.passed}
    assert TrajectoryStage.TOOL_SELECTION in failed_stages
    assert TrajectoryStage.TOOL_ARGUMENTS in failed_stages
    assert TrajectoryStage.TOOL_EXECUTION in failed_stages


def test_selected_action_requires_a_matching_observed_execution_span() -> None:
    result = evaluate_trajectory(
        _observation(
            response={
                "answer": "A governed hypothesis.",
                "result_classification": "hypothesis",
                "trace_id": "trace-unexecuted",
                "effective_access_scope": {"regions": ["APAC"]},
                "source_freshness": {"is_current": True},
            },
            trace_id="trace-unexecuted",
            selected_tools=("metricflow",),
        ),
        TrajectoryExpectation(
            expected_result_classification="hypothesis",
            expected_tool_selection=("metricflow",),
            allowed_tools=("metricflow",),
            expected_tool_argument_keys={"semantic_driver_decomposition": ("metric_name",)},
        ),
    )

    execution = next(
        finding for finding in result.findings if finding.stage is TrajectoryStage.TOOL_EXECUTION
    )
    assert not execution.passed
    assert "metricflow" in execution.detail


def test_trajectory_rejects_an_executed_span_outside_the_selected_action_contract() -> None:
    result = evaluate_trajectory(
        _observation(
            response={
                "answer": "A governed hypothesis.",
                "result_classification": "hypothesis",
                "trace_id": "trace-extra-span",
                "effective_access_scope": {"regions": ["APAC"]},
                "source_freshness": {"is_current": True},
            },
            trace_id="trace-extra-span",
            selected_tools=("metricflow",),
            tool_calls=(
                ToolCallObservation(
                    name="semantic_driver_decomposition",
                    status="success",
                    argument_keys=("metric_name",),
                ),
                ToolCallObservation(name="unbounded_sql", status="success"),
            ),
        ),
        TrajectoryExpectation(
            expected_result_classification="hypothesis",
            expected_tool_selection=("metricflow",),
            allowed_tools=("metricflow",),
            expected_tool_argument_keys={"semantic_driver_decomposition": ("metric_name",)},
        ),
    )

    execution = next(
        finding for finding in result.findings if finding.stage is TrajectoryStage.TOOL_EXECUTION
    )
    assert not execution.passed
    assert "unbounded_sql" in execution.detail


def test_multi_turn_measure_keeps_continuity_separate_from_trajectory() -> None:
    result = evaluate_turn_sequence(
        [
            _observation(trace_id="trace-1"),
            _observation(trace_id="trace-2"),
        ]
    )

    assert result.passed
    assert result.name == "multi_turn"


def test_multi_turn_measure_rejects_cached_trace_or_changed_conversation() -> None:
    result = evaluate_turn_sequence(
        [
            _observation(trace_id="trace-1"),
            _observation(trace_id="trace-1", conversation_id="conversation-2"),
        ]
    )

    assert not result.passed
    assert any(
        "trace" in finding.detail or "conversation" in finding.detail
        for finding in result.details
    )


def test_multi_turn_measure_rejects_a_summary_boundary_with_an_active_selection() -> None:
    result = evaluate_turn_sequence(
        [
            _observation(trace_id="trace-1"),
            _observation(trace_id="trace-2"),
            _observation(
                trace_id="trace-3",
                response={
                    "answer": "A definition.",
                    "result_classification": "canonical_definition",
                    "trace_id": "trace-3",
                    "has_active_investigation_selection": True,
                },
            ),
        ],
        summary_boundary_turn=3,
    )

    assert not result.passed
    assert any(
        finding.name == "summary_boundary_clears_active_selection" for finding in result.details
    )


def test_incomplete_multiturn_case_fails_the_separate_multiturn_scorecard() -> None:
    dataset = DeepEvalDatasetStore(_DATASET_PATH).load()
    case = next(case for case in dataset.cases if case.expected_turn_count == 2)
    scorecard = run_deepeval_dataset(
        dataset.model_copy(update={"cases": [case]}),
        lambda _case: [_observation(trace_id="incomplete-trace")],
    )

    assert scorecard.trajectory.failed == 1
    assert scorecard.multi_turn.failed == 1


def test_observation_from_trace_projects_only_allowlisted_evaluation_metadata() -> None:
    trace = TraceRecord(
        trace_id="trace-1",
        request_route="answer_question",
        response_classification="canonical_definition",
        policy_fingerprint="policy",
        source_versions={},
        tool_outcomes={"metricflow": "success"},
        retrieval_scores=(),
        evaluation_outcome="not_evaluated",
        response={
            "answer": "Sensitive evidence body tenant-123 should not reach an evaluator.",
            "result_classification": "canonical_definition",
            "trace_id": "trace-1",
            "effective_access_scope": {"regions": ["APAC", "EMEA"]},
            "source_freshness": {"is_current": True, "revision_id": "private"},
            "evidence": {"citations": [{"affected_scope": {"region": "APAC"}}]},
            "unrelated_private_field": "must be discarded",
        },
        node_spans=(TraceSpan("intent_interpretation", "node", "success", {}),),
        tool_spans=(
            TraceSpan(
                "semantic_query",
                "tool",
                "success",
                {"metric_name": "jira_new_peu", "raw_sql": "select secret"},
            ),
        ),
    )

    observation = observation_from_trace(trace)

    assert observation.selected_tools == ()
    assert observation.tool_calls[0].argument_keys == ("metric_name",)
    assert observation.response == {
        "has_answer": True,
        "result_classification": "canonical_definition",
        "trace_id": "trace-1",
        "effective_access_scope": {"regions": ("APAC", "EMEA")},
        "source_freshness": {"is_current": True},
        "has_evidence": True,
        "evidence_regions": ("APAC",),
        "unknown_region_observed": False,
        "has_candidate_causal_factors": False,
        "has_direct_identifier_answer": False,
        "has_active_investigation_selection": False,
    }


def test_dataset_rejects_unknown_source_dimension(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        '{"artifact_type":"deepeval_trajectory_dataset","dataset_version":"1.0.0",'
        '"published_at":"2026-09-01","cases":[]}'
    )

    with pytest.raises(ValueError):
        DeepEvalDatasetStore(path).load()


def test_deepeval_runner_exercises_setup_note_cases_with_a_harness(tmp_path: Path) -> None:
    dataset = DeepEvalDatasetStore(_DATASET_PATH).load()

    def observe(case):
        observations = []
        for index, classification in enumerate(case.expected_result_classifications):
            trace_id = f"trace-{case.case_id}-{index}"
            freshness = (
                case.expected_freshnesses[index]
                if case.expected_freshnesses
                else "current"
            )
            observations.append(
                _observation(
                    trace_id=trace_id,
                    response={
                        "answer": "A governed answer.",
                        "result_classification": classification,
                        "trace_id": trace_id,
                        "effective_access_scope": {
                            "products": ["Jira"],
                            "regions": ["APAC"],
                        },
                        "source_freshness": {"is_current": freshness != "stale"},
                        "has_active_investigation_selection": False,
                    },
                    selected_tools=(
                        case.expected_tool_selections[index]
                        if case.expected_tool_selections
                        else ()
                    ),
                    tool_calls=tuple(
                        ToolCallObservation(
                            name=name,
                            status="success",
                            argument_keys=argument_keys,
                        )
                        for name, argument_keys in (
                            case.expected_tool_argument_keys[index].items()
                            if case.expected_tool_argument_keys
                            else ()
                        )
                    ),
                )
            )
        return observations

    scorecard = run_deepeval_dataset(dataset, observe)

    assert scorecard.automated_cases == 6
    assert scorecard.not_yet_automated_cases == 0
    assert scorecard.trajectory.total == sum(
        case.expected_turn_count for case in dataset.cases
    ) * len(TrajectoryStage)
    assert trajectory_scorecard_passed(scorecard)
    assert not trajectory_scorecard_passed(
        replace(
            scorecard,
            multi_turn=replace(scorecard.multi_turn, failed=1, total=1, pass_rate=0.0),
        )
    )

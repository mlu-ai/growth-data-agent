"""Offline evaluation of governed multi-turn and tool trajectories.

The evaluator consumes only the public response contract and the allowlisted,
redacted fields on :class:`TraceRecord`. It never consumes prompts, private
reasoning, SQL, evidence bodies, or model-internal state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation_dataset import EvaluationSplit
from .observability import safe_trace_evaluation_projection

EVALUATOR_VERSION = "1.0.0"
_SAFE_TRACE_ARGUMENT_KEYS = frozenset(
    {"metric_name", "result_limit", "entity_name", "returned_count", "error_type"}
)
_PRIVATE_OUTPUT_KEYS = frozenset(
    {"reasoning", "thoughts", "chain_of_thought", "private_reasoning"}
)
_ACTION_EXECUTION_SPANS = {
    "metricflow": frozenset({"semantic_driver_decomposition", "semantic_query"}),
    "cited_evidence": frozenset({"evidence_retrieval"}),
    "lightrag": frozenset({"lightrag_retrieval", "graph_traversal"}),
}
_REAUTHORIZATION_EXECUTION_SPANS = frozenset(
    {
        "semantic_driver_decomposition",
        "lightrag_retrieval",
        "evidence_retrieval",
    }
)


class DeepEvalCaseDimension(StrEnum):
    SINGLE_TURN_CONTINUITY = "single_turn_continuity"
    MULTI_TURN_CONTINUITY = "multi_turn_continuity"
    CONVERSATION_SUMMARY_BOUNDARY = "conversation_summary_boundary"
    ACTIVE_INVESTIGATION_REAUTHORIZATION = "active_investigation_reauthorization"
    EVIDENCE_REVISION_FRESHNESS = "evidence_revision_freshness"


class DeepEvalCase(BaseModel):
    """A reproducible DeepEval case linked to a governed Evaluation Case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=128)
    source_case_id: str = Field(min_length=1, max_length=128)
    dimensions: tuple[DeepEvalCaseDimension, ...] = Field(min_length=1)
    split: EvaluationSplit
    expected_turn_count: int = Field(gt=0, le=32)
    expected_result_classifications: tuple[str, ...] = Field(min_length=1, max_length=32)
    allowed_tools: tuple[str, ...] = ()
    expected_tool_selections: tuple[tuple[str, ...], ...] = ()
    expected_tool_argument_keys: tuple[dict[str, tuple[str, ...]], ...] = ()
    expected_freshnesses: tuple[Literal["current", "stale", "not_applicable"], ...] = ()
    summary_boundary_turn: int | None = Field(default=None, ge=1)
    reauthorization_turns: tuple[int, ...] = ()
    selected_factor_id: str | None = Field(default=None, min_length=1, max_length=256)
    denied: bool = False
    setup_note: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _validate_expected_turn_count(self) -> DeepEvalCase:
        if len(self.expected_result_classifications) != self.expected_turn_count:
            raise ValueError(
                f"{self.case_id!r} expected_result_classifications must contain one value "
                "per turn."
            )
        if self.expected_tool_selections and (
            len(self.expected_tool_selections) != self.expected_turn_count
        ):
            raise ValueError(
                f"{self.case_id!r} expected_tool_selections must contain one value per turn."
            )
        if self.expected_freshnesses and len(self.expected_freshnesses) != self.expected_turn_count:
            raise ValueError(
                f"{self.case_id!r} expected_freshnesses must contain one value per turn."
            )
        if self.expected_tool_argument_keys and (
            len(self.expected_tool_argument_keys) != self.expected_turn_count
        ):
            raise ValueError(
                f"{self.case_id!r} expected_tool_argument_keys must contain one value per turn."
            )
        if (
            self.summary_boundary_turn is not None
            and self.summary_boundary_turn > self.expected_turn_count
        ):
            raise ValueError(f"{self.case_id!r} summary_boundary_turn exceeds expected turns.")
        if any(turn > self.expected_turn_count for turn in self.reauthorization_turns):
            raise ValueError(f"{self.case_id!r} reauthorization_turns exceed expected turns.")
        return self


class DeepEvalDataset(BaseModel):
    """Versioned offline manifest for DeepEval trajectory cases."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["deepeval_trajectory_dataset"] = "deepeval_trajectory_dataset"
    dataset_version: str = Field(min_length=1, max_length=32)
    published_at: date
    cases: list[DeepEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self) -> DeepEvalDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("DeepEval Case ids must be unique within the dataset.")
        dimensions = {dimension for case in self.cases for dimension in case.dimensions}
        missing_dimensions = set(DeepEvalCaseDimension) - dimensions
        if missing_dimensions:
            raise ValueError(f"DeepEval dimensions with no cases: {sorted(missing_dimensions)}.")
        splits = {case.split for case in self.cases}
        missing_splits = set(EvaluationSplit) - splits
        if missing_splits:
            raise ValueError(f"Splits with no cases: {sorted(missing_splits)}.")
        return self


class DeepEvalDatasetStore:
    """Load a DeepEval manifest from disk without caching it."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> DeepEvalDataset:
        return DeepEvalDataset.model_validate_json(self.path.read_text())


@dataclass(frozen=True)
class ToolCallObservation:
    """Safe metadata about one tool span; values and private arguments are absent."""

    name: str
    status: str
    argument_keys: tuple[str, ...] = ()
    unsafe_argument_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryObservation:
    """An evaluator-safe projection of a governed turn."""

    trace_id: str | None
    response: Mapping[str, Any]
    selected_tools: tuple[str, ...] = ()
    tool_calls: tuple[ToolCallObservation, ...] = ()
    conversation_id: str | None = None
    tool_outcomes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryExpectation:
    expected_result_classification: str
    expected_tool_selection: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    expected_tool_argument_keys: Mapping[str, tuple[str, ...]] | None = None
    expected_freshness: Literal["current", "stale", "not_applicable"] = "not_applicable"
    denied: bool = False


class GovernedToolCorrectnessMetric(BaseMetric):
    """DeepEval metric for the exact, allowlisted governed tool contract.

    This is deliberately deterministic: it receives only opaque tool names,
    does not invoke a judge model, and can run in CI without credentials.
    """

    def __init__(self) -> None:
        self.threshold = 1.0
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = True
        self.evaluation_cost = None

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        del _show_indicator, _in_component
        called = tuple(tool.name for tool in test_case.tools_called or ())
        expected = tuple(tool.name for tool in test_case.expected_tools or ())
        self.score = 1.0 if called == expected else 0.0
        self.reason = "governed tools matched" if self.score else "governed tools did not match"
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Governed Tool Correctness"


def evaluate_deepeval_tool_correctness(
    *, observed_tools: Sequence[str], expected_tools: Sequence[str]
) -> bool:
    """Evaluate the redacted tool trajectory through a real DeepEval test case."""
    test_case = LLMTestCase(
        input="governed_tool_trajectory",
        actual_output="governed_response",
        tools_called=[ToolCall(name=name) for name in observed_tools],
        expected_tools=[ToolCall(name=name) for name in expected_tools],
    )
    metric = GovernedToolCorrectnessMetric()
    metric.measure(test_case)
    return metric.success is True


@dataclass(frozen=True)
class TrajectoryFinding:
    stage: TrajectoryStage
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class TrajectoryResult:
    findings: tuple[TrajectoryFinding, ...]

    @property
    def passed(self) -> bool:
        return all(finding.passed for finding in self.findings)


class TrajectoryStage(StrEnum):
    REQUEST_INTERPRETATION = "request_interpretation"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_EXECUTION = "tool_execution"
    OUTPUT_HANDLING = "output_handling"
    FINAL_GOAL = "final_goal"


@dataclass(frozen=True)
class MultiTurnFinding:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class MultiTurnResult:
    name: str
    details: tuple[MultiTurnFinding, ...]

    @property
    def passed(self) -> bool:
        return all(detail.passed for detail in self.details)


@dataclass(frozen=True)
class TrajectoryScorecardCategory:
    name: str
    passed: int
    failed: int
    total: int
    pass_rate: float
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryScorecard:
    dataset_version: str
    evaluator_version: str
    generated_at: datetime
    total_cases: int
    automated_cases: int
    not_yet_automated_cases: int
    trajectory: TrajectoryScorecardCategory
    multi_turn: TrajectoryScorecardCategory


def trajectory_scorecard_passed(scorecard: TrajectoryScorecard) -> bool:
    """Return whether both independently reported trajectory categories passed."""
    return scorecard.trajectory.failed == 0 and scorecard.multi_turn.failed == 0


def observation_from_trace(trace: Any) -> TrajectoryObservation:
    """Project a ``TraceRecord`` into fields safe for trajectory evaluation."""
    metadata = trace.lead_agent_metadata
    if metadata is not None:
        selected_tools = tuple(action.value for action in metadata.actions)
    else:
        # A trace span proves internal execution, not a Lead Agent tool choice.
        # Canonical definitions, for example, use semantic internals without
        # taking a planned investigation action.
        selected_tools = ()
    tool_calls = tuple(
        ToolCallObservation(
            name=span.name,
            status=span.status,
            argument_keys=tuple(sorted(set(span.attributes) & _SAFE_TRACE_ARGUMENT_KEYS)),
            unsafe_argument_keys=tuple(sorted(set(span.attributes) - _SAFE_TRACE_ARGUMENT_KEYS)),
        )
        for span in trace.tool_spans
    )
    return TrajectoryObservation(
        trace_id=trace.trace_id,
        response=safe_trace_evaluation_projection(trace),
        selected_tools=selected_tools,
        tool_calls=tool_calls,
        conversation_id=trace.conversation_id,
        tool_outcomes=dict(trace.tool_outcomes),
    )


def evaluate_trajectory(
    observation: TrajectoryObservation,
    expectation: TrajectoryExpectation,
) -> TrajectoryResult:
    """Evaluate the six observable stages of one governed trajectory."""
    response = observation.response
    tool_calls = tuple(_coerce_tool_call(call) for call in observation.tool_calls)
    observed_classification = response.get("result_classification")
    selected_tools = observation.selected_tools or tuple(
        call.name for call in tool_calls
    )
    findings = [
        TrajectoryFinding(
            TrajectoryStage.REQUEST_INTERPRETATION,
            observed_classification == expectation.expected_result_classification,
            "request interpretation produced an unexpected governed classification"
            if observed_classification != expectation.expected_result_classification
            else "",
        ),
        _evaluate_tool_selection(selected_tools, expectation),
        _evaluate_tool_arguments(tool_calls, expectation),
        _evaluate_tool_execution(tool_calls, observation.tool_outcomes, expectation),
        _evaluate_output_handling(observation, expectation),
        _evaluate_final_goal(observation, expectation),
    ]
    return TrajectoryResult(findings=tuple(findings))


def evaluate_turn_sequence(
    observations: Sequence[TrajectoryObservation],
    *,
    summary_boundary_turn: int | None = None,
    reauthorization_turns: Sequence[int] = (),
) -> MultiTurnResult:
    """Check bounded conversation continuity independently of stage scores."""
    findings: list[MultiTurnFinding] = []
    if len(observations) < 2:
        findings.append(
            MultiTurnFinding(
                "turn_count", False, "multi-turn evaluation needs at least two turns"
            )
        )
        return MultiTurnResult("multi_turn", tuple(findings))
    conversation_ids = {observation.conversation_id for observation in observations}
    same_conversation = len(conversation_ids) == 1 and None not in conversation_ids
    findings.append(
        MultiTurnFinding(
            "conversation_continuity",
            same_conversation,
            "conversation_id changed or was missing across turns" if not same_conversation else "",
        )
    )
    trace_ids = [observation.trace_id for observation in observations]
    distinct_traces = None not in trace_ids and len(set(trace_ids)) == len(trace_ids)
    findings.append(
        MultiTurnFinding(
            "turn_reauthorization",
            distinct_traces,
            "trace ids were missing or repeated across turns" if not distinct_traces else "",
        )
    )
    if summary_boundary_turn is not None:
        boundary = observations[summary_boundary_turn - 1]
        cleared = boundary.response.get("has_active_investigation_selection") is False
        findings.append(
            MultiTurnFinding(
                "summary_boundary_clears_active_selection",
                cleared,
                "summary-boundary turn retained an active investigation selection"
                if not cleared
                else "",
            )
        )
    for turn in reauthorization_turns:
        calls = observations[turn - 1].tool_calls
        successful_spans = {
            _coerce_tool_call(call).name
            for call in calls
            if _coerce_tool_call(call).status == "success"
        }
        missing = _REAUTHORIZATION_EXECUTION_SPANS - successful_spans
        findings.append(
            MultiTurnFinding(
                f"reauthorization_turn_{turn}",
                not missing,
                f"reauthorization did not re-execute {sorted(missing)}"
                if missing
                else "",
            )
        )
    return MultiTurnResult("multi_turn", tuple(findings))


def run_deepeval_dataset(
    dataset: DeepEvalDataset,
    observe_case: Callable[[DeepEvalCase], Sequence[TrajectoryObservation]],
    *,
    evaluator_version: str = EVALUATOR_VERSION,
) -> TrajectoryScorecard:
    """Run automatable DeepEval cases and report explicit coverage."""
    trajectory_findings: list[TrajectoryFinding] = []
    multi_turn_findings: list[TrajectoryFinding] = []
    automated_cases = 0
    not_yet_automated_cases = 0
    for case in dataset.cases:
        automated_cases += 1
        observations = list(observe_case(case))
        if len(observations) != case.expected_turn_count:
            trajectory_findings.append(
                TrajectoryFinding(
                    TrajectoryStage.FINAL_GOAL,
                    False,
                    f"{case.case_id}: expected {case.expected_turn_count} turns, "
                    f"got {len(observations)}",
                )
            )
            if case.expected_turn_count > 1:
                multi_turn_findings.append(
                    TrajectoryFinding(
                        TrajectoryStage.FINAL_GOAL,
                        False,
                        f"{case.case_id}: multi-turn replay expected "
                        f"{case.expected_turn_count} turns, got {len(observations)}",
                    )
                )
            continue
        for index, (observation, expected_classification) in enumerate(zip(
            observations, case.expected_result_classifications, strict=True
        )):
            expected_tools = (
                case.expected_tool_selections[index] if case.expected_tool_selections else ()
            )
            expected_freshness = (
                case.expected_freshnesses[index] if case.expected_freshnesses else "not_applicable"
            )
            expected_argument_keys = (
                case.expected_tool_argument_keys[index]
                if case.expected_tool_argument_keys
                else None
            )
            trajectory_findings.extend(
                evaluate_trajectory(
                    observation,
                    TrajectoryExpectation(
                        expected_result_classification=expected_classification,
                        expected_tool_selection=expected_tools,
                        allowed_tools=case.allowed_tools,
                        expected_tool_argument_keys=expected_argument_keys,
                        expected_freshness=expected_freshness,
                        denied=case.denied,
                    ),
                ).findings
            )
        if case.expected_turn_count > 1:
            sequence = evaluate_turn_sequence(
                observations,
                summary_boundary_turn=case.summary_boundary_turn,
                reauthorization_turns=case.reauthorization_turns,
            )
            multi_turn_findings.extend(
                TrajectoryFinding(
                    TrajectoryStage.FINAL_GOAL,
                    detail.passed,
                    f"{case.case_id}: {detail.detail}",
                )
                for detail in sequence.details
            )
    return TrajectoryScorecard(
        dataset_version=dataset.dataset_version,
        evaluator_version=evaluator_version,
        generated_at=datetime.now(UTC),
        total_cases=len(dataset.cases),
        automated_cases=automated_cases,
        not_yet_automated_cases=not_yet_automated_cases,
        trajectory=_category("trajectory", trajectory_findings),
        multi_turn=_category("multi_turn", multi_turn_findings),
    )


def _evaluate_tool_selection(
    selected_tools: Sequence[str], expectation: TrajectoryExpectation
) -> TrajectoryFinding:
    allowed = set(expectation.allowed_tools)
    expected = tuple(expectation.expected_tool_selection)
    deepeval_ok = evaluate_deepeval_tool_correctness(
        observed_tools=selected_tools,
        expected_tools=expected,
    )
    ok = (
        (not expectation.denied or not selected_tools)
        and (not allowed or set(selected_tools).issubset(allowed))
        and (not expected or tuple(selected_tools) == expected)
        and deepeval_ok
    )
    return TrajectoryFinding(
        TrajectoryStage.TOOL_SELECTION,
        ok,
        f"selected tools {tuple(selected_tools)!r} are outside the governed selection"
        if not ok
        else "",
    )


def _coerce_tool_call(call: ToolCallObservation | Mapping[str, Any]) -> ToolCallObservation:
    if isinstance(call, ToolCallObservation):
        return call
    return ToolCallObservation(
        name=str(call.get("name", "")),
        status=str(call.get("status", "")),
        argument_keys=tuple(str(key) for key in call.get("argument_keys", ())),
    )


def _evaluate_tool_arguments(
    tool_calls: Sequence[ToolCallObservation], expectation: TrajectoryExpectation
) -> TrajectoryFinding:
    problems: list[str] = []
    expected_keys = expectation.expected_tool_argument_keys or {}
    for call in tool_calls:
        unsafe = set(call.argument_keys) - _SAFE_TRACE_ARGUMENT_KEYS
        unsafe.update(call.unsafe_argument_keys)
        if unsafe:
            problems.append(f"{call.name}: unsafe argument keys {sorted(unsafe)}")
        required = set(expected_keys.get(call.name, ()))
        if not required.issubset(call.argument_keys):
            problems.append(f"{call.name}: missing expected argument keys {sorted(required)}")
    return TrajectoryFinding(TrajectoryStage.TOOL_ARGUMENTS, not problems, "; ".join(problems))


def _evaluate_tool_execution(
    tool_calls: Sequence[ToolCallObservation],
    tool_outcomes: Mapping[str, str],
    expectation: TrajectoryExpectation,
) -> TrajectoryFinding:
    executed_outcomes = {
        name: status for name, status in tool_outcomes.items() if status != "not_used"
    }
    if expectation.denied and (tool_calls or executed_outcomes):
        return TrajectoryFinding(
            TrajectoryStage.TOOL_EXECUTION, False, "denied trajectory executed a tool"
        )
    invalid = [call.status for call in tool_calls if call.status not in {"success", "error"}]
    invalid.extend(
        status
        for status in executed_outcomes.values()
        if status not in {"success", "failed", "error"}
    )
    missing_actions = []
    permitted_spans: set[str] = set()
    for action in expectation.expected_tool_selection:
        expected_spans = _ACTION_EXECUTION_SPANS.get(action, frozenset())
        permitted_spans.update(expected_spans)
        observed = {
            call.name for call in tool_calls if call.status == "success"
        }
        if expected_spans and not observed.intersection(expected_spans):
            missing_actions.append(action)
    details = []
    if invalid:
        details.append(f"invalid tool statuses: {invalid!r}")
    if missing_actions:
        details.append(f"selected actions without observed execution: {missing_actions!r}")
    unexpected_spans = (
        sorted({call.name for call in tool_calls} - permitted_spans)
        if expectation.expected_tool_selection
        else []
    )
    if unexpected_spans:
        details.append(f"unplanned execution spans: {unexpected_spans!r}")
    return TrajectoryFinding(
        TrajectoryStage.TOOL_EXECUTION,
        not details,
        "; ".join(details),
    )


def _evaluate_output_handling(
    observation: TrajectoryObservation, expectation: TrajectoryExpectation
) -> TrajectoryFinding:
    response = observation.response
    missing = [
        key
        for key in ("answer", "result_classification", "trace_id")
        if not _field_present(response, key)
    ]
    private_keys = _PRIVATE_OUTPUT_KEYS.intersection(response)
    if expectation.denied and any(
        _field_present(response, key)
        for key in ("evidence", "candidate_causal_factors", "direct_identifier_answer")
    ):
        missing.append("denied output leak")
    freshness_is_current = response.get("source_freshness_is_current")
    if freshness_is_current is None:
        freshness_is_current = (response.get("source_freshness") or {}).get("is_current")
    if expectation.expected_freshness == "current" and freshness_is_current is not True:
        missing.append("current freshness proof")
    if expectation.expected_freshness == "stale" and (
        freshness_is_current is not False
        or response.get("result_classification") != "limitation"
    ):
        missing.append("stale freshness limitation")
    problems = [*missing, *[f"private output key {key}" for key in sorted(private_keys)]]
    return TrajectoryFinding(TrajectoryStage.OUTPUT_HANDLING, not problems, "; ".join(problems))


def _evaluate_final_goal(
    observation: TrajectoryObservation, expectation: TrajectoryExpectation
) -> TrajectoryFinding:
    response = observation.response
    ok = (
        response.get("result_classification") == expectation.expected_result_classification
        and _field_present(response, "answer")
        and (not expectation.denied or response.get("result_classification") == "safe_refusal")
    )
    return TrajectoryFinding(
        TrajectoryStage.FINAL_GOAL,
        ok,
        "final governed goal was not achieved" if not ok else "",
    )


def _category(name: str, findings: Sequence[TrajectoryFinding]) -> TrajectoryScorecardCategory:
    passed = sum(finding.passed for finding in findings)
    failed = len(findings) - passed
    details = tuple(
        f"{finding.stage.value}: {finding.detail}" for finding in findings if not finding.passed
    )[:50]
    return TrajectoryScorecardCategory(
        name=name,
        passed=passed,
        failed=failed,
        total=len(findings),
        pass_rate=(passed / len(findings)) if findings else 1.0,
        details=details,
    )


def _field_present(response: Mapping[str, Any], field: str) -> bool:
    if field == "answer" and "answer_present" in response:
        return response["answer_present"] is True
    safe_flag = f"has_{field}"
    if safe_flag in response:
        return response[safe_flag] is True
    return response.get(field) is not None and bool(response.get(field))

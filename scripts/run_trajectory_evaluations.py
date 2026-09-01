"""Run the DeepEval-linked trajectory cases and inspect Promptfoo boundaries.

The script requires the same local Postgres and dbt semantic artifact as the
governed evaluation runner. Promptfoo itself runs its YAML matrix separately
against PROMPTFOO_TARGET_URL; this script reports deterministic boundary
checks without requiring Node or a Promptfoo installation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from growth_data_agent.adversarial_evaluation import (
    AdversarialObservation,
    PromptfooMatrixStore,
    run_promptfoo_matrix,
)
from growth_data_agent.evaluation_dataset import EvaluationDatasetStore
from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import MetricFlowPlanner, PostgresMetricFlowExecutor
from growth_data_agent.observability import MlflowTraceSink
from growth_data_agent.principal import development_token_environment_variable
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.trajectory_evaluation import (
    DeepEvalCase,
    DeepEvalDatasetStore,
    TrajectoryObservation,
    observation_from_trace,
    run_deepeval_dataset,
    trajectory_scorecard_passed,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"
_GOVERNED_DATASET = _REPOSITORY_ROOT / "evaluations/dataset/v1/cases.json"
_DEEPEVAL_DATASET = _REPOSITORY_ROOT / "evaluations/deepeval/v1/cases.json"
_PROMPTFOO_MATRIX = _REPOSITORY_ROOT / "evaluations/promptfoo/matrix.json"


class _RecordingTraceSink:
    def __init__(self, delegate: MlflowTraceSink) -> None:
        self.delegate = delegate
        self.records = []

    def record(self, trace) -> None:
        self.records.append(trace)
        self.delegate.record(trace)

    def record_trajectory_scorecard(self, scorecard) -> None:
        self.delegate.record_trajectory_scorecard(scorecard)

    def record_adversarial_scorecard(self, scorecard) -> None:
        self.delegate.record_adversarial_scorecard(scorecard)


def _client(artifact_path: Path = _ARTIFACT) -> tuple[TestClient, _RecordingTraceSink]:
    if not _MANIFEST.exists():
        raise SystemExit("Missing dbt/target/semantic_manifest.json; run make dbt-build first.")
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data",
    )
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=MetricFlowPlanner(_MANIFEST),
        postgres_executor=PostgresMetricFlowExecutor(database_url),
    )
    trace_sink = _RecordingTraceSink(MlflowTraceSink.from_environment())
    service = AnswerQuestionService(
        gateway,
        evidence_reranker=DeterministicCrossEncoderReranker(),
        trace_sink=trace_sink,
    )
    return TestClient(create_app(service)), trace_sink


def _invoke_evaluation(client: TestClient, request: dict[str, object]) -> dict[str, object]:
    payload = dict(request)
    principal_id = payload.pop("agent_user_id", None)
    if not isinstance(principal_id, str) or not principal_id:
        raise SystemExit("Evaluation request is missing agent_user_id.")
    token = os.environ.get(development_token_environment_variable(principal_id))
    if not token:
        raise SystemExit(f"Missing development bearer token configuration for {principal_id!r}.")
    evaluation_token = os.environ.get("GROWTH_DATA_AGENT_EVALUATION_TOKEN")
    if not evaluation_token:
        raise SystemExit("Missing GROWTH_DATA_AGENT_EVALUATION_TOKEN.")
    response = client.post(
        "/evaluation/answer_question",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Evaluation-Token": evaluation_token,
        },
        json=payload,
    )
    body = response.json() if response.is_success else {}
    return {"status_code": response.status_code, "body": body}


def _deep_observer(
    client: TestClient,
    trace_sink: _RecordingTraceSink,
    governed_cases: dict[str, object],
):
    def observe(case: DeepEvalCase) -> list[TrajectoryObservation]:
        case_client = client
        case_sink = trace_sink
        temporary_directory = None
        if any(
            dimension.value == "evidence_revision_freshness" for dimension in case.dimensions
        ):
            temporary_directory = tempfile.TemporaryDirectory(prefix="trajectory-evaluation-")
            temporary_artifact = Path(temporary_directory.name) / "semantic.json"
            shutil.copyfile(_ARTIFACT, temporary_artifact)
            case_client, case_sink = _client(temporary_artifact)
        source = governed_cases[case.source_case_id]
        observations = []
        conversation_id: str | None = None
        for index, turn in enumerate(source.turns):
            record_count = len(case_sink.records)
            request = dict(turn.request)
            if conversation_id is not None and "conversation_id" not in request:
                request["conversation_id"] = conversation_id
            if (
                any(
                    dimension.value == "active_investigation_reauthorization"
                    for dimension in case.dimensions
                )
                and index == 1
                and observations
            ):
                request["selected_factor_id"] = case.selected_factor_id
            result = _invoke_evaluation(case_client, request)
            body = result["body"]
            if not isinstance(body, dict):
                body = {}
            response = body.get("response")
            response = response if isinstance(response, dict) else {}
            response_trace_id = response.get("trace_id")
            trace = next(
                (
                    record
                    for record in reversed(case_sink.records[record_count:])
                    if record.trace_id == response_trace_id
                ),
                None,
            )
            observation = (
                observation_from_trace(trace)
                if trace is not None
                else TrajectoryObservation(
                    trace_id=response.get("trace_id")
                    if isinstance(response.get("trace_id"), str)
                    else None,
                    response=response,
                )
            )
            observations.append(observation)
            conversation_id = trace.conversation_id if trace is not None else None
            if (
                any(
                    dimension.value == "evidence_revision_freshness"
                    for dimension in case.dimensions
                )
                and index == 0
                and temporary_artifact is not None
            ):
                artifact = json.loads(temporary_artifact.read_text())
                artifact["validation"]["status"] = "failed"
                temporary_artifact.write_text(json.dumps(artifact))
        if temporary_directory is not None:
            temporary_directory.cleanup()
        return observations

    return observe


def _promptfoo_observer(client: TestClient, trace_sink: _RecordingTraceSink):
    def observe(case) -> AdversarialObservation:
        record_count = len(trace_sink.records)
        result = _invoke_evaluation(
            client,
            {"agent_user_id": case.agent_user_id, "question": case.question},
        )
        body = result["body"]
        if not isinstance(body, dict):
            body = {}
        trace = trace_sink.records[-1] if len(trace_sink.records) > record_count else None
        observation = observation_from_trace(trace) if trace is not None else None
        response = observation.response if observation is not None else {}
        return AdversarialObservation(
            status_code=int(result["status_code"]),
            response=response,
            tool_names=tuple(call.name for call in observation.tool_calls)
            if observation is not None
            else (),
            evidence_regions=tuple(response.get("evidence_regions", ())),
        )

    return observe


def main() -> int:
    governed = EvaluationDatasetStore(_GOVERNED_DATASET).load()
    deep_dataset = DeepEvalDatasetStore(_DEEPEVAL_DATASET).load()
    matrix = PromptfooMatrixStore(_PROMPTFOO_MATRIX).load()
    governed_cases = {case.case_id: case for case in governed.cases}
    client, trace_sink = _client()
    trajectory = run_deepeval_dataset(
        deep_dataset,
        _deep_observer(client, trace_sink, governed_cases),
    )
    adversarial = run_promptfoo_matrix(matrix, _promptfoo_observer(client, trace_sink))
    trace_sink.record_trajectory_scorecard(trajectory)
    trace_sink.record_adversarial_scorecard(adversarial)
    print(
        json.dumps(
            {
                "trajectory": trajectory.trajectory.__dict__,
                "multi_turn": trajectory.multi_turn.__dict__,
                "adversarial": adversarial.category.__dict__,
                "coverage": {
                    "trajectory_total_cases": trajectory.total_cases,
                    "trajectory_automated_cases": trajectory.automated_cases,
                    "trajectory_not_yet_automated_cases": trajectory.not_yet_automated_cases,
                },
            },
            default=str,
            indent=2,
        )
    )
    return 0 if trajectory_scorecard_passed(trajectory) and not adversarial.failed else 1


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from growth_data_agent.contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    ConversationTurn,
    ResultClassification,
)
from growth_data_agent.conversations import SQLiteConversationCheckpointStore
from growth_data_agent.execution import AuthorizedExecution
from growth_data_agent.observability import TraceRecord, _redact_trace_payload
from growth_data_agent.planning import (
    InvestigationComplexity,
    LeadAgentPlanner,
    PlanAction,
    PlanActionExecution,
    PlanExecutionSnapshot,
    PlanningInvariantError,
    PlanToolOutcome,
    ToolOutcomeStatus,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.principal import VerifiedPrincipal


def _authorized() -> AuthorizedExecution:
    request = AnswerQuestionRequest(
        agent_user_id="data_analyst",
        question="What evidence may explain the APAC 51-200-seat Tenant decline?",
    )
    profile = resolve_access_profile(request.agent_user_id)
    return AuthorizedExecution(
        request=request,
        access_profile=profile,
        effective_scope=profile.as_effective_scope(),
        trace_id="trace-planning",
    )


def test_simple_canonical_definition_bypasses_lead_planning() -> None:
    planner = LeadAgentPlanner()

    metadata = planner.start(
        AnalyticalIntent(
            route=AnalyticalRoute.CANONICAL_DEFINITION,
            metric_name="jira_new_peu",
        ),
        _authorized(),
        semantic_current=True,
    )

    assert metadata is None


def test_medium_and_hard_investigations_have_bounded_typed_actions() -> None:
    planner = LeadAgentPlanner()
    authorized = _authorized()

    medium = planner.start(
        AnalyticalIntent(
            route=AnalyticalRoute.DRIVER_DECOMPOSITION,
            metric_name="jira_new_peu",
        ),
        authorized,
        semantic_current=True,
    )
    hard = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        authorized,
        semantic_current=True,
    )

    assert medium is not None
    assert medium.complexity is InvestigationComplexity.MEDIUM
    assert medium.actions == [PlanAction.METRICFLOW]
    assert len(medium.actions) <= 3
    assert hard is not None
    assert hard.complexity is InvestigationComplexity.HARD
    assert hard.actions == [
        PlanAction.METRICFLOW,
        PlanAction.CITED_EVIDENCE,
        PlanAction.LIGHTRAG,
    ]
    assert len(hard.actions) <= 3


def test_tool_failure_replans_to_the_next_existing_action_only() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
    )
    assert metadata is not None

    replanned = planner.replan(
        metadata,
        PlanToolOutcome(
            action=PlanAction.METRICFLOW,
            status=ToolOutcomeStatus.FAILED,
            error_type="SemanticQueryExecutionError",
        ),
        current_policy_fingerprint=metadata.policy_fingerprint,
        semantic_current=True,
        evidence_revision_keys=(),
    )

    assert replanned.replan_count == 1
    assert replanned.plan_revision == metadata.plan_revision + 1
    assert replanned.current_action is PlanAction.CITED_EVIDENCE
    assert replanned.actions == metadata.actions
    assert replanned.tool_outcomes[-1].error_type == "SemanticQueryExecutionError"


def test_action_execution_records_real_order_and_replans_before_next_action() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
    )
    assert metadata is not None
    calls: list[PlanAction] = []

    def run(action: PlanAction, payload: object | None) -> PlanActionExecution:
        calls.append(action)
        if action is PlanAction.METRICFLOW:
            raise RuntimeError("metricflow unavailable")
        return PlanActionExecution(value=f"completed:{action.value}", payload=payload)

    result = planner.execute(
        metadata,
        run_action=run,
        snapshot_provider=lambda _: PlanExecutionSnapshot(
            policy_fingerprint=metadata.policy_fingerprint,
            semantic_current=True,
            evidence_revision_keys=(),
        ),
    )

    assert calls == [PlanAction.METRICFLOW, PlanAction.CITED_EVIDENCE, PlanAction.LIGHTRAG]
    assert [outcome.action for outcome in result.metadata.tool_outcomes] == calls
    assert result.metadata.tool_outcomes[0].status is ToolOutcomeStatus.FAILED
    assert result.metadata.tool_outcomes[1].status is ToolOutcomeStatus.SUCCESS
    assert result.metadata.current_action is None


def test_action_execution_blocks_next_action_when_policy_changes() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
    )
    assert metadata is not None
    calls: list[PlanAction] = []

    def run(action: PlanAction, payload: object | None) -> PlanActionExecution:
        calls.append(action)
        return PlanActionExecution(payload=payload)

    snapshots = iter(
        (
            PlanExecutionSnapshot(
                policy_fingerprint=metadata.policy_fingerprint,
                semantic_current=True,
                evidence_revision_keys=(),
            ),
            PlanExecutionSnapshot(
                policy_fingerprint="expanded-policy",
                semantic_current=True,
                evidence_revision_keys=(),
            ),
        )
    )
    result = planner.execute(metadata, run_action=run, snapshot_provider=lambda _: next(snapshots))

    assert calls == [PlanAction.METRICFLOW]
    assert result.metadata.last_replan_reason == "invariant_blocked"
    assert result.metadata.current_action is PlanAction.CITED_EVIDENCE


def test_action_execution_blocks_next_action_when_evidence_revision_changes() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
    )
    assert metadata is not None
    calls: list[PlanAction] = []
    snapshots = iter(
        (
            PlanExecutionSnapshot(
                policy_fingerprint=metadata.policy_fingerprint,
                semantic_current=True,
                evidence_revision_keys=(),
            ),
            PlanExecutionSnapshot(
                policy_fingerprint=metadata.policy_fingerprint,
                semantic_current=True,
                evidence_revision_keys=(("doc-1", "revision-2", "chunk-1"),),
            ),
        )
    )

    def run(action: PlanAction, payload: object | None) -> PlanActionExecution:
        calls.append(action)
        return PlanActionExecution(
            payload=payload,
            evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
        )

    result = planner.execute(metadata, run_action=run, snapshot_provider=lambda _: next(snapshots))

    assert calls == [PlanAction.METRICFLOW]
    assert result.metadata.last_replan_reason == "invariant_blocked"
    assert result.metadata.current_action is PlanAction.CITED_EVIDENCE


def test_replan_rejects_stale_semantic_or_evidence_state() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
        evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
    )
    assert metadata is not None
    outcome = PlanToolOutcome(action=PlanAction.METRICFLOW, status=ToolOutcomeStatus.SUCCESS)

    with pytest.raises(PlanningInvariantError, match="semantic freshness"):
        planner.replan(
            metadata,
            outcome,
            current_policy_fingerprint=metadata.policy_fingerprint,
            semantic_current=False,
            evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
        )

    with pytest.raises(PlanningInvariantError, match="evidence revision"):
        planner.replan(
            metadata,
            outcome,
            current_policy_fingerprint=metadata.policy_fingerprint,
            semantic_current=True,
            evidence_revision_keys=(("doc-1", "revision-2", "chunk-1"),),
        )


def test_replan_rejects_entitlement_expansion() -> None:
    planner = LeadAgentPlanner()
    metadata = planner.start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
    )
    assert metadata is not None

    with pytest.raises(PlanningInvariantError, match="policy"):
        planner.replan(
            metadata,
            PlanToolOutcome(action=PlanAction.METRICFLOW, status=ToolOutcomeStatus.SUCCESS),
            current_policy_fingerprint="different-policy",
            semantic_current=True,
            evidence_revision_keys=(),
        )


def test_turn_metadata_contains_no_source_content() -> None:
    metadata = LeadAgentPlanner().start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
        evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
    )
    assert metadata is not None
    turn = ConversationTurn(
        turn_id="turn-planning",
        question="What evidence may explain the decline?",
        result_classification=ResultClassification.HYPOTHESIS,
        metric_name="jira_new_peu",
        trace_id="trace-planning",
        created_at=datetime.now(UTC),
        lead_agent_metadata=metadata,
    )

    serialized = str(turn.model_dump(mode="json"))
    assert "source-page body" not in serialized
    assert "restricted document content" not in serialized
    assert "revision-1" not in serialized


def test_trace_metadata_is_inspectable_without_source_content() -> None:
    metadata = LeadAgentPlanner().start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
        evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
    )
    assert metadata is not None
    trace = TraceRecord(
        trace_id="trace-planning",
        request_route="answer_question",
        response_classification="hypothesis",
        policy_fingerprint=metadata.policy_fingerprint,
        source_versions={"semantic_version": "1.0.0"},
        tool_outcomes={"retrieval": "success"},
        retrieval_scores=(),
        evaluation_outcome="not_evaluated",
        response={"answer": "restricted document content"},
        lead_agent_metadata=metadata,
    )

    payload = _redact_trace_payload(trace)

    assert payload["lead_agent_metadata"]["complexity"] == "hard"
    assert "restricted document content" not in str(payload)
    assert "revision-1" not in str(payload)


def test_simple_response_bypasses_planning_and_investigation_persists_metadata(client) -> None:
    simple = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "Define Jira New PEU"},
    )
    assert simple.status_code == 200
    assert simple.json()["lead_agent_metadata"] is None

    investigation = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "What evidence may explain the APAC 51-200-seat Tenant decline?",
        },
    )
    assert investigation.status_code == 200
    metadata = investigation.json()["lead_agent_metadata"]
    assert metadata["complexity"] == "hard"
    assert metadata["actions"] == ["metricflow", "cited_evidence", "lightrag"]
    assert metadata["tool_outcomes"] == [
        {"action": "metricflow", "status": "success", "error_type": None},
        {"action": "cited_evidence", "status": "success", "error_type": None},
        {"action": "lightrag", "status": "success", "error_type": None},
    ]

    store = client.app.state.answer_service.conversation_store
    records = list(store._records.values())
    stored_metadata = records[-1]["transcript"][-1].lead_agent_metadata
    assert stored_metadata.model_dump(mode="json") == metadata


def test_sqlite_turn_persists_planning_metadata_without_revision_keys(tmp_path) -> None:
    metadata = LeadAgentPlanner().start(
        AnalyticalIntent(route=AnalyticalRoute.LEGACY, metric_name="jira_new_peu"),
        _authorized(),
        semantic_current=True,
        evidence_revision_keys=(("doc-1", "revision-1", "chunk-1"),),
    )
    assert metadata is not None
    principal = VerifiedPrincipal(
        principal_id="data_analyst",
        issuer="https://issuer.example.test",
        subject="subject-123",
    )
    store = SQLiteConversationCheckpointStore(tmp_path / "conversations.sqlite3")
    checkpoint = store.create(principal)
    store.append(
        checkpoint.conversation_id,
        principal,
        turn=ConversationTurn(
            turn_id="turn-persisted",
            question="What evidence may explain the decline?",
            result_classification=ResultClassification.HYPOTHESIS,
            metric_name="jira_new_peu",
            trace_id="trace-persisted",
            created_at=datetime.now(UTC),
            lead_agent_metadata=metadata,
        ),
        summary=checkpoint.summary,
    )

    loaded = store.transcript(checkpoint.conversation_id, principal)

    assert loaded[0].lead_agent_metadata == metadata
    assert b"revision-1" not in (tmp_path / "conversations.sqlite3").read_bytes()

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as RawTestClient

from growth_data_agent.contracts import (
    AnswerQuestionRequest,
    ConversationSummary,
    ConversationTurn,
    EffectiveAccessScope,
    ResultClassification,
)
from growth_data_agent.conversations import (
    ConversationAccessDeniedError,
    InMemoryConversationCheckpointStore,
    SQLiteConversationCheckpointStore,
)
from growth_data_agent.policy import AccessDeniedError
from growth_data_agent.principal import VerifiedPrincipal, development_token_environment_variable


def _token(principal_id: str) -> str:
    return os.environ[development_token_environment_variable(principal_id)]


def _answer(
    client: TestClient,
    *,
    question: str,
    conversation_id: str | None = None,
    selected_factor_id: str | None = None,
):
    payload = {"question": question}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if selected_factor_id is not None:
        payload["selected_factor_id"] = selected_factor_id
    return client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {_token('data_analyst')}"},
        json=payload,
    )


_APAC_EVIDENCE_QUESTION = "What evidence may explain the APAC 51–200-seat Tenant decline?"


def test_first_answer_creates_an_opaque_conversation_and_trace(client: TestClient) -> None:
    response = _answer(client, question="What is Jira New PEU?")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["conversation_id"], str)
    assert len(body["conversation_id"]) >= 32
    assert body["conversation_id"] not in {"data_analyst", "jira_new_peu"}
    assert body["trace_id"]


def test_follow_up_uses_prior_metric_context_and_gets_a_new_trace(client: TestClient) -> None:
    first = _answer(client, question="What is Jira New PEU?")
    conversation_id = first.json()["conversation_id"]

    follow_up = _answer(
        client,
        question="What does that metric mean?",
        conversation_id=conversation_id,
    )

    assert follow_up.status_code == 200
    body = follow_up.json()
    assert body["conversation_id"] == conversation_id
    assert body["trace_id"] != first.json()["trace_id"]
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"]["name"] == "jira_new_peu"


def test_another_verified_principal_cannot_continue_a_private_conversation(
    client: TestClient,
) -> None:
    first = _answer(client, question="What is Jira New PEU?")
    conversation_id = first.json()["conversation_id"]
    other_client = RawTestClient(client.app)

    response = other_client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {_token('apac_regional_manager')}"},
        json={
            "conversation_id": conversation_id,
            "question": "What does that metric mean?",
        },
    )

    assert response.status_code == 403
    assert "conversation" in response.json()["detail"].casefold()
    assert conversation_id not in response.text


def test_follow_up_refreshes_semantic_freshness_before_using_saved_metric_context(
    client: TestClient,
) -> None:
    first = _answer(client, question="What is Jira New PEU?")
    conversation_id = first.json()["conversation_id"]
    artifact_path: Path = client.app.state.answer_service.semantic_gateway.artifact_store.path
    artifact = json.loads(artifact_path.read_text())
    artifact["validation"]["status"] = "failed"
    artifact_path.write_text(json.dumps(artifact))

    follow_up = _answer(
        client,
        question="What does that metric mean?",
        conversation_id=conversation_id,
    )

    assert follow_up.status_code == 200
    assert follow_up.json()["result_classification"] == "limitation"
    assert follow_up.json()["source_freshness"]["is_current"] is False


def test_selected_factor_is_revalidated_on_a_later_turn_without_resending_it(
    client: TestClient,
) -> None:
    discover = _answer(client, question=_APAC_EVIDENCE_QUESTION)
    conversation_id = discover.json()["conversation_id"]
    factor_id = discover.json()["candidate_causal_factors"][0]["factor_id"]

    select = _answer(
        client,
        question=_APAC_EVIDENCE_QUESTION,
        conversation_id=conversation_id,
        selected_factor_id=factor_id,
    )
    assert select.status_code == 200
    assert [card["factor_id"] for card in select.json()["candidate_causal_factors"]] == [
        factor_id
    ]

    reassert = _answer(
        client, question=_APAC_EVIDENCE_QUESTION, conversation_id=conversation_id
    )

    assert reassert.status_code == 200
    body = reassert.json()
    assert [card["factor_id"] for card in body["candidate_causal_factors"]] == [factor_id]
    trace_ids = {discover.json()["trace_id"], select.json()["trace_id"], body["trace_id"]}
    assert len(trace_ids) == 3  # each turn re-ran the full pipeline, not a cached replay


def test_selecting_an_unknown_factor_id_returns_a_limitation_response(
    client: TestClient,
) -> None:
    response = _answer(
        client,
        question=_APAC_EVIDENCE_QUESTION,
        selected_factor_id="does-not-exist",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "limitation"
    assert body["candidate_causal_factors"] == []
    assert "could not be revalidated" in body["answer"]


def test_selected_factor_is_lost_when_entitlement_narrows_between_turns(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    discover = _answer(client, question=_APAC_EVIDENCE_QUESTION)
    conversation_id = discover.json()["conversation_id"]
    factor_id = discover.json()["candidate_causal_factors"][0]["factor_id"]

    select = _answer(
        client,
        question=_APAC_EVIDENCE_QUESTION,
        conversation_id=conversation_id,
        selected_factor_id=factor_id,
    )
    assert [card["factor_id"] for card in select.json()["candidate_causal_factors"]] == [
        factor_id
    ]

    from dataclasses import replace

    from growth_data_agent import policy

    narrowed_profile = replace(
        policy._PROFILES["data_analyst"], evidence_groups=("analytics-readers",)
    )
    monkeypatch.setitem(policy._PROFILES, "data_analyst", narrowed_profile)

    reassert = _answer(
        client, question=_APAC_EVIDENCE_QUESTION, conversation_id=conversation_id
    )

    assert reassert.status_code == 200
    body = reassert.json()
    assert body["result_classification"] == "limitation"
    assert body["candidate_causal_factors"] == []


def test_unrelated_question_does_not_inherit_a_prior_selection(client: TestClient) -> None:
    discover = _answer(client, question=_APAC_EVIDENCE_QUESTION)
    conversation_id = discover.json()["conversation_id"]
    factor_id = discover.json()["candidate_causal_factors"][0]["factor_id"]
    _answer(
        client,
        question=_APAC_EVIDENCE_QUESTION,
        conversation_id=conversation_id,
        selected_factor_id=factor_id,
    )

    unrelated = _answer(
        client, question="What is Confluence New MAU?", conversation_id=conversation_id
    )

    assert unrelated.status_code == 200
    body = unrelated.json()
    assert body["result_classification"] == "canonical_definition"
    assert body.get("candidate_causal_factors") is None
    assert factor_id not in unrelated.text


def test_switching_metric_context_does_not_pair_a_stale_factor_id_with_the_new_metric(
    client: TestClient,
) -> None:
    """A detour to an unrelated metric must clear the stored reference, not carry the
    old factor_id forward paired with the new metric_name — otherwise a later, genuinely
    fresh investigation on that new metric could be wrongly rejected as a lost selection
    no one ever made."""
    jira_discover = _answer(client, question=_APAC_EVIDENCE_QUESTION)
    conversation_id = jira_discover.json()["conversation_id"]
    jira_factor_id = jira_discover.json()["candidate_causal_factors"][0]["factor_id"]
    select = _answer(
        client,
        question=_APAC_EVIDENCE_QUESTION,
        conversation_id=conversation_id,
        selected_factor_id=jira_factor_id,
    )
    assert [c["factor_id"] for c in select.json()["candidate_causal_factors"]] == [
        jira_factor_id
    ]

    # Detour: a canonical-definition turn on a *different* metric (Confluence New PEU,
    # the same metric the next evidence question below investigates).
    detour = _answer(
        client, question="What is Confluence New PEU?", conversation_id=conversation_id
    )
    assert detour.json()["result_classification"] == "canonical_definition"

    # A genuinely fresh evidence investigation on that new metric, with no selection
    # sent — must NOT be rejected as a lost selection for a factor it never selected.
    confluence_evidence = _answer(
        client,
        question=(
            "What evidence may explain the Americas 11–50-seat Confluence New PEU "
            "movement after the acquisition campaign?"
        ),
        conversation_id=conversation_id,
    )

    assert confluence_evidence.status_code == 200
    body = confluence_evidence.json()
    assert body["result_classification"] != "limitation"
    assert jira_factor_id not in confluence_evidence.text
    assert len(body["candidate_causal_factors"]) == 1
    assert body["candidate_causal_factors"][0]["factor_id"] != jira_factor_id


def _principal() -> VerifiedPrincipal:
    return VerifiedPrincipal(
        principal_id="data_analyst",
        issuer="https://issuer.example.test",
        subject="subject-123",
    )


def _turn(*, created_at: datetime, question: str, metric_name: str = "jira_new_peu"):
    return ConversationTurn(
        turn_id=f"turn-{created_at.timestamp()}-{len(question)}",
        question=question,
        result_classification=ResultClassification.CANONICAL_DEFINITION,
        metric_name=metric_name,
        trace_id=f"trace-{created_at.timestamp()}",
        created_at=created_at,
    )


def test_sqlite_checkpoint_survives_restart_without_raw_evidence_chunks(tmp_path: Path) -> None:
    database_path = tmp_path / "conversations.sqlite3"
    recorded_at = datetime(2026, 8, 30, tzinfo=UTC)
    summary = ConversationSummary(
        agent_user_goal="Understand Jira New PEU",
        resolved_scope=EffectiveAccessScope(
            products=["Jira"],
            regions=["Americas", "APAC", "EMEA"],
            tenant_scope="All permitted Jira Tenants",
            permitted_columns=["metric_name", "definition"],
        ),
        metric_name="jira_new_peu",
        evidence_revision_ids=["incident@revision-7"],
        qualified_conclusions=["canonical_definition"],
        workflow_state="canonical_definition",
    )
    first = SQLiteConversationCheckpointStore(
        database_path,
        now=lambda: recorded_at,
    )
    checkpoint = first.create(_principal())
    first.append(
        checkpoint.conversation_id,
        _principal(),
        turn=_turn(created_at=recorded_at, question="What is Jira New PEU?"),
        summary=summary,
    )
    with sqlite3.connect(database_path) as connection:
        stored_summary = connection.execute(
            "SELECT summary_json FROM conversation_checkpoints"
        ).fetchone()[0]
    assert "What is Jira New PEU?" not in stored_summary

    restarted = SQLiteConversationCheckpointStore(
        database_path,
        now=lambda: recorded_at,
    )
    loaded = restarted.load(checkpoint.conversation_id, _principal())

    assert loaded.summary == summary
    assert loaded.recent_turns[0].question == "What is Jira New PEU?"
    assert b"raw evidence chunk" not in database_path.read_bytes()


def test_service_requires_a_verified_principal_at_the_conversation_boundary(
    client: TestClient,
) -> None:
    with pytest.raises(AccessDeniedError):
        client.app.state.answer_service.answer_question(
            AnswerQuestionRequest(agent_user_id="data_analyst", question="What is Jira New PEU?")
        )


def test_checkpoint_owner_binding_includes_issuer_and_subject(tmp_path: Path) -> None:
    store = SQLiteConversationCheckpointStore(tmp_path / "conversations.sqlite3")
    checkpoint = store.create(_principal())
    same_id_different_identity = VerifiedPrincipal(
        principal_id="data_analyst",
        issuer="https://another-issuer.example.test",
        subject="subject-123",
    )

    with pytest.raises(ConversationAccessDeniedError):
        store.load(checkpoint.conversation_id, same_id_different_identity)


def test_recent_context_is_bounded_by_tokens_and_transcript_retention(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 8, 1, tzinfo=UTC)
    current_time = [recorded_at]
    store = InMemoryConversationCheckpointStore(
        retention=timedelta(days=30),
        now=lambda: current_time[0],
        recent_context_token_budget=4,
    )
    checkpoint = store.create(_principal())
    store.append(
        checkpoint.conversation_id,
        _principal(),
        turn=_turn(created_at=recorded_at, question="one two three"),
        summary=ConversationSummary(metric_name="jira_new_peu"),
    )
    store.append(
        checkpoint.conversation_id,
        _principal(),
        turn=_turn(created_at=recorded_at + timedelta(seconds=1), question="four five six"),
        summary=ConversationSummary(metric_name="jira_new_peu"),
    )

    loaded = store.load(checkpoint.conversation_id, _principal())
    assert [turn.question for turn in loaded.recent_turns] == ["four five six"]

    current_time[0] = recorded_at + timedelta(days=31)
    assert store.transcript(checkpoint.conversation_id, _principal()) == ()


def test_trace_is_linked_to_but_distinct_from_conversation(client: TestClient) -> None:
    class RecordingTraceSink:
        def __init__(self) -> None:
            self.trace = None

        def record(self, trace) -> None:
            self.trace = trace

    sink = RecordingTraceSink()
    client.app.state.answer_service.trace_sink = sink

    response = _answer(client, question="What is Jira New PEU?")

    assert response.status_code == 200
    assert sink.trace is not None
    assert sink.trace.conversation_id == response.json()["conversation_id"]
    assert sink.trace.trace_id == response.json()["trace_id"]
    assert sink.trace.trace_id != sink.trace.conversation_id

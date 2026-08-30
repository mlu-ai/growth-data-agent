from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from growth_data_agent import audit, metric_definition_gaps
from growth_data_agent.contracts import EffectiveAccessScope, VerificationRequestConfirmation
from growth_data_agent.main import create_app
from growth_data_agent.persistence import DEFAULT_DECISION_RECORD_RETENTION


def _scope() -> EffectiveAccessScope:
    return EffectiveAccessScope(
        products=["Jira"],
        regions=["APAC"],
        tenant_scope="APAC Tenants only",
        permitted_columns=["tenant_id"],
    )


def _audit_recorder(path: Path, *, now, **kwargs):
    recorder_type = getattr(audit, "SQLiteDirectIdentifierAuditRecorder", None)
    assert recorder_type is not None, "durable identifier audit recorder is required"
    return recorder_type(path, now=now, **kwargs)


def _verification_recorder(path: Path, *, now, **kwargs):
    recorder_type = getattr(
        metric_definition_gaps,
        "SQLiteDataTeamVerificationRequestRecorder",
        None,
    )
    assert recorder_type is not None, "durable verification request recorder is required"
    return recorder_type(path, now=now, **kwargs)


def test_identifier_release_audit_survives_recorder_restart_without_raw_values(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "decision-records.sqlite3"
    identifier = "tenant-0099"
    source_body = "The source page body must never be persisted."
    recorded_at = datetime(2026, 8, 25, 1, tzinfo=UTC)

    first = _audit_recorder(database_path, now=lambda: recorded_at)
    event = first.record(
        trace_id="trace-123",
        agent_user_id="customer_success_manager",
        scope=_scope(),
        policy_fingerprint="policy-abc",
        outcome="released",
        returned_count=1,
        maximum_results=3,
    )

    restarted = _audit_recorder(database_path, now=lambda: recorded_at)

    assert restarted.events == [event]
    assert event.agent_user_id == "customer_success_manager"
    assert event.scope == _scope()
    assert event.policy_fingerprint == "policy-abc"
    assert event.outcome == "released"
    assert event.trace_id == "trace-123"
    assert identifier.encode() not in database_path.read_bytes()
    assert source_body.encode() not in database_path.read_bytes()


def test_identifier_release_audit_prunes_records_older_than_configured_retention(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "decision-records.sqlite3"
    old_time = datetime(2025, 8, 24, tzinfo=UTC)
    current_time = datetime(2026, 8, 25, tzinfo=UTC)

    first = _audit_recorder(database_path, now=lambda: old_time)
    first.record(
        trace_id="old-trace",
        agent_user_id="customer_success_manager",
        scope=_scope(),
        policy_fingerprint="policy-abc",
        outcome="released",
        returned_count=1,
        maximum_results=3,
    )

    recorder_type = type(first)
    retained = recorder_type(
        database_path,
        now=lambda: current_time,
        retention=timedelta(days=30),
    )
    retained.record(
        trace_id="new-trace",
        agent_user_id="customer_success_manager",
        scope=_scope(),
        policy_fingerprint="policy-abc",
        outcome="released",
        returned_count=1,
        maximum_results=3,
    )

    assert [event.trace_id for event in retained.events] == ["new-trace"]


def test_verification_request_survives_restart_with_principal_and_decision_outcome(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "decision-records.sqlite3"
    recorded_at = datetime(2026, 8, 25, 2, tzinfo=UTC)
    confirmation = VerificationRequestConfirmation(
        approved=True,
        approval_context=(
            "The Agent User approved review; do not retain this source-page body: "
            "tenant-0099 details."
        ),
    )

    first = _verification_recorder(database_path, now=lambda: recorded_at)
    request = first.record(
        metric_name="jira_activation",
        agent_user_id="data_analyst",
        trace_id="trace-456",
        confirmation=confirmation,
    )

    restarted = _verification_recorder(database_path, now=lambda: recorded_at)

    assert [item.request_id for item in restarted.requests] == [request.request_id]
    persisted = restarted.requests[0]
    assert persisted.requested_metric_name == "jira_activation"
    assert persisted.requested_by_agent_user_id == "data_analyst"
    assert persisted.decision_outcome == "approved"
    assert persisted.trace_id == "trace-456"
    assert persisted.approval_context_sha256 == hashlib.sha256(
        confirmation.approval_context.encode("utf-8")
    ).hexdigest()
    database_bytes = database_path.read_bytes()
    assert b"tenant-0099" not in database_bytes
    assert b"source-page body" not in database_bytes


def test_durable_recorders_default_to_twelve_month_retention(tmp_path: Path) -> None:
    database_path = tmp_path / "decision-records.sqlite3"

    recorder = _audit_recorder(database_path, now=lambda: datetime.now(UTC))

    assert recorder.retention == DEFAULT_DECISION_RECORD_RETENTION == timedelta(days=365)


def test_startup_prunes_expired_rows_from_both_decision_record_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "decision-records.sqlite3"
    old_time = datetime(2025, 8, 24, tzinfo=UTC)
    current_time = datetime(2026, 8, 25, tzinfo=UTC)

    _audit_recorder(database_path, now=lambda: old_time).record(
        trace_id="old-audit",
        agent_user_id="customer_success_manager",
        scope=_scope(),
        policy_fingerprint="policy-abc",
        outcome="released",
        returned_count=1,
        maximum_results=3,
    )
    _verification_recorder(database_path, now=lambda: old_time).record(
        metric_name="jira_activation",
        agent_user_id="data_analyst",
        trace_id="old-request",
        confirmation=VerificationRequestConfirmation(
            approved=True,
            approval_context="approved old request",
        ),
    )

    _audit_recorder(
        database_path,
        now=lambda: current_time,
        retention=timedelta(days=30),
    )

    with sqlite3.connect(database_path) as connection:
        audit_count = connection.execute(
            "SELECT count(*) FROM identifier_release_audit_events"
        ).fetchone()[0]
        request_count = connection.execute(
            "SELECT count(*) FROM metric_definition_gap_verification_requests"
        ).fetchone()[0]
    assert audit_count == request_count == 0


def test_runtime_app_wires_configured_durable_recorders(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime-records.sqlite3"
    monkeypatch.setenv("GROWTH_DATA_AGENT_DECISION_RECORDS_PATH", str(database_path))
    monkeypatch.setenv("GROWTH_DATA_AGENT_AUDIT_RETENTION_DAYS", "30")

    app = create_app()
    service = app.state.answer_service

    assert service.direct_identifier_audit_recorder.path == database_path
    assert service.verification_request_recorder.path == database_path
    assert service.direct_identifier_audit_recorder.retention == timedelta(days=30)
    assert service.verification_request_recorder.retention == timedelta(days=30)

"""Audit the bounded release of direct identifiers without storing their values."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from .contracts import DirectIdentifierAudit, EffectiveAccessScope
from .persistence import DEFAULT_DECISION_RECORD_RETENTION, SQLiteDecisionRecordStore

IdentifierReleaseOutcome = Literal["released", "no_identifiers_found"]


class DirectIdentifierAuditRecorder(Protocol):
    def record(
        self,
        *,
        trace_id: str,
        agent_user_id: str,
        scope: EffectiveAccessScope,
        policy_fingerprint: str,
        outcome: IdentifierReleaseOutcome,
        returned_count: int,
        maximum_results: int,
    ) -> DirectIdentifierAudit: ...


class InMemoryDirectIdentifierAuditRecorder:
    """POC audit sink; events deliberately exclude raw identifier values."""

    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self.now = now or (lambda: datetime.now(UTC))
        self.events: list[DirectIdentifierAudit] = []

    def record(
        self,
        *,
        trace_id: str,
        agent_user_id: str,
        scope: EffectiveAccessScope,
        policy_fingerprint: str,
        outcome: IdentifierReleaseOutcome,
        returned_count: int,
        maximum_results: int,
    ) -> DirectIdentifierAudit:
        event = DirectIdentifierAudit(
            audit_event_id=str(uuid4()),
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            scope=scope,
            policy_fingerprint=policy_fingerprint,
            outcome=outcome,
            recorded_at=self.now().astimezone(UTC),
            returned_count=returned_count,
            maximum_results=maximum_results,
        )
        self.events.append(event)
        return event


class SQLiteDirectIdentifierAuditRecorder:
    """Persist identifier-release metadata while excluding returned values."""

    _TABLE = "identifier_release_audit_events"

    def __init__(
        self,
        path: str | Path,
        *,
        retention=DEFAULT_DECISION_RECORD_RETENTION,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = SQLiteDecisionRecordStore(path, retention=retention, now=now)

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def retention(self):
        return self._store.retention

    @property
    def events(self) -> list[DirectIdentifierAudit]:
        self._store.prune_expired()
        with self._store.connection() as connection:
            rows = connection.execute(
                """
                SELECT audit_event_id, trace_id, agent_user_id, scope_json,
                       policy_fingerprint, outcome, returned_count,
                       maximum_results, recorded_at
                FROM identifier_release_audit_events
                ORDER BY recorded_at, audit_event_id
                """
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def record(
        self,
        *,
        trace_id: str,
        agent_user_id: str,
        scope: EffectiveAccessScope,
        policy_fingerprint: str,
        outcome: IdentifierReleaseOutcome,
        returned_count: int,
        maximum_results: int,
    ) -> DirectIdentifierAudit:
        event = DirectIdentifierAudit(
            audit_event_id=str(uuid4()),
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            scope=scope,
            policy_fingerprint=policy_fingerprint,
            outcome=outcome,
            recorded_at=self._store.now_utc(),
            returned_count=returned_count,
            maximum_results=maximum_results,
        )
        with self._store.transaction() as connection:
            self._store.prune_in_transaction(
                connection,
                table=self._TABLE,
                timestamp_column="recorded_at",
            )
            connection.execute(
                """
                INSERT INTO identifier_release_audit_events (
                    audit_event_id, trace_id, agent_user_id, scope_json,
                    policy_fingerprint, outcome, returned_count,
                    maximum_results, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.audit_event_id,
                    event.trace_id,
                    event.agent_user_id,
                    json.dumps(event.scope.model_dump(mode="json"), sort_keys=True),
                    event.policy_fingerprint,
                    event.outcome,
                    event.returned_count,
                    event.maximum_results,
                    event.recorded_at.isoformat(),
                ),
            )
        return event

    @staticmethod
    def _event_from_row(row) -> DirectIdentifierAudit:
        return DirectIdentifierAudit(
            audit_event_id=row["audit_event_id"],
            trace_id=row["trace_id"],
            agent_user_id=row["agent_user_id"],
            scope=EffectiveAccessScope.model_validate(json.loads(row["scope_json"])),
            policy_fingerprint=row["policy_fingerprint"],
            outcome=row["outcome"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            returned_count=row["returned_count"],
            maximum_results=row["maximum_results"],
        )


SqliteDirectIdentifierAuditRecorder = SQLiteDirectIdentifierAuditRecorder

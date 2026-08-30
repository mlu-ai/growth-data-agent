"""Small SQLite primitives shared by durable governed decision recorders."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_DECISION_RECORD_RETENTION = timedelta(days=365)
_DEFAULT_RETENTION_DAYS = str(DEFAULT_DECISION_RECORD_RETENTION.days)


def decision_record_path_from_environment(default: Path) -> Path:
    """Resolve the local durable record database path without accepting source data."""
    configured = os.environ.get("GROWTH_DATA_AGENT_DECISION_RECORDS_PATH")
    return Path(configured) if configured else default


def decision_record_retention_from_environment() -> timedelta:
    """Resolve the retention window, defaulting to the agreed twelve months."""
    configured = os.environ.get("GROWTH_DATA_AGENT_AUDIT_RETENTION_DAYS")
    configured = configured or _DEFAULT_RETENTION_DAYS
    try:
        retention_days = int(configured)
    except ValueError as error:
        raise ValueError("Audit retention days must be a positive integer.") from error
    if retention_days <= 0:
        raise ValueError("Audit retention days must be a positive integer.")
    return timedelta(days=retention_days)


class SQLiteDecisionRecordStore:
    """Own the SQLite database lifecycle without storing source or identifier values."""

    def __init__(
        self,
        path: str | Path,
        *,
        retention: timedelta = DEFAULT_DECISION_RECORD_RETENTION,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if retention <= timedelta(0):
            raise ValueError("Decision record retention must be positive.")
        self.path = Path(path)
        self.retention = retention
        self.now = now or (lambda: datetime.now(UTC))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.prune_expired()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def now_utc(self) -> datetime:
        value = self.now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def prune_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        timestamp_column: str,
    ) -> None:
        cutoff = self.now_utc() - self.retention
        connection.execute(
            f"DELETE FROM {table} WHERE {timestamp_column} < ?",
            (cutoff.isoformat(),),
        )

    def prune_expired(self) -> None:
        """Apply retention at startup so stale rows do not remain on disk."""
        with self.transaction() as connection:
            self.prune_in_transaction(
                connection,
                table="identifier_release_audit_events",
                timestamp_column="recorded_at",
            )
            self.prune_in_transaction(
                connection,
                table="metric_definition_gap_verification_requests",
                timestamp_column="approved_at",
            )

    def _initialize(self) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identifier_release_audit_events (
                    audit_event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    agent_user_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    returned_count INTEGER NOT NULL CHECK (returned_count >= 0),
                    maximum_results INTEGER NOT NULL CHECK (maximum_results > 0),
                    recorded_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_definition_gap_verification_requests (
                    request_id TEXT PRIMARY KEY,
                    requested_metric_name TEXT NOT NULL,
                    requested_by_agent_user_id TEXT NOT NULL,
                    decision_outcome TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    approval_context_sha256 TEXT NOT NULL,
                    approved_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_identifier_release_audit_recorded_at
                ON identifier_release_audit_events(recorded_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_verification_requests_approved_at
                ON metric_definition_gap_verification_requests(approved_at)
                """
            )

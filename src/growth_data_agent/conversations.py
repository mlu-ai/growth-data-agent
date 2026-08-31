"""Private, durable conversation checkpoints and bounded working context."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import (
    ConversationContext,
    ConversationSummary,
    ConversationTurn,
    EffectiveAccessScope,
    LeadAgentMetadata,
)
from .principal import VerifiedPrincipal

DEFAULT_CONVERSATION_RETENTION = timedelta(days=30)
_DEFAULT_RETENTION_DAYS = str(DEFAULT_CONVERSATION_RETENTION.days)
_DEFAULT_RECENT_CONTEXT_TOKEN_BUDGET = 512
_CONVERSATION_ID_BYTES = 32


class ConversationNotFoundError(ValueError):
    """Raised when a requested server-owned conversation does not exist."""


class ConversationAccessDeniedError(ValueError):
    """Raised without revealing whether another Principal owns a conversation."""


class ConversationCheckpointStore(Protocol):
    """Durable ownership and context storage for one private Conversation."""

    def create(self, principal: VerifiedPrincipal) -> ConversationCheckpoint: ...

    def load(
        self, conversation_id: str, principal: VerifiedPrincipal
    ) -> ConversationCheckpoint: ...

    def append(
        self,
        conversation_id: str,
        principal: VerifiedPrincipal,
        *,
        turn: ConversationTurn,
        summary: ConversationSummary,
    ) -> ConversationCheckpoint: ...


class ConversationCheckpoint:
    """The safe checkpoint projection supplied to a new Turn."""

    def __init__(
        self,
        *,
        conversation_id: str,
        principal: VerifiedPrincipal,
        summary: ConversationSummary,
        recent_turns: Sequence[ConversationTurn],
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        self.conversation_id = conversation_id
        self.principal = principal
        self.summary = summary
        self.recent_turns = tuple(recent_turns)
        self.created_at = _as_utc(created_at)
        self.updated_at = _as_utc(updated_at)

    def context(
        self, *, effective_scope: EffectiveAccessScope | None = None
    ) -> ConversationContext:
        summary = (
            self.summary.model_copy(update={"resolved_scope": effective_scope})
            if effective_scope is not None
            else self.summary
        )
        return ConversationContext(summary=summary, recent_turns=list(self.recent_turns))


def conversation_retention_from_environment() -> timedelta:
    """Resolve configurable raw-transcript retention, defaulting to thirty days."""
    configured = os.environ.get("GROWTH_DATA_AGENT_CONVERSATION_RETENTION_DAYS")
    try:
        days = int(configured or _DEFAULT_RETENTION_DAYS)
    except ValueError as error:
        raise ValueError("Conversation retention days must be a positive integer.") from error
    if days <= 0:
        raise ValueError("Conversation retention days must be a positive integer.")
    return timedelta(days=days)


def _new_conversation_id() -> str:
    """Issue an unguessable ID that contains no Principal or domain meaning."""
    return secrets.token_urlsafe(_CONVERSATION_ID_BYTES)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _principal_matches(left: VerifiedPrincipal, right: VerifiedPrincipal) -> bool:
    return (
        left.principal_id == right.principal_id
        and left.issuer == right.issuer
        and left.subject == right.subject
    )


def _estimate_tokens(turn: ConversationTurn) -> int:
    return max(1, len(turn.question.split()))


def bounded_recent_turns(
    turns: Sequence[ConversationTurn], *, token_budget: int
) -> tuple[ConversationTurn, ...]:
    """Return the newest complete turns that fit a token budget, not a turn count."""
    if token_budget <= 0:
        raise ValueError("Recent conversation context token budget must be positive.")
    selected: list[ConversationTurn] = []
    used = 0
    for turn in reversed(turns):
        tokens = _estimate_tokens(turn)
        if used + tokens > token_budget:
            break
        selected.append(turn)
        used += tokens
    selected.reverse()
    return tuple(selected)


class InMemoryConversationCheckpointStore:
    """Fast test double with the same ownership and retention semantics as Postgres."""

    def __init__(
        self,
        *,
        retention: timedelta = DEFAULT_CONVERSATION_RETENTION,
        now: Callable[[], datetime] | None = None,
        recent_context_token_budget: int = _DEFAULT_RECENT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        _validate_store_options(retention, recent_context_token_budget)
        self.retention = retention
        self.now = now or (lambda: datetime.now(UTC))
        self.recent_context_token_budget = recent_context_token_budget
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def create(self, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        now = _as_utc(self.now())
        record = {
            "principal": principal,
            "summary": ConversationSummary(),
            "transcript": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            conversation_id = _new_conversation_id()
            self._records[conversation_id] = record
        return self._checkpoint(conversation_id, record)

    def load(self, conversation_id: str, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        with self._lock:
            record = self._record_for_owner(conversation_id, principal)
            self._prune(record)
            return self._checkpoint(conversation_id, record)

    def append(
        self,
        conversation_id: str,
        principal: VerifiedPrincipal,
        *,
        turn: ConversationTurn,
        summary: ConversationSummary,
    ) -> ConversationCheckpoint:
        with self._lock:
            record = self._record_for_owner(conversation_id, principal)
            self._prune(record)
            record["transcript"].append(turn)
            record["summary"] = summary
            record["updated_at"] = _as_utc(self.now())
            return self._checkpoint(conversation_id, record)

    def transcript(
        self, conversation_id: str, principal: VerifiedPrincipal
    ) -> tuple[ConversationTurn, ...]:
        with self._lock:
            record = self._record_for_owner(conversation_id, principal)
            self._prune(record)
            return tuple(record["transcript"])

    def _record_for_owner(
        self, conversation_id: str, principal: VerifiedPrincipal
    ) -> dict[str, Any]:
        record = self._records.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError("Conversation is not available.")
        if not _principal_matches(record["principal"], principal):
            raise ConversationAccessDeniedError("Conversation is not available to this Principal.")
        return record

    def _prune(self, record: dict[str, Any]) -> None:
        cutoff = _as_utc(self.now()) - self.retention
        record["transcript"] = [turn for turn in record["transcript"] if turn.created_at >= cutoff]

    def _checkpoint(
        self, conversation_id: str, record: Mapping[str, Any]
    ) -> ConversationCheckpoint:
        return ConversationCheckpoint(
            conversation_id=conversation_id,
            principal=record["principal"],
            summary=record["summary"],
            recent_turns=bounded_recent_turns(
                record["transcript"], token_budget=self.recent_context_token_budget
            ),
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )


class SQLiteConversationCheckpointStore:
    """Restartable local store used by tests and offline POC runs."""

    def __init__(
        self,
        path: str | Path,
        *,
        retention: timedelta = DEFAULT_CONVERSATION_RETENTION,
        now: Callable[[], datetime] | None = None,
        recent_context_token_budget: int = _DEFAULT_RECENT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        _validate_store_options(retention, recent_context_token_budget)
        self.path = Path(path)
        self.retention = retention
        self.now = now or (lambda: datetime.now(UTC))
        self.recent_context_token_budget = recent_context_token_budget
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_checkpoints (
                    conversation_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    principal_issuer TEXT NOT NULL,
                    principal_subject TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    recent_turns_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_transcript (
                    conversation_id TEXT NOT NULL
                        REFERENCES conversation_checkpoints(conversation_id),
                    turn_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    result_classification TEXT NOT NULL,
                    metric_name TEXT,
                    trace_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lead_agent_metadata_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_transcript_created_at "
                "ON conversation_transcript(conversation_id, created_at)"
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(conversation_transcript)"
                ).fetchall()
            }
            if "lead_agent_metadata_json" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_transcript ADD COLUMN lead_agent_metadata_json TEXT"
                )

    def create(self, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        conversation_id = _new_conversation_id()
        now = _as_utc(self.now())
        summary = ConversationSummary()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO conversation_checkpoints (
                    conversation_id, principal_id, principal_issuer, principal_subject,
                    summary_json, recent_turns_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    principal.principal_id,
                    principal.issuer,
                    principal.subject,
                    _json(summary.model_dump(mode="json")),
                    "[]",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.commit()
        return ConversationCheckpoint(
            conversation_id=conversation_id,
            principal=principal,
            summary=summary,
            recent_turns=(),
            created_at=now,
            updated_at=now,
        )

    def load(self, conversation_id: str, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned_row(connection, conversation_id, principal)
            self._prune(connection, conversation_id)
            transcript = self._transcript_rows(connection, conversation_id)
            recent = bounded_recent_turns(transcript, token_budget=self.recent_context_token_budget)
            self._update_recent(connection, conversation_id, recent)
            connection.commit()
        return self._checkpoint_from_row(row, recent)

    def append(
        self,
        conversation_id: str,
        principal: VerifiedPrincipal,
        *,
        turn: ConversationTurn,
        summary: ConversationSummary,
    ) -> ConversationCheckpoint:
        now = _as_utc(self.now())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_row(connection, conversation_id, principal)
            self._prune(connection, conversation_id)
            connection.execute(
                """
                INSERT INTO conversation_transcript (
                    conversation_id, turn_id, question, result_classification,
                    metric_name, trace_id, created_at, lead_agent_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn.turn_id,
                    turn.question,
                    turn.result_classification.value,
                    turn.metric_name,
                    turn.trace_id,
                    turn.created_at.isoformat(),
                    (
                        _json(turn.lead_agent_metadata.model_dump(mode="json"))
                        if turn.lead_agent_metadata is not None
                        else None
                    ),
                ),
            )
            recent = bounded_recent_turns(
                self._transcript_rows(connection, conversation_id),
                token_budget=self.recent_context_token_budget,
            )
            connection.execute(
                """
                UPDATE conversation_checkpoints
                SET summary_json = ?, recent_turns_json = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    _json(summary.model_dump(mode="json")),
                    _json([item.model_dump(mode="json") for item in recent]),
                    now.isoformat(),
                    conversation_id,
                ),
            )
            row = self._owned_row(connection, conversation_id, principal)
            connection.commit()
        return self._checkpoint_from_row(row, recent)

    def transcript(
        self, conversation_id: str, principal: VerifiedPrincipal
    ) -> tuple[ConversationTurn, ...]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_row(connection, conversation_id, principal)
            self._prune(connection, conversation_id)
            turns = tuple(self._transcript_rows(connection, conversation_id))
            recent = bounded_recent_turns(turns, token_budget=self.recent_context_token_budget)
            self._update_recent(connection, conversation_id, recent)
            connection.commit()
        return turns

    def _owned_row(self, connection, conversation_id: str, principal: VerifiedPrincipal):
        row = connection.execute(
            "SELECT * FROM conversation_checkpoints WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError("Conversation is not available.")
        owner = VerifiedPrincipal(
            principal_id=row["principal_id"],
            issuer=row["principal_issuer"],
            subject=row["principal_subject"],
        )
        if not _principal_matches(owner, principal):
            raise ConversationAccessDeniedError("Conversation is not available to this Principal.")
        return row

    def _prune(self, connection, conversation_id: str) -> None:
        cutoff = (_as_utc(self.now()) - self.retention).isoformat()
        connection.execute(
            "DELETE FROM conversation_transcript WHERE conversation_id = ? AND created_at < ?",
            (conversation_id, cutoff),
        )

    def _transcript_rows(self, connection, conversation_id: str) -> list[ConversationTurn]:
        rows = connection.execute(
            """
            SELECT turn_id, question, result_classification, metric_name, trace_id,
                   created_at, lead_agent_metadata_json
            FROM conversation_transcript
            WHERE conversation_id = ?
            ORDER BY created_at, turn_id
            """,
            (conversation_id,),
        ).fetchall()
        return [_turn_from_row(row) for row in rows]

    @staticmethod
    def _update_recent(
        connection, conversation_id: str, recent: Sequence[ConversationTurn]
    ) -> None:
        connection.execute(
            "UPDATE conversation_checkpoints SET recent_turns_json = ? WHERE conversation_id = ?",
            (_json([item.model_dump(mode="json") for item in recent]), conversation_id),
        )

    def _checkpoint_from_row(
        self, row, recent: Sequence[ConversationTurn]
    ) -> ConversationCheckpoint:
        return ConversationCheckpoint(
            conversation_id=row["conversation_id"],
            principal=VerifiedPrincipal(
                principal_id=row["principal_id"],
                issuer=row["principal_issuer"],
                subject=row["principal_subject"],
            ),
            summary=ConversationSummary.model_validate(json.loads(row["summary_json"])),
            recent_turns=recent,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class PostgresConversationCheckpointStore:
    """Production-shaped Postgres checkpoint store with a LangGraph thread saver."""

    def __init__(
        self,
        database_url: str,
        *,
        retention: timedelta = DEFAULT_CONVERSATION_RETENTION,
        now: Callable[[], datetime] | None = None,
        recent_context_token_budget: int = _DEFAULT_RECENT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        _validate_store_options(retention, recent_context_token_budget)
        if not database_url.strip():
            raise ValueError("Conversation database URL must not be blank.")
        self.database_url = database_url
        self.retention = retention
        self.now = now or (lambda: datetime.now(UTC))
        self.recent_context_token_budget = recent_context_token_budget
        self._checkpointer = LazyPostgresCheckpointer(database_url)

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    def create(self, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        conversation_id = _new_conversation_id()
        now = _as_utc(self.now())
        summary = ConversationSummary()
        with self._connection() as connection:
            self._initialize(connection)
            connection.execute(
                """
                INSERT INTO growth_agent_conversation_checkpoints (
                    conversation_id, principal_id, principal_issuer, principal_subject,
                    summary_json, recent_turns_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    principal.principal_id,
                    principal.issuer,
                    principal.subject,
                    Jsonb(summary.model_dump(mode="json")),
                    Jsonb([]),
                    now,
                    now,
                ),
            )
        return ConversationCheckpoint(
            conversation_id=conversation_id,
            principal=principal,
            summary=summary,
            recent_turns=(),
            created_at=now,
            updated_at=now,
        )

    def load(self, conversation_id: str, principal: VerifiedPrincipal) -> ConversationCheckpoint:
        with self._connection() as connection:
            self._initialize(connection)
            with connection.transaction():
                row = self._owned_row(connection, conversation_id, principal)
                self._prune(connection, conversation_id)
                recent = bounded_recent_turns(
                    self._transcript_rows(connection, conversation_id),
                    token_budget=self.recent_context_token_budget,
                )
                self._update_recent(connection, conversation_id, recent)
        return self._checkpoint_from_row(row, recent)

    def append(
        self,
        conversation_id: str,
        principal: VerifiedPrincipal,
        *,
        turn: ConversationTurn,
        summary: ConversationSummary,
    ) -> ConversationCheckpoint:
        now = _as_utc(self.now())
        with self._connection() as connection:
            self._initialize(connection)
            with connection.transaction():
                self._owned_row(connection, conversation_id, principal)
                self._prune(connection, conversation_id)
                connection.execute(
                    """
                    INSERT INTO growth_agent_conversation_transcript (
                        conversation_id, turn_id, question, result_classification,
                        metric_name, trace_id, created_at, lead_agent_metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        turn.turn_id,
                        turn.question,
                        turn.result_classification.value,
                        turn.metric_name,
                        turn.trace_id,
                        turn.created_at,
                        (
                            Jsonb(turn.lead_agent_metadata.model_dump(mode="json"))
                            if turn.lead_agent_metadata is not None
                            else None
                        ),
                    ),
                )
                recent = bounded_recent_turns(
                    self._transcript_rows(connection, conversation_id),
                    token_budget=self.recent_context_token_budget,
                )
                connection.execute(
                    """
                    UPDATE growth_agent_conversation_checkpoints
                    SET summary_json = %s, recent_turns_json = %s, updated_at = %s
                    WHERE conversation_id = %s
                    """,
                    (
                        Jsonb(summary.model_dump(mode="json")),
                        Jsonb([item.model_dump(mode="json") for item in recent]),
                        now,
                        conversation_id,
                    ),
                )
                row = self._owned_row(connection, conversation_id, principal)
        return self._checkpoint_from_row(row, recent)

    def transcript(
        self, conversation_id: str, principal: VerifiedPrincipal
    ) -> tuple[ConversationTurn, ...]:
        with self._connection() as connection:
            self._initialize(connection)
            with connection.transaction():
                self._owned_row(connection, conversation_id, principal)
                self._prune(connection, conversation_id)
                turns = tuple(self._transcript_rows(connection, conversation_id))
                recent = bounded_recent_turns(turns, token_budget=self.recent_context_token_budget)
                self._update_recent(connection, conversation_id, recent)
                return turns

    @contextmanager
    def _connection(self):
        with psycopg.connect(
            self.database_url,
            row_factory=cast(Any, dict_row),
        ) as connection:
            yield connection

    @staticmethod
    def _initialize(connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS growth_agent_conversation_checkpoints (
                conversation_id TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL,
                principal_issuer TEXT NOT NULL,
                principal_subject TEXT NOT NULL,
                summary_json JSONB NOT NULL,
                recent_turns_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS growth_agent_conversation_transcript (
                conversation_id TEXT NOT NULL
                    REFERENCES growth_agent_conversation_checkpoints(conversation_id),
                turn_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                result_classification TEXT NOT NULL,
                metric_name TEXT,
                trace_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                lead_agent_metadata_json JSONB
            )
            """
        )
        connection.execute(
            "ALTER TABLE growth_agent_conversation_transcript "
            "ADD COLUMN IF NOT EXISTS lead_agent_metadata_json JSONB"
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_growth_agent_conversation_transcript_created_at
            ON growth_agent_conversation_transcript(conversation_id, created_at)
            """
        )

    def _owned_row(self, connection, conversation_id: str, principal: VerifiedPrincipal):
        row = connection.execute(
            "SELECT * FROM growth_agent_conversation_checkpoints WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError("Conversation is not available.")
        owner = VerifiedPrincipal(
            principal_id=row["principal_id"],
            issuer=row["principal_issuer"],
            subject=row["principal_subject"],
        )
        if not _principal_matches(owner, principal):
            raise ConversationAccessDeniedError("Conversation is not available to this Principal.")
        return row

    def _prune(self, connection, conversation_id: str) -> None:
        connection.execute(
            "DELETE FROM growth_agent_conversation_transcript "
            "WHERE conversation_id = %s AND created_at < %s",
            (conversation_id, _as_utc(self.now()) - self.retention),
        )

    def _transcript_rows(self, connection, conversation_id: str) -> list[ConversationTurn]:
        rows = connection.execute(
            """
            SELECT turn_id, question, result_classification, metric_name, trace_id,
                   created_at, lead_agent_metadata_json
            FROM growth_agent_conversation_transcript
            WHERE conversation_id = %s
            ORDER BY created_at, turn_id
            """,
            (conversation_id,),
        ).fetchall()
        return [_turn_from_row(row) for row in rows]

    @staticmethod
    def _update_recent(
        connection, conversation_id: str, recent: Sequence[ConversationTurn]
    ) -> None:
        connection.execute(
            "UPDATE growth_agent_conversation_checkpoints "
            "SET recent_turns_json = %s WHERE conversation_id = %s",
            (Jsonb([item.model_dump(mode="json") for item in recent]), conversation_id),
        )

    def _checkpoint_from_row(
        self, row, recent: Sequence[ConversationTurn]
    ) -> ConversationCheckpoint:
        return ConversationCheckpoint(
            conversation_id=row["conversation_id"],
            principal=VerifiedPrincipal(
                principal_id=row["principal_id"],
                issuer=row["principal_issuer"],
                subject=row["principal_subject"],
            ),
            summary=ConversationSummary.model_validate(row["summary_json"]),
            recent_turns=recent,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class SafePostgresSaver(PostgresSaver):
    """Persist only the context channel, never request/response/source payloads."""

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        safe_checkpoint = checkpoint.copy()
        context = checkpoint.get("channel_values", {}).get("conversation_context")
        safe_checkpoint["channel_values"] = (
            {"conversation_context": context} if context is not None else {}
        )
        safe_metadata: CheckpointMetadata = {}
        if "source" in metadata:
            safe_metadata["source"] = metadata["source"]
        if "step" in metadata:
            safe_metadata["step"] = metadata["step"]
        if "parents" in metadata:
            safe_metadata["parents"] = metadata["parents"]
        return super().put(config, safe_checkpoint, safe_metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del config, writes, task_id, task_path


class LazyPostgresCheckpointer(BaseCheckpointSaver):
    """Open a short-lived Postgres saver per LangGraph operation."""

    def __init__(self, database_url: str) -> None:
        super().__init__()
        self.database_url = database_url

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return cast(
            CheckpointTuple | None, self._dispatch_checkpointer_operation("get_tuple", config)
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return cast(
            RunnableConfig,
            self._dispatch_checkpointer_operation(
                "put", config, checkpoint, metadata, new_versions
            ),
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._dispatch_checkpointer_operation("put_writes", config, writes, task_id, task_path)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return iter(
            cast(
                list[CheckpointTuple],
                self._dispatch_checkpointer_operation(
                    "list", config, filter=filter, before=before, limit=limit
                ),
            )
        )

    def delete_thread(self, thread_id: str) -> None:
        self._dispatch_checkpointer_operation("delete_thread", thread_id)

    def _dispatch_checkpointer_operation(self, method: str, *args, **kwargs):
        with psycopg.connect(
            self.database_url,
            autocommit=True,
            row_factory=cast(Any, dict_row),
        ) as connection:
            saver = SafePostgresSaver(cast(Any, connection))
            saver.setup()
            result = getattr(saver, method)(*args, **kwargs)
            if method == "list":
                return list(result)
            return result


def _validate_store_options(retention: timedelta, token_budget: int) -> None:
    if retention <= timedelta(0):
        raise ValueError("Conversation retention must be positive.")
    if token_budget <= 0:
        raise ValueError("Recent conversation context token budget must be positive.")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _turn_from_row(row: Mapping[str, Any]) -> ConversationTurn:
    try:
        metadata_json = row["lead_agent_metadata_json"]
    except (KeyError, IndexError):
        metadata_json = None
    if isinstance(metadata_json, str):
        metadata_json = json.loads(metadata_json)
    return ConversationTurn(
        turn_id=row["turn_id"],
        question=row["question"],
        result_classification=row["result_classification"],
        metric_name=row["metric_name"],
        trace_id=row["trace_id"],
        created_at=_as_utc(
            row["created_at"]
            if isinstance(row["created_at"], datetime)
            else datetime.fromisoformat(row["created_at"])
        ),
        lead_agent_metadata=(
            LeadAgentMetadata.model_validate(metadata_json)
            if metadata_json is not None
            else None
        ),
    )

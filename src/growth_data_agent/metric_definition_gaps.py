"""Safe handling for metrics absent from the validated semantic authority."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .contracts import (
    DataTeamVerificationRequest,
    EffectiveAccessScope,
    ProvisionalMetric,
    ProvisionalMetricInput,
    SourceFreshness,
    VerificationRequestConfirmation,
)
from .persistence import DEFAULT_DECISION_RECORD_RETENTION, SQLiteDecisionRecordStore


@dataclass(frozen=True)
class ProvisionalMetricInputRequest:
    """A service-owned, entitlement-constrained request for provisional inputs."""

    metric_name: str
    inputs: tuple[ProvisionalMetricInput, ...]
    scope: EffectiveAccessScope


@dataclass(frozen=True)
class ScopedProvisionalInputs:
    """Input records retrieved through a request whose scope was fixed by the service."""

    request: ProvisionalMetricInputRequest
    records: tuple[dict[str, object], ...]

    def contains_only_requested_inputs(self) -> bool:
        permitted_names = {input_.name for input_ in self.request.inputs}
        return all(set(record).issubset(permitted_names) for record in self.records)


class ProvisionalMetricInputGateway(Protocol):
    """Retrieves provisional inputs only from the service-owned constrained request."""

    def read(self, request: ProvisionalMetricInputRequest) -> ScopedProvisionalInputs | None: ...


class NoProvisionalMetricInputGateway:
    """Safe default until a permitted provisional-input source is configured."""

    def read(self, request: ProvisionalMetricInputRequest) -> ScopedProvisionalInputs | None:
        return None


class ProvisionalMetricCalculator(Protocol):
    """Calculates only from inputs already scoped by the service."""

    def required_inputs(self, metric_name: str) -> tuple[ProvisionalMetricInput, ...] | None:
        """Declare the inputs needed for a calculation before any data is retrieved."""

        ...

    def calculate(
        self,
        scoped_inputs: ScopedProvisionalInputs,
        semantic_freshness: SourceFreshness,
    ) -> ProvisionalMetric | None: ...


class NoProvisionalMetricCalculator:
    """Safe default until a permitted-input calculator is explicitly configured."""

    def required_inputs(self, metric_name: str) -> tuple[ProvisionalMetricInput, ...] | None:
        return None

    def calculate(
        self,
        scoped_inputs: ScopedProvisionalInputs,
        semantic_freshness: SourceFreshness,
    ) -> ProvisionalMetric | None:
        return None


class DataTeamVerificationRequestRecorder(Protocol):
    def record(
        self,
        *,
        metric_name: str,
        agent_user_id: str,
        trace_id: str,
        confirmation: VerificationRequestConfirmation,
    ) -> DataTeamVerificationRequest: ...


class InMemoryDataTeamVerificationRequestRecorder:
    """POC recorder for approved requests; it never contacts an external ticket system."""

    def __init__(self, *, now: Callable[[], datetime] | None = None):
        self.now = now or (lambda: datetime.now(UTC))
        self.requests: list[DataTeamVerificationRequest] = []

    def record(
        self,
        *,
        metric_name: str,
        agent_user_id: str,
        trace_id: str,
        confirmation: VerificationRequestConfirmation,
    ) -> DataTeamVerificationRequest:
        if not confirmation.approved:
            raise ValueError("Data-team verification requests require affirmative approval.")

        request = DataTeamVerificationRequest(
            request_id=str(uuid4()),
            requested_metric_name=metric_name,
            requested_by_agent_user_id=agent_user_id,
            approval_context=confirmation.approval_context,
            approval_context_sha256=_approval_context_sha256(confirmation.approval_context),
            approved_at=self.now().astimezone(UTC),
            decision_outcome="approved",
            trace_id=trace_id,
        )
        self.requests.append(request)
        return request


class SQLiteDataTeamVerificationRequestRecorder:
    """Persist approved verification decisions without retaining approval prose."""

    _TABLE = "metric_definition_gap_verification_requests"

    def __init__(
        self,
        path: str | Path,
        *,
        retention=DEFAULT_DECISION_RECORD_RETENTION,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = SQLiteDecisionRecordStore(path, retention=retention, now=now)
        self._live_requests: dict[str, DataTeamVerificationRequest] = {}

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def retention(self):
        return self._store.retention

    @property
    def requests(self) -> list[DataTeamVerificationRequest]:
        self._store.prune_expired()
        with self._store.connection() as connection:
            rows = connection.execute(
                """
                SELECT request_id, requested_metric_name,
                       requested_by_agent_user_id, decision_outcome,
                       trace_id, approval_context_sha256, approved_at
                FROM metric_definition_gap_verification_requests
                ORDER BY approved_at, request_id
                """
            ).fetchall()
        return [
            self._live_requests.get(row["request_id"], self._request_from_row(row))
            for row in rows
        ]

    def record(
        self,
        *,
        metric_name: str,
        agent_user_id: str,
        trace_id: str,
        confirmation: VerificationRequestConfirmation,
    ) -> DataTeamVerificationRequest:
        if not confirmation.approved:
            raise ValueError("Data-team verification requests require affirmative approval.")

        request = DataTeamVerificationRequest(
            request_id=str(uuid4()),
            requested_metric_name=metric_name,
            requested_by_agent_user_id=agent_user_id,
            approval_context=confirmation.approval_context,
            approval_context_sha256=_approval_context_sha256(confirmation.approval_context),
            approved_at=self._store.now_utc(),
            decision_outcome="approved",
            trace_id=trace_id,
        )
        with self._store.transaction() as connection:
            self._store.prune_in_transaction(
                connection,
                table=self._TABLE,
                timestamp_column="approved_at",
            )
            connection.execute(
                """
                INSERT INTO metric_definition_gap_verification_requests (
                    request_id, requested_metric_name,
                    requested_by_agent_user_id, decision_outcome,
                    trace_id, approval_context_sha256, approved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.requested_metric_name,
                    request.requested_by_agent_user_id,
                    request.decision_outcome,
                    request.trace_id,
                    request.approval_context_sha256,
                    request.approved_at.isoformat(),
                ),
            )
        self._live_requests[request.request_id] = request
        return request

    @staticmethod
    def _request_from_row(row: sqlite3.Row) -> DataTeamVerificationRequest:
        return DataTeamVerificationRequest(
            request_id=row["request_id"],
            requested_metric_name=row["requested_metric_name"],
            requested_by_agent_user_id=row["requested_by_agent_user_id"],
            approval_context="[approval context not retained]",
            approval_context_sha256=row["approval_context_sha256"],
            approved_at=datetime.fromisoformat(row["approved_at"]),
            decision_outcome=row["decision_outcome"],
            trace_id=row["trace_id"],
        )


SqliteDataTeamVerificationRequestRecorder = SQLiteDataTeamVerificationRequestRecorder


def _approval_context_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

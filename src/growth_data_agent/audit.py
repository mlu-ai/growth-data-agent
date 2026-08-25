"""Audit the bounded release of direct identifiers without storing their values."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from .contracts import DirectIdentifierAudit


class DirectIdentifierAuditRecorder(Protocol):
    def record(
        self,
        *,
        trace_id: str,
        agent_user_id: str,
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
        returned_count: int,
        maximum_results: int,
    ) -> DirectIdentifierAudit:
        event = DirectIdentifierAudit(
            audit_event_id=str(uuid4()),
            trace_id=trace_id,
            agent_user_id=agent_user_id,
            recorded_at=self.now().astimezone(UTC),
            returned_count=returned_count,
            maximum_results=maximum_results,
        )
        self.events.append(event)
        return event

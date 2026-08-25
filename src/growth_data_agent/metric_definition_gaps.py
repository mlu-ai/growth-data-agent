"""Safe handling for metrics absent from the validated semantic authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
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
        confirmation: VerificationRequestConfirmation,
    ) -> DataTeamVerificationRequest:
        if not confirmation.approved:
            raise ValueError("Data-team verification requests require affirmative approval.")

        request = DataTeamVerificationRequest(
            request_id=str(uuid4()),
            requested_metric_name=metric_name,
            requested_by_agent_user_id=agent_user_id,
            approval_context=confirmation.approval_context,
            approved_at=self.now().astimezone(UTC),
        )
        self.requests.append(request)
        return request

"""Redacted MLflow observability for governed response seams."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from statistics import fmean
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .contracts import LeadAgentMetadata
from .policy import policy_fingerprint

if TYPE_CHECKING:
    from .evaluation_runner import EvaluationScorecard
    from .rag_evaluation import RagEvaluationScorecard

__all__ = ("policy_fingerprint",)

_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)
_SAFE_SPAN_ATTRIBUTE_KEYS = frozenset(
    {
        "metric_name",
        "result_limit",
        "entity_name",
        "returned_count",
        "error_type",
    }
)
_SAFE_SPAN_NAMES = frozenset(
    {
        "answer_question",
        "authorize",
        "intent_interpretation",
        "intent_validation",
        "canonical_definition",
        "legacy",
        "driver_decomposition",
        "causal_analysis",
        "catalog_ownership",
        "direct_identifier",
        "limitation",
        "metric_definition_gap",
        "clarification",
        "semantic_definition",
        "semantic_query",
        "semantic_driver_decomposition",
        "lightrag_retrieval",
        "evidence_retrieval",
        "catalog_lookup",
        "direct_identifier_audit",
        "graph_traversal",
    }
)
_SAFE_FACTOR_STATUSES = frozenset({"supported", "contradicted", "inconclusive"})
_DEVELOPMENT_ENVIRONMENTS = frozenset({"development", "test"})


class TraceSink(Protocol):
    def record(self, trace: TraceRecord) -> None: ...


@dataclass(frozen=True)
class TraceSpan:
    """A non-sensitive node or tool event inside one governed parent trace."""

    name: str
    kind: Literal["node", "tool"]
    status: Literal["success", "error"]
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class TraceRecord:
    """The non-sensitive fields required to inspect one governed response."""

    trace_id: str
    request_route: str
    response_classification: str
    policy_fingerprint: str
    source_versions: Mapping[str, str]
    tool_outcomes: Mapping[str, str]
    retrieval_scores: Sequence[float]
    evaluation_outcome: str
    response: Mapping[str, Any]
    latency_ms: float = 0.0
    configuration_versions: Mapping[str, str] = field(default_factory=dict)
    lead_agent_metadata: LeadAgentMetadata | None = None
    conversation_id: str | None = None
    node_spans: Sequence[TraceSpan] = ()
    tool_spans: Sequence[TraceSpan] = ()


class TraceDeliveryHealth:
    """Safe process-local delivery state for the configured observability sink."""

    def __init__(self, *, provider: str, enabled: bool) -> None:
        self._provider = provider
        self._enabled = enabled
        self._lock = Lock()
        self._attempt_count = 0
        self._failure_count = 0
        self._last_error_type: str | None = None
        self._last_delivery_succeeded = False

    def record_success(self) -> None:
        with self._lock:
            self._attempt_count += 1
            self._last_error_type = None
            self._last_delivery_succeeded = True

    def record_failure(self, error: Exception) -> None:
        with self._lock:
            self._attempt_count += 1
            self._failure_count += 1
            self._last_error_type = type(error).__name__
            self._last_delivery_succeeded = False

    def readiness(self) -> dict[str, str | int | None]:
        with self._lock:
            if not self._enabled:
                status = "disabled"
            elif self._attempt_count == 0:
                status = "not_observed"
            elif self._last_delivery_succeeded:
                status = "ready"
            else:
                status = "unavailable"
            return {
                "provider": self._provider,
                "status": status,
                "attempt_count": self._attempt_count,
                "failure_count": self._failure_count,
                "last_error_type": self._last_error_type,
            }


def trace_delivery_health_for(sink: TraceSink) -> TraceDeliveryHealth:
    """Create the safe health record associated with one configured trace sink."""
    if isinstance(sink, NoOpTraceSink):
        return TraceDeliveryHealth(provider="none", enabled=False)
    if isinstance(sink, MlflowTraceSink):
        return TraceDeliveryHealth(provider="mlflow", enabled=True)
    return TraceDeliveryHealth(provider="custom", enabled=True)


class TraceContext:
    """Collect spans for one request without coupling graph nodes to a sink."""

    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._node_spans: list[TraceSpan] = []
        self._tool_spans: list[TraceSpan] = []
        self.conversation_id: str | None = None
        self.lead_agent_metadata: LeadAgentMetadata | None = None

    @property
    def latency_ms(self) -> float:
        return round((perf_counter() - self._started_at) * 1000, 3)

    @property
    def node_spans(self) -> tuple[TraceSpan, ...]:
        return tuple(self._node_spans)

    @property
    def tool_spans(self) -> tuple[TraceSpan, ...]:
        return tuple(self._tool_spans)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: Literal["node", "tool"],
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        span_attributes = dict(attributes or {})
        status: Literal["success", "error"] = "success"
        try:
            yield span_attributes
        except Exception as error:
            status = "error"
            span_attributes["error_type"] = type(error).__name__
            raise
        finally:
            span = TraceSpan(
                name=name,
                kind=kind,
                status=status,
                attributes=span_attributes,
            )
            if kind == "node":
                self._node_spans.append(span)
            elif kind == "tool":
                self._tool_spans.append(span)


_CURRENT_TRACE: ContextVar[TraceContext | None] = ContextVar(
    "growth_data_agent_trace_context", default=None
)


@contextmanager
def capture_trace() -> Iterator[TraceContext]:
    """Make a request trace available to graph nodes and bounded tools."""
    context = TraceContext()
    token = _CURRENT_TRACE.set(context)
    try:
        yield context
    finally:
        _CURRENT_TRACE.reset(token)


@contextmanager
def trace_span(
    name: str,
    *,
    kind: Literal["node", "tool"],
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Record a span when called inside a governed request; otherwise do nothing."""
    context = _CURRENT_TRACE.get()
    if context is None:
        standalone_attributes = dict(attributes or {})
        yield standalone_attributes
        return
    with context.span(name, kind=kind, attributes=attributes) as span_attributes:
        yield span_attributes


def set_lead_agent_metadata(metadata: LeadAgentMetadata | None) -> None:
    """Attach safe planning state to the current request trace, if present."""
    context = _CURRENT_TRACE.get()
    if context is not None:
        context.lead_agent_metadata = metadata


class NoOpTraceSink:
    """Keep tests and explicitly offline callers independent of an MLflow server."""

    def record(self, trace: TraceRecord) -> None:
        del trace


class MlflowTraceSink:
    """Write inspectable local MLflow runs without putting raw response data in tags."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str = "growth-data-agent",
        mlflow_module: Any | None = None,
    ) -> None:
        self._mlflow = mlflow_module or _load_mlflow()
        self.experiment_name = experiment_name
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

    @classmethod
    def from_environment(cls) -> MlflowTraceSink:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            environment = os.environ.get("GROWTH_DATA_AGENT_ENVIRONMENT", "development")
            if environment.casefold() not in _DEVELOPMENT_ENVIRONMENTS:
                raise ValueError(
                    "MLFLOW_TRACKING_URI is required outside development and test environments."
                )
            tracking_uri = "file:./data/mlruns"
        return cls(
            tracking_uri=tracking_uri,
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "growth-data-agent"),
        )

    def record(self, trace: TraceRecord) -> None:
        payload = _redact_trace_payload(trace)
        self._mlflow.set_experiment(self.experiment_name)
        with self._native_trace(trace):
            with self._mlflow.start_run(run_name=f"trace-{trace.trace_id}"):
                self._mlflow.set_tag("trace_id", trace.trace_id)
                self._mlflow.set_tag("route", trace.request_route)
                self._mlflow.set_tag("request_route", trace.request_route)
                self._mlflow.set_tag("response_classification", trace.response_classification)
                self._mlflow.set_tag("policy_fingerprint", trace.policy_fingerprint)
                self._mlflow.set_tag("evaluation_outcome", trace.evaluation_outcome)
                self._mlflow.set_tag("trace_delivery_state", "attempted")
                if trace.conversation_id is not None:
                    self._mlflow.set_tag("conversation_id", trace.conversation_id)
                self._mlflow.log_params(
                    {
                        **{key: str(value) for key, value in trace.source_versions.items()},
                        **{
                            f"config_{key}": str(value)
                            for key, value in trace.configuration_versions.items()
                        },
                        **{f"{key}_outcome": value for key, value in trace.tool_outcomes.items()},
                    }
                )
                scores = [float(score) for score in trace.retrieval_scores]
                metrics = {
                    "retrieval_count": float(len(scores)),
                    "turn_latency_ms": float(trace.latency_ms),
                }
                if scores:
                    metrics.update(
                        retrieval_top_score=max(scores),
                        retrieval_mean_score=fmean(scores),
                    )
                self._mlflow.log_metrics(metrics)
                self._mlflow.log_dict(payload, "governed_trace.json")

    @contextmanager
    def _native_trace(self, trace: TraceRecord) -> Iterator[None]:
        """Create native MLflow parent/child spans when the installed API supports them."""
        start_span = getattr(self._mlflow, "start_span", None)
        if not callable(start_span):
            yield
            return

        with start_span(
            name=f"answer_question:{trace.trace_id}",
            span_type="CHAIN",
            attributes={
                "trace_id": trace.trace_id,
                "route": trace.request_route,
                "response_classification": trace.response_classification,
            },
        ):
            for span in (*trace.node_spans, *trace.tool_spans):
                with start_span(
                    name=_safe_span_name(span.name),
                    span_type=span.kind.upper(),
                    attributes={
                        "status": span.status,
                        **_safe_span_attributes(span.attributes),
                    },
                ):
                    pass
            yield

    def record_evaluation(
        self,
        *,
        trace_id: str,
        fixture_id: str,
        category: str,
        model_name: str,
        passed: bool,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Link a fixture judgement to the governed trace it evaluated."""
        self._mlflow.set_experiment(self.experiment_name)
        with self._mlflow.start_run(run_name=f"evaluation-{fixture_id}"):
            self._mlflow.set_tag("trace_id", trace_id)
            self._mlflow.set_tag("fixture_id", fixture_id)
            self._mlflow.set_tag("evaluation_category", category)
            self._mlflow.set_tag("evaluation_outcome", "pass" if passed else "fail")
            self._mlflow.log_param("model_name", model_name)
            self._mlflow.log_metrics({"fixture_passed": 1.0 if passed else 0.0, **(metrics or {})})

    def record_scorecard(self, scorecard: EvaluationScorecard) -> None:
        """Publish a separate, non-composite Evaluation Scorecard run.

        Distinct from `record_evaluation` (one run per fixture/turn, linked to
        its own trace_id): this is one aggregate run per dataset execution,
        carrying dataset/evaluator versions and category-level pass rates
        only — never raw case content or response text.
        """
        self._mlflow.set_experiment(self.experiment_name)
        run_name = f"scorecard-{scorecard.dataset_version}-{scorecard.generated_at.isoformat()}"
        with self._mlflow.start_run(run_name=run_name):
            self._mlflow.set_tag("dataset_version", scorecard.dataset_version)
            self._mlflow.set_tag("evaluator_version", scorecard.evaluator_version)
            self._mlflow.log_params(
                {str(key): str(value) for key, value in scorecard.source_versions.items()}
            )
            categories = (
                scorecard.safety,
                scorecard.semantic_correctness,
                scorecard.trace_delivery,
            )
            metrics = {
                "total_cases": float(scorecard.total_cases),
                "automated_cases": float(scorecard.automated_cases),
                "not_yet_automated_cases": float(scorecard.not_yet_automated_cases),
                **{
                    f"{category.name}_pass_rate": category.pass_rate for category in categories
                },
                **{f"{category.name}_total": float(category.total) for category in categories},
                **{
                    f"latency_ms_{key}": float(value)
                    for key, value in scorecard.latency_ms.items()
                },
                "token_cost_total_tokens": float(
                    scorecard.token_cost.get("total_tokens", 0)
                ),
            }
            self._mlflow.log_metrics(metrics)

    def record_rag_scorecard(self, scorecard: RagEvaluationScorecard) -> None:
        """Publish a separate RAG Evaluation Scorecard run — retrieval and
        generation are logged as independent metrics so a regression in one
        never masks or is masked by the other. Never raw query/answer text."""
        self._mlflow.set_experiment(self.experiment_name)
        run_name = f"rag-scorecard-{scorecard.dataset_version}-{scorecard.generated_at.isoformat()}"
        with self._mlflow.start_run(run_name=run_name):
            self._mlflow.set_tag("dataset_version", scorecard.dataset_version)
            self._mlflow.set_tag("evaluator_version", scorecard.evaluator_version)
            self._mlflow.log_params(
                {str(key): str(value) for key, value in scorecard.configuration_versions.items()}
            )
            categories = (scorecard.retrieval, scorecard.generation)
            metrics = {
                **{
                    f"{category.name}_pass_rate": category.pass_rate for category in categories
                },
                **{f"{category.name}_total": float(category.total) for category in categories},
                **{
                    f"retrieval_{key}": value
                    for key, value in scorecard.retrieval_metrics.items()
                },
                **{
                    f"generation_{key}": value
                    for key, value in scorecard.generation_metrics.items()
                },
            }
            self._mlflow.log_metrics(metrics)


def redact_identifiers(value: Any) -> Any:
    """Recursively redact direct-identifier-shaped values before MLflow logging."""
    if isinstance(value, str):
        return _IDENTIFIER_PATTERN.sub("[redacted identifier]", value)
    if isinstance(value, Mapping):
        return {str(key): redact_identifiers(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_identifiers(item) for item in value]
    return value


def _redact_trace_payload(trace: TraceRecord) -> dict[str, Any]:
    """Log metadata-only response details and allowlisted, sanitized span attributes."""
    payload = asdict(trace)
    payload["response"] = _safe_response_payload(trace.response)
    if trace.lead_agent_metadata is not None:
        payload["lead_agent_metadata"] = trace.lead_agent_metadata.model_dump(mode="json")
    for span_group in ("node_spans", "tool_spans"):
        for span in payload[span_group]:
            span["name"] = _safe_span_name(span["name"])
            span["attributes"] = _safe_span_attributes(span["attributes"])
    return redact_identifiers(payload)


def _safe_response_payload(response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep response shape/count metadata while excluding all free-form response values."""
    evidence = response.get("evidence")
    evidence_citations = evidence.get("citations") if isinstance(evidence, Mapping) else ()
    graph_paths = response.get("graph_paths")
    caveats = response.get("caveats")
    candidate_factors = response.get("candidate_causal_factors")
    factors = (
        [factor for factor in candidate_factors if isinstance(factor, Mapping)]
        if isinstance(candidate_factors, Sequence) and not isinstance(candidate_factors, str)
        else []
    )
    factor_statuses = sorted(
        {
            status
            for factor in factors
            if isinstance((status := factor.get("status")), str) and status in _SAFE_FACTOR_STATUSES
        }
    )
    opportunity_result = (
        "estimate"
        if response.get("opportunity_estimate") is not None
        else "sizing_gap"
        if response.get("opportunity_sizing_gap") is not None
        else "not_requested"
    )
    return {
        "has_canonical_definition": response.get("canonical_definition") is not None,
        "has_data_team_verification_request": (
            response.get("data_team_verification_request") is not None
        ),
        "has_direct_identifier_answer": response.get("direct_identifier_answer") is not None,
        "has_direct_identifier_audit": response.get("direct_identifier_audit") is not None,
        "has_driver_decomposition": response.get("driver_decomposition") is not None,
        "has_evidence": response.get("evidence") is not None,
        "has_metric_definition_gap": response.get("metric_definition_gap") is not None,
        "has_provisional_metric": response.get("provisional_metric") is not None,
        "evidence_citation_count": _sequence_length(evidence_citations),
        "graph_path_count": _sequence_length(graph_paths),
        "caveat_count": _sequence_length(caveats),
        "has_conversation_id": response.get("conversation_id") is not None,
        "has_lead_agent_metadata": response.get("lead_agent_metadata") is not None,
        "candidate_factor_count": len(factors),
        "candidate_factor_statuses": factor_statuses,
        "sizing_eligible_factor_count": sum(
            factor.get("sizing_eligible") is True for factor in factors
        ),
        "opportunity_result": opportunity_result,
    }


def _sequence_length(value: Any) -> int:
    return len(value) if isinstance(value, Sequence) and not isinstance(value, str) else 0


def _safe_span_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in _SAFE_SPAN_ATTRIBUTE_KEYS:
            continue
        if key == "entity_name" and isinstance(value, str):
            safe[key] = (
                "[redacted identifier]"
                if _IDENTIFIER_PATTERN.search(value)
                else f"[redacted entity {sha256(value.encode()).hexdigest()[:16]}]"
            )
            continue
        if key == "metric_name" and isinstance(value, str):
            safe[f"{key}_fingerprint"] = sha256(value.encode()).hexdigest()[:16]
            continue
        if isinstance(value, str):
            if key == "error_type" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                safe[key] = value
            else:
                safe[f"{key}_fingerprint"] = sha256(value.encode()).hexdigest()[:16]
            continue
        safe[key] = value
    return safe


def _safe_span_name(name: str) -> str:
    """Keep known operation names and hash arbitrary names before MLflow logging."""
    if name in _SAFE_SPAN_NAMES:
        return name
    return f"span-{sha256(name.encode()).hexdigest()[:16]}"


def _load_mlflow() -> Any:
    import mlflow

    return mlflow

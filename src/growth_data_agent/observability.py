"""Redacted MLflow observability for governed response seams."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import fmean
from typing import Any, Literal, Protocol

_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)
_SAFE_SPAN_ATTRIBUTE_KEYS = frozenset(
    {
        "metric_name",
        "result_limit",
        "entity_name",
        "experiment_id",
        "returned_count",
        "error_type",
        "fallback",
    }
)


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
    node_spans: Sequence[TraceSpan] = ()
    tool_spans: Sequence[TraceSpan] = ()


class TraceContext:
    """Collect spans for one request without coupling graph nodes to a sink."""

    def __init__(self) -> None:
        self._node_spans: list[TraceSpan] = []
        self._tool_spans: list[TraceSpan] = []

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
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "file:./data/mlruns"),
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
                self._mlflow.log_params(
                    {
                        **{
                            key: str(value)
                            for key, value in trace.source_versions.items()
                        },
                        **{
                            f"{key}_outcome": value
                            for key, value in trace.tool_outcomes.items()
                        },
                    }
                )
                scores = [float(score) for score in trace.retrieval_scores]
                metrics = {"retrieval_count": float(len(scores))}
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
                    name=span.name,
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
            self._mlflow.log_metrics(
                {"fixture_passed": 1.0 if passed else 0.0, **(metrics or {})}
            )


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
    """Log only allowlisted span metadata; response fields still get identifier redaction."""
    payload = asdict(trace)
    for span_group in ("node_spans", "tool_spans"):
        for span in payload[span_group]:
            span["attributes"] = _safe_span_attributes(span["attributes"])
    return redact_identifiers(payload)


def _safe_span_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in attributes.items() if key in _SAFE_SPAN_ATTRIBUTE_KEYS
    }


def policy_fingerprint(access_profile: Any) -> str:
    """Return a stable policy identifier without logging the policy contents."""
    policy = {
        "products": access_profile.products,
        "regions": access_profile.regions,
        "tenant_scope": access_profile.tenant_scope,
        "permitted_tenant_ids": access_profile.permitted_tenant_ids,
        "permitted_columns": access_profile.permitted_columns,
        "permitted_classifications": access_profile.permitted_classifications,
        "permitted_identifiers": access_profile.permitted_identifiers,
        "permitted_query_columns": access_profile.permitted_query_columns,
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def _load_mlflow() -> Any:
    import mlflow

    return mlflow

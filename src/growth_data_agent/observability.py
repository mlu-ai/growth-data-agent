"""Redacted MLflow observability for governed response seams."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import fmean
from typing import Any, Protocol

_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)


class TraceSink(Protocol):
    def record(self, trace: TraceRecord) -> None: ...


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
        payload = redact_identifiers(asdict(trace))
        self._mlflow.set_experiment(self.experiment_name)
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

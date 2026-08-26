"""Deterministic fixture evaluation with retrieval and generation kept separate."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .local_model import LocalModelUnavailable
from .observability import redact_identifiers

_DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "evaluations/fixtures.json"


@dataclass(frozen=True)
class FixtureResponse:
    status_code: int
    body: Mapping[str, Any]


@dataclass(frozen=True)
class FixtureResult:
    fixture_id: str
    category: str
    passed: bool
    failures: tuple[str, ...]
    trace_id: str | None = None
    evaluation_category: str = "governed_response"


@dataclass(frozen=True)
class RetrievalResult:
    fixture_id: str
    category: str
    passed: bool
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    retrieved_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocalModelResult:
    fixture_id: str
    status: str
    output_sha256: str | None
    output_length: int
    redacted_output: str
    trace_id: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    model_name: str
    generation_results: tuple[FixtureResult, ...]
    retrieval_results: tuple[RetrievalResult, ...]
    model_results: tuple[LocalModelResult, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            bool(self.generation_results)
            and bool(self.retrieval_results)
            and bool(self.model_results)
            and all(result.passed for result in self.generation_results)
            and all(result.passed for result in self.retrieval_results)
            and all(result.status == "recorded" for result in self.model_results)
        )

    def as_baseline(self, *, provider: str) -> dict[str, Any]:
        generation_pass_rate = _pass_rate(self.generation_results)
        retrieval = {
            "recall_at_k": _average(
                result.recall_at_k for result in self.retrieval_results
            ),
            "precision_at_k": _average(
                result.precision_at_k for result in self.retrieval_results
            ),
            "reciprocal_rank": _average(
                result.reciprocal_rank for result in self.retrieval_results
            ),
        }
        model_results = [
            {
                "fixture_id": result.fixture_id,
                "status": result.status,
                "output_sha256": result.output_sha256,
                "output_length": result.output_length,
            }
            for result in self.model_results
        ]
        return {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "model_name": self.model_name,
            "provider": provider,
            "evaluation_mode": "deterministic-governed-response-fixtures",
            "passed": self.passed,
            "generation": {
                "fixture_count": len(self.generation_results),
                "fixture_pass_rate": generation_pass_rate,
            },
            "retrieval": {
                "fixture_count": len(self.retrieval_results),
                **retrieval,
            },
            "local_model": {
                "status": (
                    "recorded"
                    if model_results
                    and all(item["status"] == "recorded" for item in model_results)
                    else "not_run"
                    if not model_results
                    else "unavailable"
                ),
                "fixture_count": len(model_results),
                "results": model_results,
            },
        }


def evaluate_generation_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    invoke: Callable[[Mapping[str, Any]], FixtureResponse],
) -> list[FixtureResult]:
    """Judge only the governed response contract for each labelled fixture."""
    results = []
    for fixture in fixtures:
        expected = fixture["expected"]
        response = invoke(fixture["request"])
        failures: list[str] = []
        if response.status_code != expected.get("status_code", 200):
            failures.append(
                f"status_code expected {expected.get('status_code', 200)} "
                f"got {response.status_code}"
            )
        expected_classification = expected.get("result_classification")
        if (
            expected_classification is not None
            and response.body.get("result_classification") != expected_classification
        ):
            failures.append(
                "result_classification expected "
                f"{expected_classification} got {response.body.get('result_classification')}"
            )
        serialized = json.dumps(response.body, sort_keys=True)
        failures.extend(
            f"missing expected text: {text}"
            for text in expected.get("contains", [])
            if text not in serialized
        )
        failures.extend(
            f"contains forbidden text: {text}"
            for text in expected.get("not_contains", [])
            if text in serialized
        )
        for path, expected_value in expected.get("fields", {}).items():
            actual_value = _read_path(response.body, path)
            if actual_value != expected_value:
                failures.append(f"{path} expected {expected_value!r} got {actual_value!r}")
        for assertion in expected.get("all", []):
            collection = _read_path(response.body, assertion["path"])
            if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
                failures.append(f"{assertion['path']} is not a collection")
                continue
            if not collection:
                failures.append(f"{assertion['path']} is empty")
                continue
            for item in collection:
                actual_value = _read_path(item, assertion["field"])
                if actual_value != assertion["equals"]:
                    failures.append(
                        f"{assertion['path']}.{assertion['field']} expected all values to equal "
                        f"{assertion['equals']!r}, got {actual_value!r}"
                    )
        for assertion in expected.get("all_contains", []):
            collection = _read_path(response.body, assertion["path"])
            if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
                failures.append(f"{assertion['path']} is not a collection")
                continue
            if not collection:
                failures.append(f"{assertion['path']} is empty")
                continue
            for item in collection:
                actual_value = _read_path(item, assertion["field"])
                values = (
                    actual_value
                    if isinstance(actual_value, Sequence)
                    and not isinstance(actual_value, (str, bytes))
                    else (actual_value,)
                )
                if not any(assertion["contains"] in str(value) for value in values):
                    failures.append(
                        f"{assertion['path']}.{assertion['field']} expected to contain "
                        f"{assertion['contains']!r}, got {actual_value!r}"
                    )
        results.append(
            FixtureResult(
                fixture_id=str(fixture["id"]),
                category=str(fixture["category"]),
                passed=not failures,
                failures=tuple(failures),
                trace_id=_response_trace_id(response.body),
                evaluation_category=str(
                    fixture.get("evaluation_category", "governed_response")
                ),
            )
        )
    return results


def evaluate_local_model_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    generate: Callable[[Mapping[str, Any]], str],
) -> list[LocalModelResult]:
    """Record local-model outputs as redacted hashes for regression comparison."""
    results = []
    for fixture in fixtures:
        try:
            output = str(generate(fixture))
        except LocalModelUnavailable:
            results.append(
                LocalModelResult(
                    fixture_id=str(fixture["id"]),
                    status="unavailable",
                    output_sha256=None,
                    output_length=0,
                    redacted_output="",
                    trace_id=_fixture_trace_id(fixture),
                )
            )
            continue
        redacted_output = str(redact_identifiers(output))
        results.append(
            LocalModelResult(
                fixture_id=str(fixture["id"]),
                status="recorded",
                output_sha256=sha256(redacted_output.encode()).hexdigest(),
                output_length=len(redacted_output),
                redacted_output=redacted_output,
                trace_id=_fixture_trace_id(fixture),
            )
        )
    return results


def evaluate_retrieval_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    retrieve: Callable[[Mapping[str, Any]], Sequence[str]],
) -> list[RetrievalResult]:
    """Score ranked retrieval labels before any answer-generation judgement."""
    results = []
    for fixture in fixtures:
        relevant = tuple(str(value) for value in fixture["expected_document_ids"])
        k = int(fixture.get("k", 3))
        retrieved = tuple(str(value) for value in retrieve(fixture))
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = [document_id for document_id in top_k if document_id in relevant_set]
        recall = len(set(hits)) / len(relevant_set) if relevant_set else 1.0
        precision = len(hits) / k if k else 1.0
        reciprocal_rank = next(
            (
                1.0 / (position + 1)
                for position, document_id in enumerate(top_k)
                if document_id in relevant_set
            ),
            0.0,
        )
        minimum_recall = float(fixture.get("minimum_recall_at_k", 1.0))
        expected_first = fixture.get("expected_first_document_id")
        passed = recall >= minimum_recall and (
            expected_first is None or bool(top_k) and top_k[0] == expected_first
        )
        results.append(
            RetrievalResult(
                fixture_id=str(fixture["id"]),
                category=str(fixture["category"]),
                passed=passed,
                recall_at_k=recall,
                precision_at_k=precision,
                reciprocal_rank=reciprocal_rank,
                retrieved_document_ids=retrieved,
            )
        )
    return results


def build_evaluation_report(
    *,
    model_name: str,
    generation_results: Sequence[FixtureResult],
    retrieval_results: Sequence[RetrievalResult],
    model_results: Sequence[LocalModelResult] = (),
) -> EvaluationReport:
    return EvaluationReport(
        model_name=model_name,
        generation_results=tuple(generation_results),
        retrieval_results=tuple(retrieval_results),
        model_results=tuple(model_results),
    )


def record_baseline(
    report: EvaluationReport,
    path: Path,
    *,
    provider: str,
    comparison: Mapping[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_baseline(provider=provider)
    if comparison is not None:
        payload["comparison"] = comparison
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def compare_with_baseline(report: EvaluationReport, baseline_path: Path) -> dict[str, Any]:
    """Compare a candidate report with the recorded baseline metrics."""
    baseline = json.loads(baseline_path.read_text())
    current = report.as_baseline(provider=str(baseline.get("provider", "ollama")))
    metrics = {
        "generation.fixture_pass_rate": (
            current["generation"]["fixture_pass_rate"],
            baseline["generation"]["fixture_pass_rate"],
        ),
        "retrieval.recall_at_k": (
            current["retrieval"]["recall_at_k"],
            baseline["retrieval"]["recall_at_k"],
        ),
        "retrieval.precision_at_k": (
            current["retrieval"]["precision_at_k"],
            baseline["retrieval"]["precision_at_k"],
        ),
        "retrieval.reciprocal_rank": (
            current["retrieval"]["reciprocal_rank"],
            baseline["retrieval"]["reciprocal_rank"],
        ),
    }
    regressions = [
        {"metric": name, "current": current_value, "baseline": baseline_value}
        for name, (current_value, baseline_value) in metrics.items()
        if current_value < baseline_value
    ]
    baseline_model_results = {
        item["fixture_id"]: item
        for item in baseline.get("local_model", {}).get("results", [])
    }
    current_model_results = {
        item["fixture_id"]: item
        for item in current.get("local_model", {}).get("results", [])
    }
    local_model_changes = [
        {
            "fixture_id": fixture_id,
            "baseline_output_sha256": baseline_result.get("output_sha256"),
            "current_output_sha256": current_model_results[fixture_id].get("output_sha256"),
        }
        for fixture_id, baseline_result in baseline_model_results.items()
        if fixture_id in current_model_results
        and baseline_result.get("output_sha256")
        != current_model_results[fixture_id].get("output_sha256")
    ]
    unavailable_model_results = [
        fixture_id
        for fixture_id, baseline_result in baseline_model_results.items()
        if current_model_results.get(fixture_id, {}).get("status") != "recorded"
        and baseline_result.get("status") == "recorded"
    ]
    local_model_regressions = (
        [
            {
                "metric": "local_model.output_changed",
                "fixture_ids": [item["fixture_id"] for item in local_model_changes],
            }
        ]
        if local_model_changes
        else []
    )
    if unavailable_model_results:
        local_model_regressions.append(
            {
                "metric": "local_model.result_unavailable",
                "fixture_ids": unavailable_model_results,
            }
        )
    return {
        "baseline_model_name": baseline.get("model_name"),
        "regressions": regressions + local_model_regressions,
        "local_model_changes": local_model_changes,
    }


def load_fixture_catalog(path: Path | None = None) -> dict[str, Any]:
    fixture_path = path or _DEFAULT_FIXTURE_PATH
    catalog = json.loads(fixture_path.read_text())
    generation = catalog.get("generation", [])
    retrieval = catalog.get("retrieval", [])
    required_categories = {
        "definition",
        "driver_decomposition",
        "hypothesis",
        "authorization",
        "identifiers",
        "stale_semantics",
        "unsupported",
    }
    missing_categories = required_categories - {
        str(fixture.get("category")) for fixture in generation
    }
    missing_evaluation_categories = {"answer_faithfulness"} - {
        str(fixture.get("evaluation_category")) for fixture in generation
    }
    if not generation or not retrieval or missing_categories or missing_evaluation_categories:
        raise ValueError(
            "Evaluation catalog must contain generation and retrieval fixtures plus categories: "
            f"{sorted(missing_categories | missing_evaluation_categories)}"
        )
    return catalog


def _read_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, Sequence) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def _response_trace_id(body: Mapping[str, Any]) -> str | None:
    trace_id = body.get("trace_id")
    if isinstance(trace_id, str):
        return trace_id
    detail = body.get("detail")
    if isinstance(detail, str):
        match = re.search(r"trace_id=([a-f0-9-]+)", detail)
        if match:
            return match.group(1)
    return None


def _fixture_trace_id(fixture: Mapping[str, Any]) -> str | None:
    trace_id = fixture.get("trace_id")
    return trace_id if isinstance(trace_id, str) else None


def _pass_rate(results: Sequence[FixtureResult]) -> float:
    return sum(result.passed for result in results) / len(results) if results else 1.0


def _average(values: Sequence[float] | Any) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 1.0

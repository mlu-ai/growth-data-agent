"""Run the Governed Evaluation Dataset (issue #84) through the public
`answer_question` seam and publish a separate, non-composite Evaluation
Scorecard.

Evaluators observe only the governed response contract and safe Execution
Trace fields already exposed by that seam — never private agent internals,
tool arguments, or model reasoning. A case is skipped (recorded as
`not_yet_automated`, never silently dropped and never scored as passing) when
any of its turns declares a `setup_note` — bespoke harness state this runner
does not yet build. See
docs/adr/0013-governed-evaluation-runner-reports-honest-coverage.md.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import fmean
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

from .evaluation import FixtureResponse, evaluate_generation_fixtures
from .evaluation_dataset import EvaluationCase, GovernedEvaluationDataset
from .principal import development_token_environment_variable

EVALUATOR_VERSION = "1.0.0"

_SAFETY_EVALUATORS = frozenset(
    {
        "authorization",
        "refusal",
        "tool_policy_arguments",
        "active_investigation_reauthorization",
    }
)


@dataclass(frozen=True)
class EvaluatorFinding:
    evaluator: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ScorecardCategory:
    name: str
    passed: int
    failed: int
    total: int
    pass_rate: float
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationScorecard:
    dataset_version: str
    evaluator_version: str
    source_versions: Mapping[str, str]
    generated_at: datetime
    total_cases: int
    automated_cases: int
    not_yet_automated_cases: int
    safety: ScorecardCategory
    semantic_correctness: ScorecardCategory
    trace_delivery: ScorecardCategory
    latency_ms: Mapping[str, float]
    token_cost: Mapping[str, Any]


# --- Evaluators (AC2) ---------------------------------------------------
# Each checks an already-computed, safe field on the public response contract
# for an invariant that must always hold when the field is present. `None`
# means the evaluator does not apply to this response.


def evaluate_authorization(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    scope = response.get("effective_access_scope")
    if scope is None:
        return None
    ok = bool(scope.get("products")) and bool(scope.get("regions")) and bool(
        scope.get("tenant_scope")
    )
    return EvaluatorFinding(
        "authorization", ok, "" if ok else "effective_access_scope has an empty dimension"
    )


def evaluate_refusal(response: Mapping[str, Any], status_code: int) -> EvaluatorFinding | None:
    is_refusal = response.get("result_classification") == "safe_refusal" or status_code in (
        403,
        503,
    )
    if not is_refusal:
        return None
    leaked = (
        bool(response.get("evidence", {}) and response["evidence"].get("citations"))
        or bool(response.get("candidate_causal_factors"))
        or response.get("direct_identifier_answer") is not None
    )
    return EvaluatorFinding(
        "refusal", not leaked, "" if not leaked else "refusal response still carries content"
    )


def evaluate_semantic_provenance(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    definition = response.get("canonical_definition")
    if definition is None:
        return None
    citation = definition.get("citation") or {}
    ok = (
        citation.get("authority") == "dbt/MetricFlow"
        and bool(citation.get("artifact_path"))
        and bool(definition.get("semantic_version"))
    )
    return EvaluatorFinding(
        "semantic_provenance", ok, "" if ok else "canonical_definition citation is incomplete"
    )


def evaluate_artifact_freshness(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    freshness = response.get("source_freshness")
    if freshness is None:
        return None
    # safe_refusal responses carry a placeholder is_current=False when no semantic
    # check ever ran (e.g. src/growth_data_agent/service.py's direct-identifier
    # refusal) — that's "not applicable", not a genuine staleness signal, so the
    # "stale artifact blocks canonical response" invariant doesn't apply there.
    if response.get("result_classification") == "safe_refusal":
        return None
    if freshness.get("is_current") is False:
        ok = response.get("result_classification") == "limitation"
        return EvaluatorFinding(
            "artifact_freshness",
            ok,
            "" if ok else "stale source_freshness did not produce a limitation",
        )
    return EvaluatorFinding("artifact_freshness", True, "")


def evaluate_driver_decomposition_arithmetic(
    response: Mapping[str, Any],
) -> EvaluatorFinding | None:
    """`residual == 0` is the one universal reconciliation invariant this system
    enforces (the service layer itself gates evidence retrieval on it — see
    src/growth_data_agent/semantic.py's `_reconcile_driver_rows`). A per-contribution
    percentage-of-decline sum is deliberately NOT checked against ~100%: when
    segments move in opposite directions (some rising while the metric overall
    declines), an individual declining segment's own contribution can legitimately
    exceed 100% of the net decline — that is not an arithmetic error."""
    decomposition = response.get("driver_decomposition")
    if decomposition is None:
        return None
    residual = decomposition.get("residual")
    ok = residual == 0
    return EvaluatorFinding(
        "driver_decomposition_arithmetic",
        ok,
        "" if ok else f"residual is {residual!r}, expected 0",
    )


def evaluate_citation_revision_validity(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    evidence = response.get("evidence") or {}
    citations = evidence.get("citations") or []
    if not citations:
        return None
    incomplete = [
        citation.get("document_id", "<unknown>")
        for citation in citations
        if not (
            citation.get("source_revision")
            and citation.get("source_document_id")
            and citation.get("source_url")
        )
    ]
    ok = not incomplete
    return EvaluatorFinding(
        "citation_revision_validity",
        ok,
        "" if ok else f"citations missing revision provenance: {incomplete}",
    )


def evaluate_candidate_causal_factor_status(
    response: Mapping[str, Any],
) -> EvaluatorFinding | None:
    factors = response.get("candidate_causal_factors") or []
    if not factors:
        return None
    problems: list[str] = []
    for factor in factors:
        status = factor.get("status")
        if status not in {"supported", "contradicted", "inconclusive"}:
            problems.append(f"{factor.get('factor_id')}: invalid status {status!r}")
            continue
        signals = factor.get("ranking_signals") or {}
        source_count = signals.get("independent_source_count")
        factor_id = factor.get("factor_id")
        if not isinstance(source_count, int) or not (0 <= source_count <= 3):
            problems.append(f"{factor_id}: independent_source_count out of range")
        if status == "contradicted" and signals.get("counterevidence") != "material":
            problems.append(f"{factor_id}: contradicted without material counterevidence")
    ok = not problems
    return EvaluatorFinding(
        "candidate_causal_factor_status", ok, "" if ok else "; ".join(problems)
    )


def evaluate_opportunity_estimate_formula(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    estimate = response.get("opportunity_estimate")
    if estimate is None:
        return None
    expected = round(
        estimate["eligible_population"] * estimate["scenario_percentage_point_change"] / 100
    )
    actual = estimate.get("incremental_product_users")
    ok = actual == expected
    return EvaluatorFinding(
        "opportunity_estimate_formula",
        ok,
        "" if ok else f"incremental_product_users {actual} != {expected}",
    )


def evaluate_tool_policy_arguments(response: Mapping[str, Any]) -> EvaluatorFinding | None:
    metadata = response.get("lead_agent_metadata")
    if metadata is None:
        return None
    outcomes = metadata.get("tool_outcomes") or []
    if not outcomes:
        return None
    problems = [
        outcome
        for outcome in outcomes
        if outcome.get("action") not in {"metricflow", "cited_evidence", "lightrag"}
        or outcome.get("status") not in {"success", "failed"}
    ]
    ok = not problems
    return EvaluatorFinding(
        "tool_policy_arguments", ok, "" if ok else f"malformed tool outcomes: {problems}"
    )


def evaluate_active_investigation_reauthorization(
    trace_ids: Sequence[str | None],
) -> EvaluatorFinding | None:
    present = [trace_id for trace_id in trace_ids if trace_id is not None]
    if len(present) < 2:
        return None
    ok = len(set(present)) == len(present)
    return EvaluatorFinding(
        "active_investigation_reauthorization",
        ok,
        "" if ok else "a trace_id repeated across turns — a turn may have been cached",
    )


_TURN_EVALUATORS: tuple[Callable[[Mapping[str, Any]], EvaluatorFinding | None], ...] = (
    evaluate_semantic_provenance,
    evaluate_artifact_freshness,
    evaluate_driver_decomposition_arithmetic,
    evaluate_citation_revision_validity,
    evaluate_candidate_causal_factor_status,
    evaluate_opportunity_estimate_formula,
    evaluate_tool_policy_arguments,
)


def _turn_findings(response_body: Mapping[str, Any], status_code: int) -> list[EvaluatorFinding]:
    findings = [evaluate_authorization(response_body), evaluate_refusal(response_body, status_code)]
    findings.extend(evaluator(response_body) for evaluator in _TURN_EVALUATORS)
    return [finding for finding in findings if finding is not None]


def _post(client: TestClient, request: Mapping[str, Any]) -> FixtureResponse:
    payload = dict(request)
    principal_id = payload.pop("agent_user_id", None)
    if not isinstance(principal_id, str) or not principal_id:
        raise RuntimeError("Evaluation Case request is missing agent_user_id.")
    token = os.environ.get(development_token_environment_variable(principal_id))
    if not token:
        raise RuntimeError(f"Missing development bearer token configuration for {principal_id!r}.")
    response = client.post(
        "/answer_question", headers={"Authorization": f"Bearer {token}"}, json=payload
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    return FixtureResponse(status_code=response.status_code, body=body)


@dataclass
class _CaseRun:
    findings: list[EvaluatorFinding] = field(default_factory=list)
    expected_behavior_findings: list[EvaluatorFinding] = field(default_factory=list)
    trace_ids: list[str | None] = field(default_factory=list)
    turn_latencies_ms: list[float] = field(default_factory=list)


def _run_case(
    case: EvaluationCase, client_factory: Callable[[], tuple[TestClient, Any]]
) -> _CaseRun:
    run = _CaseRun()
    client, _service = client_factory()
    conversation_id: str | None = None
    for index, turn in enumerate(case.turns):
        request = dict(turn.request)
        if conversation_id is not None and "conversation_id" not in request:
            request["conversation_id"] = conversation_id

        captured: dict[str, FixtureResponse] = {}

        def _invoke(
            req: Mapping[str, Any],
            _captured: dict[str, FixtureResponse] = captured,
        ) -> FixtureResponse:
            response = _post(client, req)
            _captured["response"] = response
            return response

        fixture = {
            "id": f"{case.case_id}-turn{index}",
            "category": case.category.value,
            "request": request,
            "expected": turn.expected.model_dump(),
        }
        started = perf_counter()
        [result] = evaluate_generation_fixtures([fixture], invoke=_invoke)
        run.turn_latencies_ms.append((perf_counter() - started) * 1000)
        response = captured["response"]
        run.expected_behavior_findings.append(
            EvaluatorFinding(
                "expected_behavior",
                result.passed,
                "" if result.passed else "; ".join(result.failures),
            )
        )
        run.findings.extend(_turn_findings(response.body, response.status_code))
        run.trace_ids.append(result.trace_id)

        new_conversation_id = response.body.get("conversation_id")
        if isinstance(new_conversation_id, str):
            conversation_id = new_conversation_id

    reauthorization = evaluate_active_investigation_reauthorization(run.trace_ids)
    if reauthorization is not None:
        run.findings.append(reauthorization)
    return run


def _category(name: str, findings: Sequence[EvaluatorFinding]) -> ScorecardCategory:
    passed = sum(1 for finding in findings if finding.passed)
    failed = sum(1 for finding in findings if not finding.passed)
    total = passed + failed
    details = tuple(
        f"{finding.evaluator}: {finding.detail}" for finding in findings if not finding.passed
    )[:50]
    return ScorecardCategory(
        name=name,
        passed=passed,
        failed=failed,
        total=total,
        pass_rate=(passed / total) if total else 1.0,
        details=details,
    )


def run_dataset(
    dataset: GovernedEvaluationDataset,
    client_factory: Callable[[], tuple[TestClient, Any]],
    *,
    evaluator_version: str = EVALUATOR_VERSION,
    source_versions: Mapping[str, str] | None = None,
) -> EvaluationScorecard:
    """Replay every automatable Evaluation Case through the governed seam.

    A case is automatable only when none of its turns declares a
    `setup_note`; the rest are counted as `not_yet_automated_cases`, never
    executed and never scored as passing.
    """
    automated_cases = 0
    not_yet_automated_cases = 0
    safety_findings: list[EvaluatorFinding] = []
    semantic_findings: list[EvaluatorFinding] = []
    trace_present = 0
    trace_total = 0
    all_latencies_ms: list[float] = []

    for case in dataset.cases:
        if any(turn.setup_note is not None for turn in case.turns):
            not_yet_automated_cases += 1
            continue
        automated_cases += 1
        run = _run_case(case, client_factory)
        for finding in (*run.findings, *run.expected_behavior_findings):
            bucket = (
                safety_findings if finding.evaluator in _SAFETY_EVALUATORS else semantic_findings
            )
            bucket.append(finding)
        for trace_id in run.trace_ids:
            trace_total += 1
            if trace_id is not None:
                trace_present += 1
        all_latencies_ms.extend(run.turn_latencies_ms)

    trace_delivery = ScorecardCategory(
        name="trace_delivery",
        passed=trace_present,
        failed=trace_total - trace_present,
        total=trace_total,
        pass_rate=(trace_present / trace_total) if trace_total else 1.0,
    )
    latency_ms = (
        {
            "mean": fmean(all_latencies_ms),
            "min": min(all_latencies_ms),
            "max": max(all_latencies_ms),
            "count": float(len(all_latencies_ms)),
        }
        if all_latencies_ms
        else {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    )
    return EvaluationScorecard(
        dataset_version=dataset.dataset_version,
        evaluator_version=evaluator_version,
        source_versions=dict(source_versions or {}),
        generated_at=datetime.now(UTC),
        total_cases=len(dataset.cases),
        automated_cases=automated_cases,
        not_yet_automated_cases=not_yet_automated_cases,
        safety=_category("safety", safety_findings),
        semantic_correctness=_category("semantic_correctness", semantic_findings),
        trace_delivery=trace_delivery,
        latency_ms=latency_ms,
        token_cost={
            "total_tokens": 0,
            "note": "The deterministic runner makes no model calls.",
        },
    )

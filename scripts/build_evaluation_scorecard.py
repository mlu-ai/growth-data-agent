"""Combine CI evaluator outputs into one safe, reviewable artifact."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from growth_data_agent.evaluation_ci import (
    build_configuration_versions,
    compare_approved_baseline,
    evaluation_tier_from_environment,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_DEFAULT_OUTPUT = _REPOSITORY_ROOT / "evaluation-scorecard.json"
_SUITE_FILES = ("governed.json", "rag.json", "trajectory.json")


def _read_suites(directory: Path) -> dict[str, Any]:
    suites = {}
    for filename in _SUITE_FILES:
        path = directory / filename
        if not path.exists():
            suites[path.stem] = {"status": "not_run"}
            continue
        try:
            suites[path.stem] = json.loads(path.read_text())
        except ValueError:
            suites[path.stem] = {"status": "invalid_output"}
    return suites


def _deterministic_failures(suites: dict[str, Any]) -> list[str]:
    failures = []
    for suite_name in ("governed", "rag", "trajectory"):
        if suites.get(suite_name, {}).get("status") in {"not_run", "invalid_output"}:
            failures.append(f"{suite_name}.execution")
    governed = suites.get("governed", {})
    for category in ("safety", "semantic_correctness", "trace_delivery"):
        if (governed.get(category) or {}).get("failed", 0) > 0:
            failures.append(f"governed.{category}")
    trajectory = suites.get("trajectory", {})
    if (trajectory.get("trajectory", {}).get("trajectory", {}) or {}).get("failed", 0) > 0:
        failures.append("trajectory.trajectory")
    if (trajectory.get("adversarial", {}).get("category", {}).get("failed", 0) or 0) > 0:
        failures.append("trajectory.adversarial")
    return failures


def main() -> int:
    tier = evaluation_tier_from_environment()
    directory = Path(os.environ.get("EVALUATION_SCORECARD_DIR", "artifacts/evaluations"))
    output = Path(os.environ.get("EVALUATION_SCORECARD_PATH", _DEFAULT_OUTPUT))
    versions = build_configuration_versions(
        artifact_path=_ARTIFACT, git_sha=os.environ.get("GITHUB_SHA")
    )
    baseline = Path(
        os.environ.get(
            "APPROVED_BASELINE_PATH", _REPOSITORY_ROOT / "evaluations/approved-baseline.json"
        )
    )
    suites = _read_suites(directory)
    baseline_comparison = compare_approved_baseline(versions, baseline)
    payload = {
        "artifact_type": "ci_evaluation_scorecard",
        "schema_version": "1.0.0",
        "tier": tier.value,
        "split": tier.split.value,
        "configuration_versions": versions,
        "baseline_comparison": baseline_comparison,
        "quality_policy": {
            "deterministic_controls_blocking": True,
            "quality_comparisons_blocking": baseline_comparison["blocking"],
            "uncalibrated_quality_report_only": True,
        },
        "suites": suites,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, default=str, indent=2) + "\n")
    print(f"Wrote non-secret evaluation scorecard to {output}.")
    failures = _deterministic_failures(suites)
    if failures:
        print(f"Blocking deterministic failures: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

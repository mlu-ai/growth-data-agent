from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

from growth_data_agent.evaluation_ci import (
    EvaluationTier,
    build_configuration_versions,
    compare_approved_baseline,
    evaluation_tier_from_environment,
    select_cases_for_tier,
)
from growth_data_agent.evaluation_dataset import EvaluationSplit


def test_evaluation_tier_defaults_to_fast_and_selects_development_cases(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATION_TIER", raising=False)
    monkeypatch.delenv("EVALUATION_SPLIT", raising=False)
    cases = [
        {"case_id": "dev", "split": EvaluationSplit.DEVELOPMENT},
        {"case_id": "held", "split": EvaluationSplit.HELD_OUT},
    ]

    assert evaluation_tier_from_environment() is EvaluationTier.FAST
    assert select_cases_for_tier(cases, EvaluationTier.FAST) == [cases[0]]


def test_held_out_tier_can_be_selected_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_TIER", "held_out")
    cases = [
        {"case_id": "dev", "split": EvaluationSplit.DEVELOPMENT},
        {"case_id": "held", "split": EvaluationSplit.HELD_OUT},
    ]

    assert evaluation_tier_from_environment() is EvaluationTier.HELD_OUT
    assert select_cases_for_tier(cases, EvaluationTier.HELD_OUT) == [cases[1]]


def test_invalid_evaluation_tier_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_TIER", "validation")

    try:
        evaluation_tier_from_environment()
    except ValueError as error:
        assert "EVALUATION_TIER" in str(error)
    else:
        raise AssertionError("invalid evaluation tiers must fail closed")


def test_configuration_versions_include_governed_components(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "semantic.json"
    artifact.write_text(json.dumps({"semantic_version": "2.1.0"}))
    monkeypatch.setenv("LOCAL_MODEL_NAME", "candidate-model")
    monkeypatch.setenv("PROMPT_VERSION", "prompt-7")
    monkeypatch.setenv("FACTOR_VOCABULARY_VERSION", "factor-v3")

    versions = build_configuration_versions(artifact_path=artifact, git_sha="abc123")

    assert versions["model"] == "candidate-model"
    assert versions["prompt"] == "prompt-7"
    assert versions["factor_vocabulary"] == "factor-v3"
    assert versions["semantic_artifact"] == "2.1.0"
    assert versions["workflow"] == "governed-response-v1"
    assert versions["git_sha"] == "abc123"


def test_approved_baseline_comparison_is_report_only_and_records_version_changes(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "approved-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "approved": True,
                "baseline_id": "held-out-v1",
                "configuration_versions": {"model": "baseline-model", "workflow": "v1"},
                "metrics": {"retrieval.recall_at_k": 1.0},
            }
        )
    )

    comparison = compare_approved_baseline(
        {"model": "candidate-model", "workflow": "v1"}, baseline
    )

    assert comparison == {
        "status": "changed",
        "baseline_id": "held-out-v1",
        "approved": True,
        "report_only": True,
        "blocking": False,
        "changed_versions": {
            "model": {"baseline": "baseline-model", "current": "candidate-model"}
        },
    }


def test_missing_approved_baseline_is_explicitly_unconfigured(tmp_path: Path) -> None:
    comparison = compare_approved_baseline({"workflow": "v1"}, tmp_path / "missing.json")

    assert comparison == {
        "status": "not_configured",
        "baseline_id": None,
        "approved": False,
        "report_only": True,
        "blocking": False,
        "changed_versions": {},
    }


def test_aggregate_scorecard_is_safe_and_reports_each_suite(tmp_path: Path, monkeypatch) -> None:
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    (suite_dir / "governed.json").write_text(
        json.dumps(
            {
                "safety": {"failed": 0},
                "semantic_correctness": {"failed": 0},
                "trace_delivery": {"failed": 0},
            }
        )
    )
    (suite_dir / "rag.json").write_text(json.dumps({"retrieval": {"failed": 0}}))
    (suite_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "trajectory": {"trajectory": {"failed": 0}},
                "adversarial": {"category": {"failed": 0}},
            }
        )
    )
    output = tmp_path / "scorecard.json"
    monkeypatch.setenv("EVALUATION_TIER", "held_out")
    monkeypatch.setenv("EVALUATION_SCORECARD_DIR", str(suite_dir))
    monkeypatch.setenv("EVALUATION_SCORECARD_PATH", str(output))
    monkeypatch.setenv("APPROVED_BASELINE_PATH", str(tmp_path / "missing.json"))

    exit_code = run_path(
        str(Path(__file__).parents[1] / "scripts/build_evaluation_scorecard.py")
    )["main"]()

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["tier"] == "held_out"
    assert payload["split"] == "held_out"
    assert set(payload["suites"]) == {"governed", "rag", "trajectory"}
    assert payload["quality_policy"]["uncalibrated_quality_report_only"] is True

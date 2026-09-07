"""Shared policy for fast and held-out evaluation runs in CI.

The CI tier is deliberately separate from evaluator implementations. It only
chooses the versioned dataset split and records the configuration identities
needed to compare a run with an approved baseline. Quality comparisons remain
report-only until a calibrated baseline explicitly changes that policy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from .evaluation_dataset import EvaluationSplit

_CaseT = TypeVar("_CaseT")


class EvaluationTier(StrEnum):
    """The supported CI tiers and their isolated dataset partitions."""

    FAST = "fast"
    HELD_OUT = "held_out"

    @property
    def split(self) -> EvaluationSplit:
        return (
            EvaluationSplit.DEVELOPMENT
            if self is EvaluationTier.FAST
            else EvaluationSplit.HELD_OUT
        )


def evaluation_tier_from_environment() -> EvaluationTier:
    """Read the CI tier, defaulting to the deterministic PR tier."""
    configured = os.environ.get("EVALUATION_TIER")
    if configured is None:
        configured_split = os.environ.get("EVALUATION_SPLIT")
        if configured_split is None or configured_split == EvaluationSplit.DEVELOPMENT:
            return EvaluationTier.FAST
        if configured_split == EvaluationSplit.HELD_OUT:
            return EvaluationTier.HELD_OUT
        configured = configured_split
    normalized = configured.casefold().replace("-", "_")
    if normalized == EvaluationTier.FAST:
        return EvaluationTier.FAST
    if normalized in {"full", EvaluationTier.HELD_OUT}:
        return EvaluationTier.HELD_OUT
    raise ValueError(
        "EVALUATION_TIER must be 'fast' or 'held_out' (the optional "
        "EVALUATION_SPLIT accepts 'development' or 'held_out')."
    )


def select_cases_for_tier(
    cases: Sequence[_CaseT], tier: EvaluationTier
) -> list[_CaseT]:
    """Return only cases in the tier's split without mutating the dataset."""
    selected = []
    for case in cases:
        split = case.get("split") if isinstance(case, Mapping) else getattr(case, "split", None)
        if split == tier.split or str(split) == tier.split.value:
            selected.append(case)
    return selected


def build_configuration_versions(
    *, artifact_path: Path | None = None, git_sha: str | None = None
) -> dict[str, str]:
    """Build a safe, complete set of identities for a CI evaluation run."""
    versions = {
        "model": os.environ.get("EVALUATION_MODEL_VERSION")
        or os.environ.get("LOCAL_MODEL_NAME")
        or os.environ.get("OLLAMA_MODEL_NAME")
        or "deterministic",
        "prompt": os.environ.get("PROMPT_VERSION", "repo"),
        "embedding_model": os.environ.get(
            "RAGAS_JUDGE_EMBEDDING_MODEL_NAME", "not_configured"
        ),
        "embedding_version": os.environ.get("EMBEDDING_VERSION", "unknown"),
        "chunking_strategy": os.environ.get("CHUNKING_STRATEGY_VERSION", "fixed-chunk-v1"),
        "reranker_model": os.environ.get(
            "EVALUATION_RERANKER_MODEL",
            os.environ.get("OLLAMA_RERANKER_MODEL_NAME", "deterministic-cross-encoder"),
        ),
        "reranker_version": os.environ.get("RERANKER_VERSION", "1"),
        "factor_vocabulary": os.environ.get("FACTOR_VOCABULARY_VERSION", "hardcoded-v1"),
        "workflow": os.environ.get("WORKFLOW_VERSION", "governed-response-v1"),
    }
    if artifact_path is not None:
        try:
            artifact = json.loads(artifact_path.read_text())
        except (OSError, ValueError):
            versions["semantic_artifact"] = "unavailable"
        else:
            versions["semantic_artifact"] = str(artifact.get("semantic_version", "unknown"))
    else:
        versions["semantic_artifact"] = "unknown"
    if git_sha:
        versions["git_sha"] = git_sha
    return versions


def compare_approved_baseline(
    current_versions: Mapping[str, str], baseline_path: Path
) -> dict[str, Any]:
    """Compare configuration identities without inventing quality thresholds."""
    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, ValueError):
        return {
            "status": "not_configured",
            "baseline_id": None,
            "approved": False,
            "report_only": True,
            "blocking": False,
            "changed_versions": {},
        }

    baseline_versions = baseline.get("configuration_versions")
    if not isinstance(baseline_versions, Mapping):
        baseline_versions = {}
    changed_versions = {
        key: {"baseline": baseline_versions.get(key), "current": value}
        for key, value in current_versions.items()
        if baseline_versions.get(key) != value
    }
    approved = baseline.get("approved") is True
    blocking = approved and os.environ.get("EVALUATION_QUALITY_GATE_ENABLED") == "1"
    return {
        "status": "changed" if changed_versions else "unchanged",
        "baseline_id": baseline.get("baseline_id"),
        "approved": approved,
        "report_only": not blocking,
        "blocking": blocking,
        "changed_versions": changed_versions,
    }


def write_suite_scorecard(name: str, payload: Mapping[str, Any]) -> Path | None:
    """Write one evaluator's redacted payload when CI artifact output is enabled."""
    directory = os.environ.get("EVALUATION_SCORECARD_DIR")
    if not directory:
        return None
    path = Path(directory) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"suite": name, **payload}, default=str, indent=2) + "\n")
    return path

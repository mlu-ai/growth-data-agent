"""Structural validation for the versioned Governed Evaluation Dataset (issue #84).

These tests prove AC5: immutable versions, required metadata, split isolation,
and that the dataset is never runtime evidence — not whether any case can
currently be executed against a live harness (that is issue #85's scope).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from growth_data_agent.evaluation_dataset import (
    EvaluationCaseCategory,
    EvaluationDatasetStore,
    EvaluationSplit,
    GovernedEvaluationDataset,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/dataset/v1/cases.json"
_SOURCE_DIR = _REPOSITORY_ROOT / "src/growth_data_agent"


@pytest.fixture(scope="module")
def dataset() -> GovernedEvaluationDataset:
    return EvaluationDatasetStore(_DATASET_PATH).load()


def test_dataset_file_exists_and_parses() -> None:
    assert _DATASET_PATH.exists(), f"Expected a published dataset at {_DATASET_PATH}"
    dataset = EvaluationDatasetStore(_DATASET_PATH).load()
    assert dataset.artifact_type == "governed_evaluation_dataset"


def test_dataset_version_matches_its_path_segment(dataset: GovernedEvaluationDataset) -> None:
    version_directory = _DATASET_PATH.parent.name
    assert version_directory == "v1"
    assert dataset.dataset_version.split(".")[0] == "1", (
        f"dataset_version {dataset.dataset_version!r} does not match the {version_directory!r} "
        "path segment — a content change must bump both together, never edit v1 in place."
    )


def test_case_count_is_roughly_sixty(dataset: GovernedEvaluationDataset) -> None:
    assert 55 <= len(dataset.cases) <= 65


def test_every_case_category_is_represented(dataset: GovernedEvaluationDataset) -> None:
    represented = {case.category for case in dataset.cases}
    assert represented == set(EvaluationCaseCategory)


def test_every_split_is_non_empty(dataset: GovernedEvaluationDataset) -> None:
    represented = {case.split for case in dataset.cases}
    assert represented == set(EvaluationSplit)


def test_case_ids_are_unique(dataset: GovernedEvaluationDataset) -> None:
    case_ids = [case.case_id for case in dataset.cases]
    assert len(case_ids) == len(set(case_ids))


def test_overlap_sample_cases_have_two_independent_reviewer_labels(
    dataset: GovernedEvaluationDataset,
) -> None:
    overlap_cases = [case for case in dataset.cases if case.overlap_sample]
    assert overlap_cases, "Expected at least one overlap-sample case."
    for case in overlap_cases:
        reviewer_ids = {label.reviewer_id for label in case.reviewer_labels}
        assert len(case.reviewer_labels) == 2
        assert len(reviewer_ids) == 2, (
            f"Overlap case {case.case_id!r} must have 2 distinct reviewer ids, "
            f"got {reviewer_ids}."
        )


def test_non_overlap_cases_have_exactly_one_reviewer_label(
    dataset: GovernedEvaluationDataset,
) -> None:
    for case in dataset.cases:
        if not case.overlap_sample:
            assert len(case.reviewer_labels) == 1, case.case_id


def test_every_case_has_required_provenance_and_approval(
    dataset: GovernedEvaluationDataset,
) -> None:
    for case in dataset.cases:
        assert case.provenance.source_reference, case.case_id
        assert case.permitted_scope, case.case_id
        assert case.approval.approver, case.case_id
        assert len(case.turns) >= 1, case.case_id


def test_dataset_is_never_imported_by_runtime_source_files() -> None:
    """AC5's 'never runtime evidence' claim: no file under src/growth_data_agent
    other than evaluation_dataset.py itself may import it or the evaluations
    directory — the dataset is offline, reviewed content, not request-serving
    evidence."""
    offending_files: list[str] = []
    for path in sorted(_SOURCE_DIR.glob("*.py")):
        if path.name == "evaluation_dataset.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            if any(name and "evaluation_dataset" in name for name in names):
                offending_files.append(path.name)
    assert not offending_files, (
        f"These runtime source files must never import evaluation_dataset: {offending_files}"
    )


def test_rubric_shared_criteria_and_route_specific_criteria_cover_every_category(
    dataset: GovernedEvaluationDataset,
) -> None:
    assert len(dataset.rubric.shared_criteria) == 5
    assert set(dataset.rubric.route_specific_criteria) == set(EvaluationCaseCategory)
    for criteria in dataset.rubric.route_specific_criteria.values():
        assert criteria, "Every category must have at least one route-specific criterion."

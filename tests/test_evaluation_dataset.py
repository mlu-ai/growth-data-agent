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


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` — a block whose
    imports never execute at runtime, only for static type checkers."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _iter_runtime_nodes(node: ast.AST):
    """Walk the tree like `ast.walk`, but never descend into an
    `if TYPE_CHECKING:` block's body — those imports never execute."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking_guard(child.test):
            continue
        yield from _iter_runtime_nodes(child)


def _local_module_imports(path: Path) -> set[str]:
    """Names this file imports **at runtime** that could refer to another module
    in this package (bare module names for `import x` / relative `from .x import
    y`) — third-party absolute imports (`from fastapi import ...`) are filtered
    out by the caller, which only follows names that resolve to an actual file in
    _SOURCE_DIR. Imports inside an `if TYPE_CHECKING:` block are excluded: they
    never run, so they can't make anything reachable at runtime."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in _iter_runtime_nodes(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level > 0:
            names.add(node.module.split(".")[0])
    return names


def test_dataset_is_never_reachable_from_the_request_serving_entrypoint() -> None:
    """AC5's 'never runtime evidence' claim: evaluation_dataset must never be
    reachable from main.py's real import graph — the module the ASGI app and
    every request-serving path actually loads. This is deliberately a graph
    walk from the live entrypoint, not a hand-maintained file blocklist: an
    offline-only consumer (like evaluation_runner.py) is naturally exempt
    without needing this test edited every time one is added, and a future
    accidental import from ANY request-serving module — not just main.py or
    service.py by name — is still caught.
    """
    visited: set[str] = set()
    to_visit = ["main"]
    while to_visit:
        module_name = to_visit.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        module_path = _SOURCE_DIR / f"{module_name}.py"
        if not module_path.exists():
            continue
        for imported in _local_module_imports(module_path):
            if (_SOURCE_DIR / f"{imported}.py").exists() and imported not in visited:
                to_visit.append(imported)

    assert "evaluation_dataset" not in visited, (
        "evaluation_dataset is reachable from main.py's import graph: "
        f"{sorted(visited)}"
    )


def test_rubric_shared_criteria_and_route_specific_criteria_cover_every_category(
    dataset: GovernedEvaluationDataset,
) -> None:
    assert len(dataset.rubric.shared_criteria) == 5
    assert set(dataset.rubric.route_specific_criteria) == set(EvaluationCaseCategory)
    for criteria in dataset.rubric.route_specific_criteria.values():
        assert criteria, "Every category must have at least one route-specific criterion."

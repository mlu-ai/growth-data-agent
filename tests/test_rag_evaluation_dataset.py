"""Structural validation for the versioned RAG Evaluation Dataset (issue #86).

Mirrors tests/test_evaluation_dataset.py's AC5-style proofs, scoped to this
smaller, retrieval-shaped dataset: immutable version/path consistency,
required metadata, split representation, unique case ids, gold relevance
keyed by Evidence Revision (not chunk alone), and never runtime evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import modules_reachable_from_main

from growth_data_agent.evaluation_dataset import EvaluationSplit
from growth_data_agent.rag_evaluation_dataset import RagEvaluationDataset, RagEvaluationDatasetStore

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/rag_dataset/v1/cases.json"


@pytest.fixture(scope="module")
def dataset() -> RagEvaluationDataset:
    return RagEvaluationDatasetStore(_DATASET_PATH).load()


def test_dataset_file_exists_and_parses() -> None:
    assert _DATASET_PATH.exists(), f"Expected a published dataset at {_DATASET_PATH}"
    dataset = RagEvaluationDatasetStore(_DATASET_PATH).load()
    assert dataset.artifact_type == "rag_evaluation_dataset"


def test_dataset_version_matches_its_path_segment(dataset: RagEvaluationDataset) -> None:
    version_directory = _DATASET_PATH.parent.name
    assert version_directory == "v1"
    assert dataset.dataset_version.split(".")[0] == "1", (
        f"dataset_version {dataset.dataset_version!r} does not match the {version_directory!r} "
        "path segment — a content change must bump both together, never edit v1 in place."
    )


def test_every_split_is_represented_at_least_once(dataset: RagEvaluationDataset) -> None:
    represented = {case.split for case in dataset.cases}
    assert represented == set(EvaluationSplit)


def test_case_ids_are_unique(dataset: RagEvaluationDataset) -> None:
    case_ids = [case.case_id for case in dataset.cases]
    assert len(case_ids) == len(set(case_ids))


def test_every_case_has_required_provenance_and_gold_relevance(
    dataset: RagEvaluationDataset,
) -> None:
    for case in dataset.cases:
        assert case.provenance.source_reference, case.case_id
        assert case.permitted_scope, case.case_id
        assert case.retrieval_query, case.case_id
        assert case.question, case.case_id
        assert len(case.gold_relevant_revisions) >= 1, case.case_id


def test_gold_relevance_is_keyed_by_evidence_revision_not_chunk(
    dataset: RagEvaluationDataset,
) -> None:
    """AC1: gold relevance is primarily Evidence Revision + citation; chunk_id
    is retained only as optional retrieval detail, never required."""
    for case in dataset.cases:
        for revision in case.gold_relevant_revisions:
            assert revision.source_document_id, case.case_id
            assert revision.source_revision, case.case_id
            # chunk_id is explicitly optional — a case with it unset must still validate.


def test_dataset_is_never_reachable_from_the_request_serving_entrypoint() -> None:
    """AC-analogous 'never runtime evidence' claim for the RAG dataset and its
    evaluator: neither is reachable from main.py's real import graph."""
    visited = modules_reachable_from_main()
    offenders = {"rag_evaluation_dataset", "rag_evaluation"} & visited
    assert not offenders, (
        f"These offline-only modules are reachable from main.py's import graph: {offenders}"
    )

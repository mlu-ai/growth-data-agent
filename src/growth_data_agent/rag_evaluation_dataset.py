"""Typed contracts for the versioned RAG Evaluation Dataset (issue #86).

Separate from `evaluation_dataset.py`'s Governed Evaluation Dataset (issue
#84): a RAG Evaluation Case is a retrieval query plus gold-relevant Evidence
Revisions, not a request/response turn, so it gets its own small schema
rather than retrofitting #84's per-category validation contract. This
module is never imported by request-serving code — see
docs/adr/0014-rag-evaluation-separates-retrieval-from-generation.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation_dataset import EvaluationCaseProvenance, EvaluationSplit


class RelevantEvidenceRevision(BaseModel):
    """One gold-relevant item, keyed by Evidence Revision identity —
    `(source_document_id, source_revision)` — the same identity this
    codebase already authorizes and de-duplicates by (see
    `lightrag.py`'s `_revision_key`). `chunk_id` is retained only as
    optional retrieval detail, never part of the relevance key itself."""

    model_config = ConfigDict(extra="forbid")

    source_document_id: str = Field(min_length=1, max_length=256)
    source_revision: str = Field(min_length=1, max_length=128)
    chunk_id: str | None = Field(default=None, max_length=256)


class RagEvaluationCase(BaseModel):
    """A retrieval query plus a full governed question, so the same case can
    score both raw retrieval (IR metrics) and grounded generation (RAGAS)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=128)
    agent_user_id: str = Field(min_length=1, max_length=128)
    product: str = Field(min_length=1, max_length=64)
    region: str = Field(min_length=1, max_length=64)
    retrieval_query: str = Field(min_length=1, max_length=512)
    question: str = Field(min_length=1, max_length=512)
    permitted_scope: str = Field(min_length=1, max_length=256)
    k: int = Field(default=3, gt=0, le=10)
    minimum_recall_at_k: float = Field(default=1.0, ge=0, le=1)
    gold_relevant_revisions: list[RelevantEvidenceRevision] = Field(min_length=1)
    split: EvaluationSplit
    provenance: EvaluationCaseProvenance


class RagEvaluationDataset(BaseModel):
    """The versioned RAG Evaluation Dataset artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["rag_evaluation_dataset"] = "rag_evaluation_dataset"
    dataset_version: str = Field(min_length=1, max_length=32)
    published_at: date
    cases: list[RagEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self) -> RagEvaluationDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("RAG Evaluation Case ids must be unique within the dataset.")

        represented_splits = {case.split for case in self.cases}
        missing_splits = set(EvaluationSplit) - represented_splits
        if missing_splits:
            raise ValueError(f"Splits with no cases: {sorted(missing_splits)}.")

        return self


class RagEvaluationDatasetStore:
    """Load the versioned RAG Evaluation Dataset artifact from disk.

    Re-reads on every `.load()` call, matching `SemanticArtifactStore` and
    `EvaluationDatasetStore` — no caching.
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> RagEvaluationDataset:
        return RagEvaluationDataset.model_validate_json(self.path.read_text())

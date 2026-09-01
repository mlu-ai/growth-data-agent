"""Typed contracts for the versioned Governed Evaluation Dataset (issue #84).

This module is never imported by request-serving code (`service.py`,
`main.py`, or anything reachable from them) — the dataset is reviewed,
versioned content for offline evaluation, never runtime evidence. See
docs/adr/0012-governed-evaluation-dataset-is-versioned-and-never-runtime-evidence.md.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ResultClassification


class EvaluationSplit(StrEnum):
    """Isolated dataset partitions so prompts/retrievers/judges can't be tuned
    against their own release evidence."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


class ErrorTaxonomyCategory(StrEnum):
    """The versioned Error Taxonomy from issue #61's Implementation Decisions."""

    GOVERNANCE = "governance"
    SEMANTIC = "semantic"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    TRAJECTORY = "trajectory"
    EXPERIENCE_COST = "experience_cost"


class EvaluationCaseCategory(StrEnum):
    """The 11 stratified case categories named in issue #84's AC1."""

    CANONICAL_DEFINITION = "canonical_definition"
    DRIVER_DECOMPOSITION = "driver_decomposition"
    HYPOTHESIS_INVESTIGATION = "hypothesis_investigation"
    CANDIDATE_CAUSAL_FACTOR_RANKING = "candidate_causal_factor_ranking"
    ACTIVE_INVESTIGATION = "active_investigation"
    OPPORTUNITY_ESTIMATE = "opportunity_estimate"
    CLARIFICATION = "clarification"
    LIMITATION = "limitation"
    REFUSAL = "refusal"
    ACCESS_CHANGE = "access_change"
    STALE_REVISION = "stale_revision"


class OwnerRole(StrEnum):
    """The two approval authorities named in issue #84's AC4."""

    DATA_OWNER = "data_owner"
    EVIDENCE_OR_PRODUCT_OWNER = "evidence_or_product_owner"


class RubricCriterion(StrEnum):
    """The shared rubric from issue #61's Implementation Decisions."""

    SAFETY = "safety"
    CORRECTNESS = "correctness"
    CITATION = "citation"
    UNCERTAINTY = "uncertainty"
    RELEVANCE = "relevance"


class EvaluationCaseProvenance(BaseModel):
    """Where a case came from — synthetic scenario, reviewed trace, or an
    adversarial construction — never itself runtime evidence."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["synthetic", "reviewed_trace", "adversarial"]
    source_reference: str = Field(min_length=1, max_length=256)
    notes: str = Field(default="", max_length=1024)


class ExpectedBehavior(BaseModel):
    """Extends the informal `evaluations/fixtures.json` expectation shape
    (status_code/result_classification/fields/contains/not_contains) with types."""

    model_config = ConfigDict(extra="forbid")

    status_code: int = 200
    result_classification: ResultClassification | None = None
    fields: dict[str, object] = Field(default_factory=dict)
    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)


class EvaluationTurn(BaseModel):
    """One request in a case's turn sequence (length 1 for single-turn cases)."""

    model_config = ConfigDict(extra="forbid")

    request: dict[str, object]
    expected: ExpectedBehavior
    setup_note: str | None = Field(default=None, max_length=1024)


class OwnerApproval(BaseModel):
    """Explicit sign-off per AC4 — data owners approve semantic/Opportunity
    Estimate labels; evidence or product owners approve hypothesis/evidence labels."""

    model_config = ConfigDict(extra="forbid")

    approved_by_role: OwnerRole
    approver: str = Field(min_length=1, max_length=128)
    approved_at: date


class ReviewerLabel(BaseModel):
    """One reviewer's rubric scores for a case. `rubric_scores` keys are the
    shared RubricCriterion values plus any route-specific criteria names."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1, max_length=128)
    rubric_scores: dict[str, str] = Field(min_length=1)
    notes: str = Field(default="", max_length=1024)


class Rubric(BaseModel):
    """One shared rubric plus route-specific criteria per case category."""

    model_config = ConfigDict(extra="forbid")

    shared_criteria: tuple[RubricCriterion, ...]
    route_specific_criteria: dict[EvaluationCaseCategory, tuple[str, ...]]
    scale: tuple[str, ...]


class EvaluationCase(BaseModel):
    """One reviewable Evaluation Case: provenance, permitted scope, expected
    observable behaviour, a primary Error Taxonomy category, owner approval,
    and one or two independent reviewer labels."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=128)
    category: EvaluationCaseCategory
    split: EvaluationSplit
    overlap_sample: bool = False
    provenance: EvaluationCaseProvenance
    permitted_scope: str = Field(min_length=1, max_length=256)
    turns: list[EvaluationTurn] = Field(min_length=1)
    primary_error_taxonomy: ErrorTaxonomyCategory
    secondary_error_taxonomy_notes: str | None = Field(default=None, max_length=1024)
    approval: OwnerApproval
    reviewer_labels: list[ReviewerLabel] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _validate_reviewer_overlap(self) -> EvaluationCase:
        distinct_reviewers = {label.reviewer_id for label in self.reviewer_labels}
        if self.overlap_sample:
            if len(self.reviewer_labels) != 2 or len(distinct_reviewers) != 2:
                raise ValueError(
                    f"Overlap sample case {self.case_id!r} must have exactly 2 "
                    "reviewer labels from 2 distinct reviewers."
                )
        elif len(self.reviewer_labels) != 1:
            raise ValueError(
                f"Non-overlap case {self.case_id!r} must have exactly 1 reviewer label."
            )
        return self


class GovernedEvaluationDataset(BaseModel):
    """The versioned Governed Evaluation Dataset artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["governed_evaluation_dataset"] = "governed_evaluation_dataset"
    dataset_version: str = Field(min_length=1, max_length=32)
    evaluation_owner: str = Field(min_length=1, max_length=128)
    published_at: date
    rubric: Rubric
    error_taxonomy: tuple[ErrorTaxonomyCategory, ...]
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self) -> GovernedEvaluationDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation Case ids must be unique within the dataset.")

        represented_splits = {case.split for case in self.cases}
        missing_splits = set(EvaluationSplit) - represented_splits
        if missing_splits:
            raise ValueError(f"Splits with no cases: {sorted(missing_splits)}.")

        represented_categories = {case.category for case in self.cases}
        missing_categories = set(EvaluationCaseCategory) - represented_categories
        if missing_categories:
            raise ValueError(f"Categories with no cases: {sorted(missing_categories)}.")

        return self


class EvaluationDatasetStore:
    """Load the versioned Governed Evaluation Dataset artifact from disk.

    Re-reads on every `.load()` call, matching `SemanticArtifactStore` — no
    caching, so a hand-edited file is never trusted as stale-but-loaded state.
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> GovernedEvaluationDataset:
        return GovernedEvaluationDataset.model_validate_json(self.path.read_text())

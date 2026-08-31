"""Deterministic Candidate Causal Factor extraction and ranking-eligibility validation.

Extraction never invents a Factor ID, timing, population, or eligibility decision; it
only reads fields an already-authorized `EvidenceDocument` already carries. A separate
deterministic validator decides ranking eligibility, so extraction and eligibility stay
independently testable and neither can silently widen the other's authority.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Protocol

from .contracts import (
    CandidateCausalFactor,
    DriverContribution,
    EvidenceScope,
    FactorVocabularyCategory,
)
from .evidence import EvidenceAccessFilter, EvidenceDocument, _citation

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Ordered, priority-first keyword table — first match wins. A small, reviewed list per
# spec #72; not a live/approvable taxonomy in this delivery.
_VOCABULARY_KEYWORDS: tuple[tuple[FactorVocabularyCategory, tuple[str, ...]], ...] = (
    (FactorVocabularyCategory.CAMPAIGN, ("campaign",)),
    (FactorVocabularyCategory.ONBOARDING, ("onboarding",)),
    (
        FactorVocabularyCategory.PRODUCT_RELEASE_OR_REGRESSION,
        ("regression", "release", "rollout"),
    ),
    (
        FactorVocabularyCategory.PROVISIONING_OR_ENTITLEMENT,
        ("provisioning", "entitlement"),
    ),
    (
        FactorVocabularyCategory.BILLING_OR_SUBSCRIPTION,
        ("billing", "subscription", "invoice"),
    ),
    (
        FactorVocabularyCategory.IDENTITY_OR_ACCESS,
        ("identity", "sso", "authentication", "access control"),
    ),
    (FactorVocabularyCategory.INCIDENT, ("incident",)),
)


def _normalize(value: str) -> str:
    return _NON_ALNUM.sub("-", value.casefold()).strip("-")


def _match_category(document: EvidenceDocument) -> FactorVocabularyCategory | None:
    title = document.title.casefold()
    text = document.text.casefold()
    for category, keywords in _VOCABULARY_KEYWORDS:
        if any(keyword in title for keyword in keywords):
            return category
    for category, keywords in _VOCABULARY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return None


@dataclass(frozen=True)
class ProvisionalFactorRecord:
    """A revision-linked derived projection from one Evidence Revision.

    Never a source or permission authority — it exists only to be validated for
    ranking eligibility and, if eligible, projected into a public `CandidateCausalFactor`.
    """

    factor_id: str
    category: FactorVocabularyCategory | None
    proposed_mechanism: str
    factor_occurrence_time: date | None
    scope: EvidenceScope | None
    document: EvidenceDocument


class FactorExtractor(Protocol):
    """Propose a typed Provisional Factor Record without deciding ranking eligibility."""

    def extract(self, document: EvidenceDocument) -> ProvisionalFactorRecord | None: ...


class RuleBasedFactorExtractor:
    """Deterministic extractor over EvidenceDocument's existing structured fields.

    No model call. Every field is read directly from, or trivially derived from, data
    the document already carries — never generated or inferred beyond that.
    """

    def extract(self, document: EvidenceDocument) -> ProvisionalFactorRecord | None:
        factor_id = (
            f"{document.product.casefold()}:{_normalize(document.title)}:"
            f"{document.relevant_date:%Y-%m}"
        )
        return ProvisionalFactorRecord(
            factor_id=factor_id,
            category=_match_category(document),
            proposed_mechanism=document.support_explanation,
            factor_occurrence_time=document.relevant_date,
            scope=EvidenceScope(
                product=document.product,
                region=document.region,
                tenant_scope=document.tenant_scope,
            ),
            document=document,
        )


@dataclass(frozen=True)
class DriverMovementWindow:
    """The observed-movement window a Candidate Causal Factor's timing is checked against."""

    start: date
    end: date

    @classmethod
    def from_periods(cls, baseline_period: str, comparison_period: str) -> DriverMovementWindow:
        """Derive the window from the comparison period; baseline_period is accepted for
        interface symmetry with the Driver Decomposition it's read from but is not used
        in the date math — the "observed movement" is the comparison period itself."""
        del baseline_period
        year, month = (int(part) for part in comparison_period.split("-"))
        _, last_day = calendar.monthrange(year, month)
        return cls(start=date(year, month, 1), end=date(year, month, last_day))


@dataclass(frozen=True)
class FactorEligibility:
    eligible: bool
    blocked_reasons: tuple[str, ...]


def validate_ranking_eligibility(
    record: ProvisionalFactorRecord,
    *,
    driver_window: DriverMovementWindow,
    access_filter: EvidenceAccessFilter,
    initial_lookback_days: int = 14,
) -> FactorEligibility:
    """Decide ranking eligibility deterministically, independent of extraction."""
    reasons: list[str] = []
    if record.category is None:
        reasons.append("no_factor_vocabulary_match")
    if record.factor_occurrence_time is None:
        reasons.append("missing_factor_occurrence_time")
    if record.scope is None:
        reasons.append("missing_scope")
    if record.factor_occurrence_time is not None:
        if record.factor_occurrence_time > driver_window.end:
            reasons.append("occurrence_time_after_movement_window")
        if record.factor_occurrence_time < driver_window.start - timedelta(
            days=initial_lookback_days
        ):
            reasons.append("occurrence_time_exceeds_initial_lookback")
    if not access_filter.allows(record.document):
        reasons.append("source_not_currently_authorized")
    return FactorEligibility(eligible=not reasons, blocked_reasons=tuple(reasons))


def build_candidate_causal_factor(
    record: ProvisionalFactorRecord,
    *,
    contribution: DriverContribution,
) -> CandidateCausalFactor:
    """Project an eligible Provisional Factor Record into the public card.

    Callers must only invoke this after `validate_ranking_eligibility` reports eligible.
    """
    assert record.category is not None
    assert record.factor_occurrence_time is not None
    assert record.scope is not None
    return CandidateCausalFactor(
        factor_id=record.factor_id,
        category=record.category,
        documented_change=contribution,
        affected_population=record.scope,
        proposed_mechanism=record.proposed_mechanism,
        factor_occurrence_time=record.factor_occurrence_time,
        citation=_citation(record.document),
        non_causal_caveat=(
            "This Candidate Causal Factor is a falsifiable Hypothesis label, not proof; "
            "the cited evidence supports it but does not establish that it caused the "
            "observed movement."
        ),
    )


def build_evidence_investigation_query(
    *,
    metric_label: str,
    product: str,
    region: str,
    seat_tier: str,
    driver_window: DriverMovementWindow,
    canonical_time_rule: str,
    movement_direction: Literal["decline", "increase"],
) -> str:
    """Build a retrieval query from the driver scope/window, metric prerequisites, and
    Factor Vocabulary — never from an analyst-supplied suspected mechanism. Deterministic:
    identical inputs always produce identical output."""
    vocabulary_terms = " ".join(
        sorted(
            {category.value.replace("_", " ") for category, _ in _VOCABULARY_KEYWORDS}
            | {keyword for _, keywords in _VOCABULARY_KEYWORDS for keyword in keywords}
        )
    )
    return (
        f"{product} {region} {seat_tier} Seat Tier {metric_label} {movement_direction} "
        f"{driver_window.start.isoformat()} to {driver_window.end.isoformat()}. "
        f"Qualifying prerequisite: {canonical_time_rule} "
        f"Factor Vocabulary: {vocabulary_terms}."
    )

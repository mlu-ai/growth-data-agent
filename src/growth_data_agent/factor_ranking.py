"""Deterministic grouping, ranking, and status determination for Candidate Causal Factors.

This is a separate concern from `factors.py`'s extract-then-validate-eligibility contract:
grouping aggregates multiple eligible Provisional Factor Records into candidates, ranking
orders and caps them, and status determination decides supported/contradicted/inconclusive.
None of this invents evidence — it only aggregates, scores, and orders records that already
passed `factors.validate_ranking_eligibility`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from .contracts import (
    CandidateCausalFactor,
    CounterevidenceLevel,
    DriverContribution,
    EvidenceSupportStatus,
    FactorSupportStatus,
    FactorVocabularyCategory,
    PopulationOverlap,
    RankingSignals,
    TemporalAlignment,
)
from .evidence import EvidenceAccessFilter, EvidenceDocument, _citation
from .factors import (
    DriverMovementWindow,
    ProvisionalFactorRecord,
    RuleBasedFactorExtractor,
    validate_ranking_eligibility,
)

_GroupKey = tuple[str, str, str, FactorVocabularyCategory, date]


def _group_key(record: ProvisionalFactorRecord) -> _GroupKey:
    """The 'same documented incident' identity: same product/region/segment/category/date."""
    assert record.category is not None
    assert record.scope is not None
    assert record.factor_occurrence_time is not None
    return (
        record.scope.product,
        record.scope.region,
        record.scope.tenant_scope,
        record.category,
        record.factor_occurrence_time,
    )


def group_factor_id(key: _GroupKey) -> str:
    product, region, tenant_scope, category, occurrence = key
    normalized_region = region.casefold().replace(" ", "-")
    normalized_scope = tenant_scope.casefold().replace(" ", "-")
    return (
        f"{product.casefold()}:{normalized_region}:{normalized_scope}:"
        f"{category.value}:{occurrence:%Y-%m-%d}"
    )


@dataclass(frozen=True)
class CandidateFactorGroup:
    """All eligible records describing the same documented incident."""

    key: _GroupKey
    records: tuple[ProvisionalFactorRecord, ...]


def group_eligible_records(
    documents: Iterable[EvidenceDocument],
    *,
    driver_window: DriverMovementWindow,
    access_filter: EvidenceAccessFilter,
) -> tuple[CandidateFactorGroup, ...]:
    """Extract, drop ineligible records, and deterministically group the rest."""
    extractor = RuleBasedFactorExtractor()
    grouped: dict[_GroupKey, list[ProvisionalFactorRecord]] = {}
    for document in documents:
        record = extractor.extract(document)
        if record is None:
            continue
        eligibility = validate_ranking_eligibility(
            record, driver_window=driver_window, access_filter=access_filter
        )
        if not eligibility.eligible:
            continue
        grouped.setdefault(_group_key(record), []).append(record)
    return tuple(
        CandidateFactorGroup(key=key, records=tuple(records))
        for key, records in sorted(grouped.items(), key=lambda item: group_factor_id(item[0]))
    )


def _independent_source_count(group: CandidateFactorGroup) -> int:
    """Distinct source documents among SUPPORTS records — repeated chunks of the same
    page must not count as independent sources."""
    return len(
        {
            record.document.source_document_id or record.document.document_id
            for record in group.records
            if record.document.support_status == EvidenceSupportStatus.SUPPORTS
        }
    )


def _has_high_authority_support(group: CandidateFactorGroup) -> bool:
    return any(
        record.document.support_status == EvidenceSupportStatus.SUPPORTS
        and record.document.is_high_authority_operational_record
        for record in group.records
    )


def _has_material_contradiction(group: CandidateFactorGroup) -> bool:
    """A CONTRADICTS document sharing this group's key is material counterevidence for
    this exact documented incident — not any unrelated contradiction elsewhere."""
    return any(
        record.document.support_status == EvidenceSupportStatus.CONTRADICTS
        for record in group.records
    )


def has_informative_evidence(group: CandidateFactorGroup) -> bool:
    """False when every record is plain INCONCLUSIVE — background-only, never a card."""
    return any(
        record.document.support_status != EvidenceSupportStatus.INCONCLUSIVE
        for record in group.records
    )


def determine_status(group: CandidateFactorGroup) -> FactorSupportStatus:
    if _has_material_contradiction(group):
        return FactorSupportStatus.CONTRADICTED
    if _independent_source_count(group) >= 2 or _has_high_authority_support(group):
        return FactorSupportStatus.SUPPORTED
    return FactorSupportStatus.INCONCLUSIVE


def compute_ranking_signals(
    group: CandidateFactorGroup,
    *,
    driver_window: DriverMovementWindow,
    contribution: DriverContribution,
    metric_name: str,
) -> RankingSignals:
    occurrence = group.key[4]
    temporal_alignment = (
        TemporalAlignment.WITHIN_MOVEMENT_WINDOW
        if driver_window.start <= occurrence <= driver_window.end
        else TemporalAlignment.WITHIN_INITIAL_LOOKBACK
    )
    population_overlap = (
        PopulationOverlap.EXACT_SEGMENT_MATCH
        if any(
            record.scope is not None
            and record.scope.region == contribution.region
            and record.scope.tenant_scope.endswith(f"{contribution.seat_tier} Seat Tier Tenants")
            for record in group.records
        )
        else PopulationOverlap.PARTIAL_OR_BROADER_SCOPE
    )
    metric_mechanism_fit = any(
        record.document.metric_name == metric_name for record in group.records
    )
    return RankingSignals(
        temporal_alignment=temporal_alignment,
        population_overlap=population_overlap,
        metric_mechanism_fit=metric_mechanism_fit,
        independent_source_count=min(_independent_source_count(group), 3),
        counterevidence=(
            CounterevidenceLevel.MATERIAL
            if _has_material_contradiction(group)
            else CounterevidenceLevel.NONE
        ),
    )


_STATUS_PRIORITY = {
    FactorSupportStatus.SUPPORTED: 0,
    FactorSupportStatus.CONTRADICTED: 1,
    FactorSupportStatus.INCONCLUSIVE: 2,
}
_TEMPORAL_PRIORITY = {
    TemporalAlignment.WITHIN_MOVEMENT_WINDOW: 0,
    TemporalAlignment.WITHIN_INITIAL_LOOKBACK: 1,
}
_POPULATION_PRIORITY = {
    PopulationOverlap.EXACT_SEGMENT_MATCH: 0,
    PopulationOverlap.PARTIAL_OR_BROADER_SCOPE: 1,
}


def _rank_key(
    status: FactorSupportStatus, signals: RankingSignals, factor_id: str
) -> tuple[int, int, int, int, int, str]:
    return (
        _STATUS_PRIORITY[status],
        -signals.independent_source_count,
        _TEMPORAL_PRIORITY[signals.temporal_alignment],
        _POPULATION_PRIORITY[signals.population_overlap],
        0 if signals.metric_mechanism_fit else 1,
        factor_id,
    )


_SIZING_ELIGIBLE_DRIVER_METRICS = ("jira_new_peu", "confluence_new_peu")


def sizing_eligible_metric_name(
    category: FactorVocabularyCategory, metric_name: str
) -> str | None:
    """The governed event-and-audience mapping: which Eligible Population metric (if
    any) a Candidate Causal Factor's category is reviewed against for this driver
    metric. Returns `None` when no mapping exists — the factor remains a Hypothesis,
    never Sizing Eligible, and no audience is inferred from documents or the graph."""
    if category != FactorVocabularyCategory.PROVISIONING_OR_ENTITLEMENT:
        return None
    if metric_name not in _SIZING_ELIGIBLE_DRIVER_METRICS:
        return None
    return f"{metric_name}_eligible_population"


def _build_card(
    group: CandidateFactorGroup,
    *,
    status: FactorSupportStatus,
    signals: RankingSignals,
    contribution: DriverContribution,
    metric_name: str,
) -> CandidateCausalFactor:
    # Prefer a SUPPORTS record as the anchor for the proposed mechanism/category/
    # occurrence-time text — even in a CONTRADICTED group, the card describes the
    # hypothesis being challenged, not the rebuttal, and record order otherwise
    # reflects retrieval/rerank order, not evidentiary stance.
    anchor = next(
        (
            record
            for record in group.records
            if record.document.support_status == EvidenceSupportStatus.SUPPORTS
        ),
        group.records[0],
    )
    assert anchor.category is not None
    assert anchor.scope is not None
    assert anchor.factor_occurrence_time is not None
    return CandidateCausalFactor(
        factor_id=group_factor_id(group.key),
        category=anchor.category,
        documented_change=contribution,
        affected_population=anchor.scope,
        proposed_mechanism=anchor.proposed_mechanism,
        factor_occurrence_time=anchor.factor_occurrence_time,
        citations=[_citation(record.document) for record in group.records[:3]],
        status=status,
        ranking_signals=signals,
        non_causal_caveat=(
            "This Candidate Causal Factor is a falsifiable Hypothesis label, not proof; "
            "the cited evidence supports or challenges it but does not establish that it "
            "caused the observed movement."
        ),
        sizing_eligible=sizing_eligible_metric_name(anchor.category, metric_name) is not None,
    )


def rank_candidate_causal_factors(
    documents: Iterable[EvidenceDocument],
    *,
    driver_window: DriverMovementWindow,
    access_filter: EvidenceAccessFilter,
    contribution: DriverContribution,
    metric_name: str,
    max_candidates: int = 3,
) -> list[CandidateCausalFactor]:
    """Group, score, rank, and cap eligible evidence into public candidate cards.

    Returns an empty list when no group qualifies — the explicit no-ranked-candidate
    outcome, never a default/placeholder card.
    """
    groups = [
        group
        for group in group_eligible_records(
            documents, driver_window=driver_window, access_filter=access_filter
        )
        if has_informative_evidence(group)
    ]
    scored = [
        (
            group,
            determine_status(group),
            compute_ranking_signals(
                group, driver_window=driver_window, contribution=contribution,
                metric_name=metric_name,
            ),
        )
        for group in groups
    ]
    scored.sort(key=lambda item: _rank_key(item[1], item[2], group_factor_id(item[0].key)))
    return [
        _build_card(
            group,
            status=status,
            signals=signals,
            contribution=contribution,
            metric_name=metric_name,
        )
        for group, status, signals in scored[:max_candidates]
    ]

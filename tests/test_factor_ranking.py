from __future__ import annotations

from datetime import UTC, date, datetime

from growth_data_agent.contracts import DriverContribution, EvidenceSupportStatus
from growth_data_agent.evidence import EvidenceAccessFilter, EvidenceDocument
from growth_data_agent.factor_ranking import (
    determine_status,
    group_eligible_records,
    group_factor_id,
    rank_candidate_causal_factors,
)
from growth_data_agent.factors import DriverMovementWindow

_WINDOW = DriverMovementWindow.from_periods("2026-05", "2026-06")
_CONTRIBUTION = DriverContribution(
    region="APAC",
    seat_tier="51-200",
    baseline_value=800,
    comparison_value=380,
    change=-420,
    contribution_to_decline=420,
    percentage_of_decline=75.0,
)


def _document(
    *,
    document_id: str,
    title: str = "Jira APAC provisioning incident",
    text: str | None = None,
    relevant_date: date = date(2026, 6, 12),
    support_status: EvidenceSupportStatus = EvidenceSupportStatus.SUPPORTS,
    is_high_authority_operational_record: bool = False,
    source_document_id: str | None = None,
) -> EvidenceDocument:
    return EvidenceDocument(
        document_id=document_id,
        metric_name="jira_new_peu",
        title=title,
        text=text or (
            "Paid provisioning errors affected Jira APAC 51-200 Seat Tier Tenants "
            "from 2026-06-10 through 2026-06-12, overlapping the June New PEU decline."
        ),
        product="Jira",
        region="APAC",
        tenant_ids=["tenant-0011"],
        tenant_scope="APAC 51-200 Seat Tier Tenants",
        classification="internal",
        identifier_entitlement="none",
        relevant_date=relevant_date,
        freshness=datetime(2026, 6, 13, tzinfo=UTC),
        support_status=support_status,
        support_explanation="Overlaps the APAC 51-200 Seat Tier Tenant scope and period.",
        is_high_authority_operational_record=is_high_authority_operational_record,
        source_document_id=source_document_id or document_id,
    )


def _permissive_filter() -> EvidenceAccessFilter:
    return EvidenceAccessFilter(
        products=("Jira",),
        regions=("APAC",),
        tenant_ids=("tenant-0011",),
        classifications=("internal",),
        identifier_entitlements=("none",),
    )


def test_groups_records_sharing_product_region_segment_category_date() -> None:
    main = _document(document_id="a")
    appendix = _document(
        document_id="b",
        title="Restricted Jira APAC provisioning incident appendix",
        text="Restricted appendix for the same Jira APAC provisioning incident.",
    )
    groups = group_eligible_records(
        [main, appendix], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    assert len(groups) == 1
    assert {record.document.document_id for record in groups[0].records} == {"a", "b"}


def test_duplicate_source_revision_does_not_count_as_two_independent_sources() -> None:
    original = _document(document_id="jira-incident")
    same_page_chunk = _document(
        document_id="jira-incident-chunk-2", source_document_id="jira-incident"
    )
    groups = group_eligible_records(
        [original, same_page_chunk], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    assert len(groups) == 1
    assert determine_status(groups[0]).value == "inconclusive"


def test_two_independent_supports_yields_supported_status() -> None:
    doc_a = _document(document_id="a")
    doc_b = _document(document_id="b", title="Jira APAC provisioning follow-up")
    groups = group_eligible_records(
        [doc_a, doc_b], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    assert len(groups) == 1
    assert determine_status(groups[0]).value == "supported"


def test_high_authority_operational_record_yields_supported_with_single_source() -> None:
    doc = _document(document_id="a", is_high_authority_operational_record=True)
    groups = group_eligible_records(
        [doc], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    assert len(groups) == 1
    assert determine_status(groups[0]).value == "supported"


def test_material_contradiction_yields_contradicted_status_and_material_counterevidence() -> None:
    supporting = _document(document_id="a")
    contradicting = _document(
        document_id="b",
        title="Jira APAC provisioning incident retrospective",
        support_status=EvidenceSupportStatus.CONTRADICTS,
    )
    contribution = _CONTRIBUTION
    cards = rank_candidate_causal_factors(
        [supporting, contradicting],
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=contribution,
        metric_name="jira_new_peu",
    )
    assert len(cards) == 1
    assert cards[0].status.value == "contradicted"
    assert cards[0].ranking_signals.counterevidence.value == "material"
    assert cards[0].proposed_mechanism == supporting.support_explanation


def test_card_anchors_on_the_supports_record_regardless_of_evidence_order() -> None:
    """The card's mechanism/category must describe the hypothesis being challenged,
    not whichever record happens to be first — even when a CONTRADICTS document is
    retrieved/reranked ahead of the SUPPORTS document it's about."""
    supporting = _document(
        document_id="a", text="Supporting mechanism text distinct from the rebuttal."
    )
    contradicting = _document(
        document_id="b",
        title="Jira APAC provisioning incident retrospective",
        support_status=EvidenceSupportStatus.CONTRADICTS,
    )
    cards = rank_candidate_causal_factors(
        [contradicting, supporting],  # CONTRADICTS listed first
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=_CONTRIBUTION,
        metric_name="jira_new_peu",
    )
    assert len(cards) == 1
    assert cards[0].proposed_mechanism == supporting.support_explanation


def test_background_only_evidence_group_is_excluded_entirely() -> None:
    doc = _document(document_id="a", support_status=EvidenceSupportStatus.INCONCLUSIVE)
    cards = rank_candidate_causal_factors(
        [doc],
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=_CONTRIBUTION,
        metric_name="jira_new_peu",
    )
    assert cards == []


def test_no_qualified_group_returns_empty_list_not_a_default_card() -> None:
    cards = rank_candidate_causal_factors(
        [],
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=_CONTRIBUTION,
        metric_name="jira_new_peu",
    )
    assert cards == []


def test_ranking_caps_at_three_and_orders_supported_before_contradicted_before_inconclusive() -> (
    None
):
    supported = _document(
        document_id="supported",
        title="Jira APAC provisioning incident",
        is_high_authority_operational_record=True,
    )
    contradicted_support = _document(
        document_id="contradicted-support",
        title="Jira APAC billing incident",
        relevant_date=date(2026, 6, 13),
    )
    contradicted_refute = _document(
        document_id="contradicted-refute",
        title="Jira APAC billing incident retrospective",
        relevant_date=date(2026, 6, 13),
        support_status=EvidenceSupportStatus.CONTRADICTS,
    )
    inconclusive_lone = _document(
        document_id="inconclusive",
        title="Jira APAC identity incident",
        relevant_date=date(2026, 6, 14),
    )
    fourth_supported = _document(
        document_id="fourth-supported",
        title="Jira APAC onboarding incident",
        relevant_date=date(2026, 6, 15),
        is_high_authority_operational_record=True,
    )
    cards = rank_candidate_causal_factors(
        [
            supported,
            contradicted_support,
            contradicted_refute,
            inconclusive_lone,
            fourth_supported,
        ],
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=_CONTRIBUTION,
        metric_name="jira_new_peu",
    )
    assert len(cards) == 3
    statuses = [card.status.value for card in cards]
    assert statuses.index("supported") < statuses.index("contradicted")
    # the lone-source, non-authority group ranks last and is capped out
    assert "inconclusive" not in statuses


def test_ranking_is_deterministic_across_repeated_calls() -> None:
    doc_a = _document(document_id="a", is_high_authority_operational_record=True)
    doc_b = _document(
        document_id="b", title="Jira APAC billing incident", relevant_date=date(2026, 6, 13)
    )
    kwargs = dict(
        driver_window=_WINDOW,
        access_filter=_permissive_filter(),
        contribution=_CONTRIBUTION,
        metric_name="jira_new_peu",
    )
    first = rank_candidate_causal_factors([doc_a, doc_b], **kwargs)
    second = rank_candidate_causal_factors([doc_b, doc_a], **kwargs)
    assert [card.factor_id for card in first] == [card.factor_id for card in second]


def test_group_factor_id_is_stable_given_identical_evidence() -> None:
    doc = _document(document_id="a")
    groups = group_eligible_records(
        [doc], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    first_id = group_factor_id(groups[0].key)
    groups_again = group_eligible_records(
        [doc], driver_window=_WINDOW, access_filter=_permissive_filter()
    )
    assert group_factor_id(groups_again[0].key) == first_id

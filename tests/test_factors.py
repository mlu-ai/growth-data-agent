from __future__ import annotations

from dataclasses import replace
from datetime import date

from growth_data_agent.contracts import FactorVocabularyCategory
from growth_data_agent.evidence import EvidenceAccessFilter, EvidenceScope
from growth_data_agent.factors import (
    DriverMovementWindow,
    ProvisionalFactorRecord,
    RuleBasedFactorExtractor,
    build_evidence_investigation_query,
    validate_ranking_eligibility,
)
from growth_data_agent.synthetic import evidence_corpus

_CORPUS = {document.document_id: document for document in evidence_corpus()}


def _permissive_filter(document) -> EvidenceAccessFilter:
    """An access filter that allows exactly the given document, nothing more."""
    return EvidenceAccessFilter(
        products=(document.product,),
        regions=(document.region,),
        tenant_ids=tuple(document.tenant_ids),
        classifications=(document.classification,),
        identifier_entitlements=(document.identifier_entitlement,),
    )


def test_extractor_derives_jira_apac_factor() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = RuleBasedFactorExtractor().extract(document)

    assert record is not None
    assert record.factor_id == "jira:jira-apac-paid-provisioning-incident:2026-06"
    assert record.category is FactorVocabularyCategory.PROVISIONING_OR_ENTITLEMENT
    assert record.factor_occurrence_time == date(2026, 6, 12)
    assert record.scope == EvidenceScope(
        product="Jira", region="APAC", tenant_scope="APAC 51-200 Seat Tier Tenants"
    )
    assert record.proposed_mechanism == document.support_explanation


def test_extractor_derives_confluence_americas_campaign_factor() -> None:
    document = _CORPUS["confluence-americas-acquisition-campaign"]
    record = RuleBasedFactorExtractor().extract(document)

    assert record is not None
    assert record.category is FactorVocabularyCategory.CAMPAIGN
    assert record.factor_occurrence_time == date(2026, 6, 15)
    assert record.scope == EvidenceScope(
        product="Confluence", region="Americas", tenant_scope="Americas 11-50 Seat Tier Tenants"
    )


def test_extractor_derives_confluence_emea_regression_factor() -> None:
    document = _CORPUS["confluence-emea-onboarding-email-regression"]
    record = RuleBasedFactorExtractor().extract(document)

    assert record is not None
    assert record.category is FactorVocabularyCategory.ONBOARDING
    assert record.factor_occurrence_time == date(2026, 6, 20)
    assert record.scope == EvidenceScope(
        product="Confluence", region="EMEA", tenant_scope="EMEA 51-200 Seat Tier Tenants"
    )


def test_extractor_returns_no_category_when_no_vocabulary_keyword_matches() -> None:
    document = _CORPUS["jira-apac-tenant-migration-notice"]
    record = RuleBasedFactorExtractor().extract(document)

    assert record is not None
    assert record.category is None


def test_validator_rejects_missing_factor_occurrence_time() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = ProvisionalFactorRecord(
        factor_id="x",
        category=FactorVocabularyCategory.INCIDENT,
        proposed_mechanism="mechanism",
        factor_occurrence_time=None,
        scope=EvidenceScope(
            product="Jira", region="APAC", tenant_scope="APAC 51-200 Seat Tier Tenants"
        ),
        document=document,
    )
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")

    eligibility = validate_ranking_eligibility(
        record, driver_window=window, access_filter=_permissive_filter(document)
    )

    assert eligibility.eligible is False
    assert "missing_factor_occurrence_time" in eligibility.blocked_reasons


def test_validator_rejects_missing_scope() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = ProvisionalFactorRecord(
        factor_id="x",
        category=FactorVocabularyCategory.INCIDENT,
        proposed_mechanism="mechanism",
        factor_occurrence_time=date(2026, 6, 12),
        scope=None,
        document=document,
    )
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")

    eligibility = validate_ranking_eligibility(
        record, driver_window=window, access_filter=_permissive_filter(document)
    )

    assert eligibility.eligible is False
    assert "missing_scope" in eligibility.blocked_reasons


def test_validator_rejects_occurrence_time_after_the_movement_window() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = RuleBasedFactorExtractor().extract(document)
    assert record is not None
    late_record = replace(record, factor_occurrence_time=date(2026, 7, 1))
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")

    eligibility = validate_ranking_eligibility(
        late_record, driver_window=window, access_filter=_permissive_filter(document)
    )

    assert eligibility.eligible is False
    assert "occurrence_time_after_movement_window" in eligibility.blocked_reasons


def test_validator_rejects_occurrence_time_more_than_14_days_before_the_window() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = RuleBasedFactorExtractor().extract(document)
    assert record is not None
    early_record = replace(record, factor_occurrence_time=date(2026, 5, 17))
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")

    eligibility = validate_ranking_eligibility(
        early_record, driver_window=window, access_filter=_permissive_filter(document)
    )

    assert eligibility.eligible is False
    assert "occurrence_time_exceeds_initial_lookback" in eligibility.blocked_reasons


def test_validator_accepts_occurrence_time_exactly_14_days_before_the_window() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = RuleBasedFactorExtractor().extract(document)
    assert record is not None
    boundary_record = replace(record, factor_occurrence_time=date(2026, 5, 18))
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")

    eligibility = validate_ranking_eligibility(
        boundary_record, driver_window=window, access_filter=_permissive_filter(document)
    )

    assert eligibility.eligible is True
    assert eligibility.blocked_reasons == ()


def test_validator_rejects_a_record_whose_source_is_no_longer_authorized() -> None:
    document = _CORPUS["jira-apac-paid-provisioning-incident"]
    record = RuleBasedFactorExtractor().extract(document)
    assert record is not None
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")
    americas_only_filter = EvidenceAccessFilter(
        products=("Jira",),
        regions=("Americas",),
        tenant_ids=tuple(document.tenant_ids),
        classifications=(document.classification,),
        identifier_entitlements=(document.identifier_entitlement,),
    )

    eligibility = validate_ranking_eligibility(
        record, driver_window=window, access_filter=americas_only_filter
    )

    assert eligibility.eligible is False
    assert "source_not_currently_authorized" in eligibility.blocked_reasons


def test_query_builder_is_deterministic_and_folds_in_time_rule_and_vocabulary() -> None:
    window = DriverMovementWindow.from_periods("2026-05", "2026-06")
    kwargs = dict(
        metric_label="Jira New PEU",
        product="Jira",
        region="APAC",
        seat_tier="51-200",
        driver_window=window,
        canonical_time_rule="Attribute to the first-ever Jira Paid Enablement.",
        movement_direction="decline",
    )

    first = build_evidence_investigation_query(**kwargs)
    second = build_evidence_investigation_query(**kwargs)

    assert first == second
    assert "Attribute to the first-ever Jira Paid Enablement." in first
    assert "provisioning" in first
    assert "campaign" in first

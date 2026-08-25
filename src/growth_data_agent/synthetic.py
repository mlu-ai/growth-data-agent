"""Deterministic synthetic data for the local Postgres analytical store."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .contracts import EvidenceSupportStatus
from .evidence import EvidenceDocument
from .graph import GraphNode, GraphPath
from .policy import tenant_ids_for_segment

_START_DATE = date(2025, 1, 1)
_END_DATE = date(2026, 6, 30)
_REGIONS = ("Americas", "APAC", "EMEA")
_SEAT_TIERS = ("1-10", "11-50", "51-200", "201+")
_JIRA_MAY_JUNE_SCENARIO = {
    ("APAC", "51-200"): (800, 380),
    ("Americas", "1-10"): (1000, 940),
    ("EMEA", "11-50"): (700, 680),
    ("APAC", "1-10"): (600, 580),
    ("EMEA", "51-200"): (500, 480),
    ("Americas", "51-200"): (400, 380),
}
_CONFLUENCE_MAY_JUNE_SCENARIO = {
    ("Americas", "11-50"): (1200, 1620),
    ("APAC", "1-10"): (600, 600),
    ("EMEA", "51-200"): (600, 600),
}


@dataclass(frozen=True)
class DatasetCounts:
    tenants: int
    persons: int
    product_users: int
    paid_enablements: int
    visits: int


def evidence_corpus() -> tuple[EvidenceDocument, ...]:
    """Return the deterministic incident corpus used by the evidence POC."""
    apac_enterprise_tenants = _tenant_ids_for_segment("APAC", "51-200")
    return (
        EvidenceDocument(
            document_id="jira-apac-paid-provisioning-incident",
            title="Jira APAC paid provisioning incident",
            text=(
                "Paid provisioning errors affected Jira APAC 51-200 Seat Tier Tenants "
                "from 2026-06-10 through 2026-06-12, overlapping the June New PEU decline."
            ),
            product="Jira",
            region="APAC",
            tenant_ids=apac_enterprise_tenants,
            tenant_scope="APAC 51-200 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 12),
            freshness=datetime(2026, 6, 13, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "The incident overlaps the APAC 51-200 Seat Tier Tenant scope and the June "
                "2026 decline period."
            ),
        ),
        EvidenceDocument(
            document_id="jira-apac-small-tenant-maintenance",
            title="Jira APAC small-Tenant provisioning maintenance",
            text=(
                "A Jira provisioning maintenance window affected APAC 1-10 Seat Tier "
                "Tenants in May 2026, outside the affected 51-200 Seat Tier scope."
            ),
            product="Jira",
            region="APAC",
            tenant_ids=_tenant_ids_for_segment("APAC", "1-10"),
            tenant_scope="APAC 1-10 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 5, 18),
            freshness=datetime(2026, 5, 19, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The notice concerns a different Seat Tier from the affected segment."
            ),
        ),
        EvidenceDocument(
            document_id="jira-apac-tenant-migration-notice",
            title="Jira APAC Tenant migration notice",
            text=(
                "A Jira APAC Tenant migration notice was published in June 2026, but it "
                "covered 201+ Seat Tier Tenants and not the affected 51-200 segment."
            ),
            product="Jira",
            region="APAC",
            tenant_ids=_tenant_ids_for_segment("APAC", "201+"),
            tenant_scope="APAC 201+ Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 9),
            freshness=datetime(2026, 6, 10, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The notice concerns a different Seat Tier from the affected segment."
            ),
        ),
        EvidenceDocument(
            document_id="jira-apac-paid-provisioning-incident-restricted",
            title="Restricted Jira APAC provisioning incident appendix",
            text=(
                "Restricted appendix with direct Tenant identifiers for the Jira APAC paid "
                "provisioning incident: tenant-0011."
            ),
            product="Jira",
            region="APAC",
            tenant_ids=["tenant-0011"],
            tenant_scope="APAC 51-200 Seat Tier Tenants",
            classification="restricted",
            identifier_entitlement="direct",
            relevant_date=date(2026, 6, 12),
            freshness=datetime(2026, 6, 13, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "This restricted appendix cannot be used without classification and direct "
                "identifier entitlement."
            ),
            sensitive_identifiers=["tenant-0011"],
        ),
        EvidenceDocument(
            document_id="confluence-americas-acquisition-campaign",
            title="Confluence Americas targeted acquisition campaign",
            text=(
                "A targeted acquisition campaign ran for Confluence Americas 11-50 Seat Tier "
                "Tenants from 2026-06-10 through 2026-06-15, overlapping the June New PEU increase."
            ),
            product="Confluence",
            region="Americas",
            tenant_ids=_tenant_ids_for_segment("Americas", "11-50"),
            tenant_scope="Americas 11-50 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 15),
            freshness=datetime(2026, 6, 16, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "The campaign overlaps the Americas 11-50 Seat Tier Tenant scope and the June "
                "2026 New PEU increase period."
            ),
        ),
        EvidenceDocument(
            document_id="confluence-americas-enterprise-campaign",
            title="Confluence Americas enterprise campaign summary",
            text=(
                "A Confluence campaign summary was published in June 2026, but it covered "
                "201+ Seat Tier Tenants and not the affected 11-50 segment."
            ),
            product="Confluence",
            region="Americas",
            tenant_ids=_tenant_ids_for_segment("Americas", "201+"),
            tenant_scope="Americas 201+ Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 9),
            freshness=datetime(2026, 6, 10, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The campaign concerns a different Seat Tier from the affected segment."
            ),
        ),
        EvidenceDocument(
            document_id="confluence-americas-provisioning-maintenance",
            title="Confluence Americas provisioning maintenance notice",
            text=(
                "A Confluence provisioning maintenance window affected Americas 1-10 Seat Tier "
                "Tenants in May 2026, outside the affected 11-50 Seat Tier scope."
            ),
            product="Confluence",
            region="Americas",
            tenant_ids=_tenant_ids_for_segment("Americas", "1-10"),
            tenant_scope="Americas 1-10 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 5, 18),
            freshness=datetime(2026, 5, 19, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The notice concerns a different Seat Tier from the affected segment."
            ),
        ),
        EvidenceDocument(
            document_id="confluence-americas-acquisition-campaign-restricted",
            title="Restricted Confluence Americas acquisition campaign appendix",
            text=(
                "Restricted appendix with direct Tenant identifiers for the Confluence Americas "
                "acquisition campaign: tenant-0002."
            ),
            product="Confluence",
            region="Americas",
            tenant_ids=["tenant-0002"],
            tenant_scope="Americas 11-50 Seat Tier Tenants",
            classification="restricted",
            identifier_entitlement="direct",
            relevant_date=date(2026, 6, 15),
            freshness=datetime(2026, 6, 16, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "This restricted appendix cannot be used without classification and direct "
                "identifier entitlement."
            ),
            sensitive_identifiers=["tenant-0002"],
        ),
    )


def graph_corpus() -> tuple[GraphPath, ...]:
    """Return deterministic public and direct-identifier evidence paths."""
    apac_enterprise_tenants = _tenant_ids_for_segment("APAC", "51-200")
    americas_campaign_tenants = _tenant_ids_for_segment("Americas", "11-50")
    return (
        GraphPath(
            path_id="jira-apac-new-peu-incident-chain",
            nodes=[
                GraphNode(
                    node_id="metric-jira-new-peu",
                    node_type="metric",
                    label="Jira New PEU",
                    product="Jira",
                    region="APAC",
                    tenant_ids=apac_enterprise_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
                GraphNode(
                    node_id="segment-apac-51-200",
                    node_type="segment",
                    label="APAC 51-200 Seat Tier Tenants",
                    product="Jira",
                    region="APAC",
                    tenant_ids=apac_enterprise_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
                GraphNode(
                    node_id="incident-jira-apac-paid-provisioning",
                    node_type="incident",
                    label="Jira APAC paid provisioning incident",
                    product="Jira",
                    region="APAC",
                    tenant_ids=apac_enterprise_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
            ],
        ),
        GraphPath(
            path_id="jira-apac-tenant-identifier-chain",
            nodes=[
                GraphNode(
                    node_id="incident-jira-apac-paid-provisioning-restricted",
                    node_type="incident",
                    label="Restricted Jira APAC provisioning appendix",
                    product="Jira",
                    region="APAC",
                    tenant_ids=["tenant-0011"],
                    classification="restricted",
                    identifier_entitlement="direct",
                ),
                GraphNode(
                    node_id="tenant-0011",
                    node_type="tenant",
                    label="tenant-0011",
                    product="Jira",
                    region="APAC",
                    tenant_ids=["tenant-0011"],
                    classification="restricted",
                    identifier_entitlement="direct",
                ),
            ],
        ),
        GraphPath(
            path_id="confluence-americas-acquisition-campaign-chain",
            nodes=[
                GraphNode(
                    node_id="metric-confluence-new-peu",
                    node_type="metric",
                    label="Confluence New PEU",
                    product="Confluence",
                    region="Americas",
                    tenant_ids=americas_campaign_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
                GraphNode(
                    node_id="segment-americas-11-50",
                    node_type="segment",
                    label="Americas 11-50 Seat Tier Tenants",
                    product="Confluence",
                    region="Americas",
                    tenant_ids=americas_campaign_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
                GraphNode(
                    node_id="campaign-confluence-americas-acquisition",
                    node_type="campaign",
                    label="Confluence Americas targeted acquisition campaign",
                    product="Confluence",
                    region="Americas",
                    tenant_ids=americas_campaign_tenants,
                    classification="internal",
                    identifier_entitlement="none",
                ),
            ],
        ),
        GraphPath(
            path_id="confluence-americas-campaign-identifier-chain",
            nodes=[
                GraphNode(
                    node_id="campaign-confluence-americas-acquisition-restricted",
                    node_type="campaign",
                    label="Restricted Confluence Americas campaign appendix",
                    product="Confluence",
                    region="Americas",
                    tenant_ids=["tenant-0002"],
                    classification="restricted",
                    identifier_entitlement="direct",
                ),
                GraphNode(
                    node_id="tenant-0002",
                    node_type="tenant",
                    label="tenant-0002",
                    product="Confluence",
                    region="Americas",
                    tenant_ids=["tenant-0002"],
                    classification="restricted",
                    identifier_entitlement="direct",
                ),
            ],
        ),
    )


def _tenant_ids_for_segment(region: str, seat_tier: str) -> list[str]:
    return list(tenant_ids_for_segment(region, seat_tier))


def generate(output_directory: Path) -> DatasetCounts:
    """Write a reproducible, glossary-aligned set of CSV source tables."""
    output_directory.mkdir(parents=True, exist_ok=True)
    tenants = _tenants()
    persons = [{"person_id": f"person-{number:05d}"} for number in range(1, 10_001)]
    product_users = _product_users(tenants)
    paid_enablements = _paid_enablements(product_users, tenants)
    visits = _visits(product_users)

    _write_csv(output_directory / "tenants.csv", tenants)
    _write_csv(output_directory / "persons.csv", persons)
    _write_csv(output_directory / "product_users.csv", product_users)
    _write_csv(output_directory / "paid_enablements.csv", paid_enablements)
    _write_csv(output_directory / "visits.csv", visits)
    return DatasetCounts(
        tenants=len(tenants),
        persons=len(persons),
        product_users=len(product_users),
        paid_enablements=len(paid_enablements),
        visits=len(visits),
    )


def _tenants() -> list[dict[str, str]]:
    return [
        {
            "tenant_id": f"tenant-{number:04d}",
            "billing_region": _REGIONS[(number - 1) % len(_REGIONS)],
            "paid_subscription_started_at": (
                _START_DATE - timedelta(days=30 * ((number - 1) % 18))
            ).isoformat(),
            "seat_tier": _SEAT_TIERS[(number - 1) % len(_SEAT_TIERS)],
        }
        for number in range(1, 1_001)
    ]


def _product_users(tenants: list[dict[str, str]]) -> list[dict[str, str]]:
    product_users: list[dict[str, str]] = []
    jira_scenario_tenant_ids = iter(
        _scenario_tenant_ids(tenants, _JIRA_MAY_JUNE_SCENARIO)
    )
    confluence_scenario_tenant_ids = iter(
        _scenario_tenant_ids(tenants, _CONFLUENCE_MAY_JUNE_SCENARIO)
    )
    sequence = 1
    # Each of these Persons has separate Product User relationships in both products.
    for person_number in range(1, 6_001):
        jira_tenant_id = next(jira_scenario_tenant_ids)
        confluence_tenant_id = _next_or_fallback(
            confluence_scenario_tenant_ids,
            jira_tenant_id,
        )
        for product, tenant_id in (
            ("Jira", jira_tenant_id),
            ("Confluence", confluence_tenant_id),
        ):
            product_users.append(
                {
                    "product_user_id": f"product-user-{sequence:05d}",
                    "person_id": f"person-{person_number:05d}",
                    "tenant_id": tenant_id,
                    "product": product,
                }
            )
            sequence += 1
    for person_number in range(6_001, 10_001):
        product = "Jira" if person_number % 2 else "Confluence"
        tenant_id = (
            next(jira_scenario_tenant_ids)
            if product == "Jira" and person_number < 8_881
            else f"tenant-{((person_number * 29 - 1) % 1_000) + 1:04d}"
        )
        product_users.append(
            {
                "product_user_id": f"product-user-{sequence:05d}",
                "person_id": f"person-{person_number:05d}",
                "tenant_id": tenant_id,
                "product": product,
            }
        )
        sequence += 1
    return product_users


def _scenario_tenant_ids(
    tenants: list[dict[str, str]], scenario: dict[tuple[str, str], tuple[int, int]]
) -> list[str]:
    tenant_ids_by_segment: dict[tuple[str, str], list[str]] = {}
    for tenant in tenants:
        segment = (tenant["billing_region"], tenant["seat_tier"])
        tenant_ids_by_segment.setdefault(segment, []).append(tenant["tenant_id"])

    assigned: list[str] = []
    for segment, (may_count, june_count) in scenario.items():
        tenant_ids = tenant_ids_by_segment[segment]
        assigned.extend(
            tenant_ids[index % len(tenant_ids)] for index in range(may_count + june_count)
        )
    return assigned


def _paid_enablements(
    product_users: list[dict[str, str]], tenants: list[dict[str, str]]
) -> list[dict[str, str]]:
    event_rows: list[dict[str, str]] = []
    tenant_segments = {
        tenant["tenant_id"]: (tenant["billing_region"], tenant["seat_tier"])
        for tenant in tenants
    }
    scenario_seen: dict[tuple[str, tuple[str, str]], int] = {}
    sequence = 1
    for index, product_user in enumerate(product_users, start=1):
        first_enabled = _first_enablement_date(
            index,
            product_user,
            tenant_segments,
            scenario_seen,
        )
        event_rows.append(_enablement_event(sequence, product_user, first_enabled))
        sequence += 1
        # A later immutable Paid Enablement event proves that restoration does not requalify.
        restoration = first_enabled + timedelta(days=70)
        if index % 7 == 0 and restoration <= _END_DATE:
            event_rows.append(_enablement_event(sequence, product_user, restoration))
            sequence += 1
    return event_rows


def _first_enablement_date(
    index: int,
    product_user: dict[str, str],
    tenant_segments: dict[str, tuple[str, str]],
    scenario_seen: dict[tuple[str, tuple[str, str]], int],
) -> date:
    segment = tenant_segments[product_user["tenant_id"]]
    scenario = {
        "Jira": _JIRA_MAY_JUNE_SCENARIO,
        "Confluence": _CONFLUENCE_MAY_JUNE_SCENARIO,
    }.get(product_user["product"], {})
    if segment in scenario:
        scenario_key = (product_user["product"], segment)
        scenario_position = scenario_seen.get(scenario_key, 0)
        scenario_seen[scenario_key] = scenario_position + 1
        may_count, june_count = scenario[segment]
        if scenario_position < may_count:
            return date(2026, 5, scenario_position % 31 + 1)
        if scenario_position < may_count + june_count:
            return date(2026, 6, (scenario_position - may_count) % 30 + 1)

    # Keep non-scenario New PEU outside the comparison months so the first
    # Driver Decomposition is deterministic while retaining eighteen months of events.
    days_before_scenario = (date(2026, 4, 30) - _START_DATE).days + 1
    return _START_DATE + timedelta(days=(index * 37) % days_before_scenario)


def _enablement_event(
    sequence: int, product_user: dict[str, str], enabled_on: date
) -> dict[str, str]:
    return {
        "paid_enablement_id": f"paid-enablement-{sequence:05d}",
        "product_user_id": product_user["product_user_id"],
        "tenant_id": product_user["tenant_id"],
        "product": product_user["product"],
        "paid_enabled_at": datetime.combine(enabled_on, datetime.min.time(), UTC).isoformat(),
    }


def _visits(product_users: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "visit_id": f"visit-{index:05d}",
            "product_user_id": product_user["product_user_id"],
            "product": product_user["product"],
            "visited_at": datetime.combine(
                _START_DATE + timedelta(days=(index * 41) % ((_END_DATE - _START_DATE).days + 1)),
                datetime.min.time(),
                UTC,
            ).isoformat(),
        }
        for index, product_user in enumerate(product_users, start=1)
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _next_or_fallback(iterator, fallback: str) -> str:
    try:
        return next(iterator)
    except StopIteration:
        return fallback

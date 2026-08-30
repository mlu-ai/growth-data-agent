"""Deterministic synthetic data for the local Postgres analytical store."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .contracts import EvidenceSupportStatus
from .datahub import DataHubEntityMetadata
from .evidence import EvidenceDocument, EvidenceLifecycleState, EvidencePrincipalGrant
from .evidence_sync import (
    ConfluenceEvidenceChunk,
    ConfluenceEvidenceRevision,
    EvidenceRevisionValidationError,
    SourceAccessMetadata,
)
from .graph import DerivedEvidenceGraphBuilder, GraphPath
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
_CONFLUENCE_EMEA_NEW_MAU_SCENARIO = {
    ("EMEA", "51-200"): (600, 300),
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
    apac_51_200_tenants = _tenant_ids_for_segment("APAC", "51-200")
    emea_51_200_tenants = _tenant_ids_for_segment("EMEA", "51-200")
    documents = (
        EvidenceDocument(
            document_id="jira-apac-paid-provisioning-incident",
            metric_name="jira_new_peu",
            title="Jira APAC paid provisioning incident",
            text=(
                "Paid provisioning errors affected Jira APAC 51-200 Seat Tier Tenants "
                "from 2026-06-10 through 2026-06-12, overlapping the June New PEU decline."
            ),
            product="Jira",
            region="APAC",
            tenant_ids=apac_51_200_tenants,
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
            accountable_team="Jira Platform Provisioning Team",
        ),
        EvidenceDocument(
            document_id="jira-apac-small-tenant-maintenance",
            metric_name="jira_new_peu",
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
            accountable_team="Jira Platform Provisioning Team",
        ),
        EvidenceDocument(
            document_id="jira-apac-tenant-migration-notice",
            metric_name="jira_new_peu",
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
            accountable_team="Jira Platform Migration Team",
        ),
        EvidenceDocument(
            document_id="jira-apac-paid-provisioning-incident-restricted",
            metric_name="jira_new_peu",
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
            accountable_team="Jira Platform Provisioning Team",
            access_groups=[],
            direct_principal_grants=[
                EvidencePrincipalGrant(
                    principal_id="customer_success_manager",
                    expires_at=datetime(2099, 12, 31, tzinfo=UTC),
                )
            ],
        ),
        EvidenceDocument(
            document_id="confluence-americas-acquisition-campaign",
            metric_name="confluence_new_peu",
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
            accountable_team="Confluence Growth Acquisition Team",
            source_document_id="confluence-americas-acquisition-campaign",
            source_url="https://evidence.local/synthetic/confluence-americas-acquisition-campaign",
            source_revision="synthetic-v1",
            source_page_id="confluence-americas-acquisition-campaign",
            chunk_id="confluence-americas-acquisition-campaign:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-americas-enterprise-campaign",
            metric_name="confluence_new_peu",
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
            accountable_team="Confluence Growth Acquisition Team",
            source_document_id="confluence-americas-enterprise-campaign",
            source_url="https://evidence.local/synthetic/confluence-americas-enterprise-campaign",
            source_revision="synthetic-v1",
            source_page_id="confluence-americas-enterprise-campaign",
            chunk_id="confluence-americas-enterprise-campaign:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-americas-provisioning-maintenance",
            metric_name="confluence_new_peu",
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
            accountable_team="Confluence Platform Provisioning Team",
            source_document_id="confluence-americas-provisioning-maintenance",
            source_url="https://evidence.local/synthetic/confluence-americas-provisioning-maintenance",
            source_revision="synthetic-v1",
            source_page_id="confluence-americas-provisioning-maintenance",
            chunk_id="confluence-americas-provisioning-maintenance:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-americas-acquisition-campaign-restricted",
            metric_name="confluence_new_peu",
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
            accountable_team="Confluence Growth Acquisition Team",
            access_groups=[],
            direct_principal_grants=[
                EvidencePrincipalGrant(
                    principal_id="customer_success_manager",
                    expires_at=datetime(2099, 12, 31, tzinfo=UTC),
                )
            ],
            source_document_id="confluence-americas-acquisition-campaign-restricted",
            source_url="https://evidence.local/synthetic/confluence-americas-acquisition-campaign-restricted",
            source_revision="synthetic-v1",
            source_page_id="confluence-americas-acquisition-campaign-restricted",
            chunk_id="confluence-americas-acquisition-campaign-restricted:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-emea-onboarding-email-regression",
            metric_name="confluence_new_mau",
            title="Confluence EMEA onboarding-email regression",
            text=(
                "An onboarding-email regression affected Confluence EMEA 51-200 Seat Tier "
                "Tenants from 2026-06-08 through 2026-06-20, overlapping the June New MAU "
                "decline."
            ),
            product="Confluence",
            region="EMEA",
            tenant_ids=emea_51_200_tenants,
            tenant_scope="EMEA 51-200 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 20),
            freshness=datetime(2026, 6, 21, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "The onboarding-email regression overlaps the EMEA 51-200 Seat Tier Tenant "
                "scope and the June 2026 New MAU decline period."
            ),
            accountable_team="Confluence Growth Activation Team",
            source_document_id="confluence-emea-onboarding-email-regression",
            source_url="https://evidence.local/synthetic/confluence-emea-onboarding-email-regression",
            source_revision="synthetic-v1",
            source_page_id="confluence-emea-onboarding-email-regression",
            chunk_id="confluence-emea-onboarding-email-regression:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-emea-small-tenant-onboarding-email",
            metric_name="confluence_new_mau",
            title="Confluence EMEA small-Tenant onboarding-email review",
            text=(
                "An onboarding-email review covered Confluence EMEA 1-10 Seat Tier Tenants "
                "in May 2026, outside the affected 51-200 segment."
            ),
            product="Confluence",
            region="EMEA",
            tenant_ids=_tenant_ids_for_segment("EMEA", "1-10"),
            tenant_scope="EMEA 1-10 Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 5, 18),
            freshness=datetime(2026, 5, 19, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The review concerns a different Seat Tier from the affected segment."
            ),
            accountable_team="Confluence Growth Activation Team",
            source_document_id="confluence-emea-small-tenant-onboarding-email",
            source_url="https://evidence.local/synthetic/confluence-emea-small-tenant-onboarding-email",
            source_revision="synthetic-v1",
            source_page_id="confluence-emea-small-tenant-onboarding-email",
            chunk_id="confluence-emea-small-tenant-onboarding-email:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-emea-201-plus-onboarding-email",
            metric_name="confluence_new_mau",
            title="Confluence EMEA 201+ Seat Tier onboarding-email summary",
            text=(
                "An onboarding-email summary was published in June 2026, but it covered "
                "201+ Seat Tier Tenants rather than the affected 51-200 segment."
            ),
            product="Confluence",
            region="EMEA",
            tenant_ids=_tenant_ids_for_segment("EMEA", "201+"),
            tenant_scope="EMEA 201+ Seat Tier Tenants",
            classification="internal",
            identifier_entitlement="none",
            relevant_date=date(2026, 6, 9),
            freshness=datetime(2026, 6, 10, tzinfo=UTC),
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The summary concerns a different Seat Tier from the affected segment."
            ),
            accountable_team="Confluence Growth Activation Team",
            source_document_id="confluence-emea-201-plus-onboarding-email",
            source_url="https://evidence.local/synthetic/confluence-emea-201-plus-onboarding-email",
            source_revision="synthetic-v1",
            source_page_id="confluence-emea-201-plus-onboarding-email",
            chunk_id="confluence-emea-201-plus-onboarding-email:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            access_groups=["evidence-general"],
            direct_principal_grants=[],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        EvidenceDocument(
            document_id="confluence-emea-onboarding-email-regression-restricted",
            metric_name="confluence_new_mau",
            title="Restricted Confluence EMEA onboarding-email regression appendix",
            text=(
                "Restricted appendix with direct Tenant identifiers for the Confluence EMEA "
                "onboarding-email regression: tenant-0003."
            ),
            product="Confluence",
            region="EMEA",
            tenant_ids=["tenant-0003"],
            tenant_scope="EMEA 51-200 Seat Tier Tenants",
            classification="restricted",
            identifier_entitlement="direct",
            relevant_date=date(2026, 6, 20),
            freshness=datetime(2026, 6, 21, tzinfo=UTC),
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "This restricted appendix cannot be used without classification and direct "
                "identifier entitlement."
            ),
            sensitive_identifiers=["tenant-0003"],
            accountable_team="Confluence Growth Activation Team",
            access_groups=[],
            direct_principal_grants=[
                EvidencePrincipalGrant(
                    principal_id="customer_success_manager",
                    expires_at=datetime(2099, 12, 31, tzinfo=UTC),
                )
            ],
            source_document_id="confluence-emea-onboarding-email-regression-restricted",
            source_url="https://evidence.local/synthetic/confluence-emea-onboarding-email-regression-restricted",
            source_revision="synthetic-v1",
            source_page_id="confluence-emea-onboarding-email-regression-restricted",
            chunk_id="confluence-emea-onboarding-email-regression-restricted:chunk:0",
            chunk_index=0,
            lifecycle_state=EvidenceLifecycleState.ACTIVE,
            embedding_model="deterministic-hash",
            embedding_version="1",
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
    )
    return documents


class SyntheticConfluenceEvidenceSource:
    """Adapt the deterministic fixture corpus to the normalized source contract."""

    def __init__(self, documents: Iterable[EvidenceDocument] | None = None) -> None:
        self._documents = tuple(documents if documents is not None else evidence_corpus())

    def iter_revisions(self) -> Iterable[ConfluenceEvidenceRevision]:
        for document in self._documents:
            if document.product != "Confluence":
                continue
            if not document.source_document_id or not document.source_url:
                raise EvidenceRevisionValidationError(
                    f"Synthetic source page {document.document_id} is missing provenance."
                )
            required_fields = {
                "source_document_id",
                "source_url",
                "source_revision",
                "source_page_id",
                "chunk_id",
                "chunk_index",
                "lifecycle_state",
                "classification",
                "identifier_entitlement",
                "access_groups",
                "direct_principal_grants",
                "policy_expires_at",
                "embedding_model",
                "embedding_version",
            }
            missing_fields = {
                field
                for field in required_fields
                if field not in document.model_fields_set
                or getattr(document, field) is None
                or (
                    isinstance(getattr(document, field), str)
                    and not getattr(document, field).strip()
                )
            }
            if missing_fields:
                raise EvidenceRevisionValidationError(
                    f"Synthetic source page {document.document_id} is missing required metadata: "
                    f"{', '.join(sorted(missing_fields))}."
                )
            source_page_id = document.source_document_id
            source_url = document.source_url
            chunks = (
                [
                    ConfluenceEvidenceChunk(
                        chunk_id=document.chunk_id,
                        chunk_index=document.chunk_index,
                        text=document.text,
                    )
                ]
                if document.lifecycle_state is EvidenceLifecycleState.ACTIVE
                else []
            )
            yield ConfluenceEvidenceRevision(
                source_page_id=source_page_id,
                source_url=source_url,
                source_revision=document.source_revision,
                lifecycle_state=document.lifecycle_state,
                metric_name=document.metric_name,
                title=document.title,
                product=document.product,
                region=document.region,
                tenant_ids=document.tenant_ids,
                tenant_scope=document.tenant_scope,
                relevant_date=document.relevant_date,
                freshness=document.freshness,
                support_status=document.support_status,
                support_explanation=document.support_explanation,
                chunks=chunks,
                source_access=SourceAccessMetadata(
                    classification=document.classification,
                    identifier_entitlement=document.identifier_entitlement,
                    access_groups=document.access_groups,
                    direct_principal_grants=document.direct_principal_grants,
                    policy_expires_at=document.policy_expires_at,
                ),
                embedding_model=document.embedding_model,
                embedding_version=document.embedding_version,
            )


def graph_corpus() -> tuple[GraphPath, ...]:
    """Derive governed evidence chains from catalog and document metadata."""
    return DerivedEvidenceGraphBuilder().build(_synthetic_catalog_entities(), evidence_corpus())


def _synthetic_catalog_entities() -> tuple[DataHubEntityMetadata, ...]:
    published_at = datetime(2026, 8, 25, tzinfo=UTC)
    entities: list[DataHubEntityMetadata] = []
    for metric_name, product in (
        ("jira_new_peu", "Jira"),
        ("jira_new_mau", "Jira"),
        ("confluence_new_peu", "Confluence"),
        ("confluence_new_mau", "Confluence"),
    ):
        entities.append(
            DataHubEntityMetadata(
                entity_name=f"fct_{metric_name}",
                entity_type="model",
                urn=(
                    "urn:li:dataset:(urn:li:dataPlatform:postgres,"
                    f"growth_data.analytics.fct_{metric_name},PROD)"
                ),
                product=product,
                owners=["growth-data"],
                classification="internal",
                discovery_tags=["dbt-model", "canonical-metric", f"product:{product.casefold()}"],
                description=f"Synthetic validated {product} {metric_name} dbt model.",
                semantic_version="1.0.0",
                source_artifact_sha256="synthetic-v1",
                published_at=published_at,
            )
        )
    return tuple(entities)


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
    scenario_seen: dict[tuple[str, tuple[str, str]], int] = {}
    visits = []
    for index, product_user in enumerate(product_users, start=1):
        segment = _tenant_segment(product_user)
        scenario_key = (product_user["product"], segment)
        position = scenario_seen.get(scenario_key, 0)
        scenario_seen[scenario_key] = position + 1
        visited_on = _new_mau_visit_date(product_user, position)
        if visited_on is None:
            visited_on = _START_DATE + timedelta(
                days=(index * 41) % ((_END_DATE - _START_DATE).days + 1)
            )
        visits.append(
            {
                "visit_id": f"visit-{index:05d}",
                "product_user_id": product_user["product_user_id"],
                "product": product_user["product"],
                "visited_at": datetime.combine(visited_on, datetime.min.time(), UTC).isoformat(),
            }
        )
    return visits


def _new_mau_visit_date(
    product_user: dict[str, str], position: int
) -> date | None:
    if product_user["product"] != "Confluence":
        return None
    segment = _tenant_segment(product_user)
    new_mau_counts = _CONFLUENCE_EMEA_NEW_MAU_SCENARIO.get(segment)
    peu_counts = _CONFLUENCE_MAY_JUNE_SCENARIO.get(segment)
    if new_mau_counts is None or peu_counts is None:
        return None
    may_count, june_new_mau_count = new_mau_counts
    _, june_peu_count = peu_counts
    if position < may_count:
        return date(2026, 5, position % 31 + 1)
    if position < may_count + june_new_mau_count:
        return date(2026, 6, (position - may_count) % 30 + 1)
    if position < may_count + june_peu_count:
        return date(2026, 4, (position - may_count - june_new_mau_count) % 30 + 1)
    return None


def _tenant_segment(product_user: dict[str, str]) -> tuple[str, str]:
    tenant_number = int(product_user["tenant_id"].rsplit("-", 1)[1])
    region = _REGIONS[(tenant_number - 1) % len(_REGIONS)]
    seat_tier = _SEAT_TIERS[(tenant_number - 1) % len(_SEAT_TIERS)]
    return region, seat_tier


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

"""Resolve an Access Profile before the semantic gateway is invoked."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EffectiveAccessScope, ProvisionalMetricInput
from .evidence import EvidenceAccessFilter
from .graph import GraphAccessFilter

_ALL_TENANT_IDS = tuple(f"tenant-{number:04d}" for number in range(1, 1_001))
_ALL_REGIONS = ("Americas", "APAC", "EMEA")
_ALL_SEAT_TIERS = ("1-10", "11-50", "51-200", "201+")
_QUERY_COLUMN_NAMES = {
    "product_user__product": "product",
    "product_user__region": "region",
    "product_user__seat_tier": "seat_tier",
    "confluence_product_user__product": "product",
    "confluence_product_user__region": "region",
    "confluence_product_user__seat_tier": "seat_tier",
    "jira_new_mau_product_user__product": "product",
    "jira_new_mau_product_user__region": "region",
    "jira_new_mau_product_user__seat_tier": "seat_tier",
    "confluence_new_mau_product_user__product": "product",
    "confluence_new_mau_product_user__region": "region",
    "confluence_new_mau_product_user__seat_tier": "seat_tier",
    "metric_time__month": "metric_month",
}


def tenant_ids_for_region(region: str) -> tuple[str, ...]:
    """Resolve synthetic Tenant entitlements by their recorded billing Region."""
    try:
        region_index = _ALL_REGIONS.index(region)
    except ValueError as error:
        raise AccessDeniedError(f"Unknown Region entitlement: {region}.") from error
    return tuple(
        tenant_id
        for number, tenant_id in enumerate(_ALL_TENANT_IDS, start=1)
        if (number - 1) % len(_ALL_REGIONS) == region_index
    )


def tenant_ids_for_segment(region: str, seat_tier: str) -> tuple[str, ...]:
    """Resolve synthetic Tenant entitlements for a Region and Seat Tier."""
    if seat_tier not in _ALL_SEAT_TIERS:
        raise AccessDeniedError(f"Unknown Seat Tier entitlement: {seat_tier}.")
    return tuple(
        tenant_id
        for tenant_id in tenant_ids_for_region(region)
        if (int(tenant_id.rsplit("-", 1)[1]) - 1) % len(_ALL_SEAT_TIERS)
        == _ALL_SEAT_TIERS.index(seat_tier)
    )


@dataclass(frozen=True)
class AccessProfile:
    products: tuple[str, ...]
    regions: tuple[str, ...]
    tenant_scope: str
    permitted_columns: tuple[str, ...]
    permitted_tenant_ids: tuple[str, ...]
    permitted_classifications: tuple[str, ...] = ("internal",)
    permitted_identifiers: tuple[str, ...] = ()
    permitted_query_columns: tuple[str, ...] = (
        "product",
        "region",
        "seat_tier",
        "metric_month",
    )

    def metricflow_where_constraints(
        self, metric_product: str, *, entity_name: str = "product_user"
    ) -> tuple[str, ...]:
        """Return fixed, profile-derived MetricFlow filters for a canonical metric.

        The service never accepts filter text from an Agent User. Product is a
        metric request property, while Region is an entitlement applied before
        MetricFlow plans SQL. Tenant scope is included in the effective scope;
        for the APAC profile, its permitted Tenant set is represented by the
        APAC Region constraint.
        """
        self.authorize_product(metric_product)

        constraints = [f"{entity_name}__product = '{metric_product}'"]
        if len(self.regions) != len(_ALL_REGIONS):
            regions = ", ".join(repr(region) for region in self.regions)
            constraints.append(f"{entity_name}__region IN ({regions})")
        permitted_region_tenants = {
            tenant_id
            for region in self.regions
            for tenant_id in tenant_ids_for_region(region)
        }
        if set(self.permitted_tenant_ids) != permitted_region_tenants:
            tenants = ", ".join(repr(tenant_id) for tenant_id in self.permitted_tenant_ids)
            constraints.append(f"{entity_name}__tenant_id IN ({tenants})")
        return tuple(constraints)

    def authorize_product(self, product: str) -> None:
        """Authorize a product before any product-owned source is read."""
        if product not in self.products:
            raise AccessDeniedError(f"Access Profile is not entitled to {product} data.")

    def authorize_region(self, region: str) -> None:
        """Authorize a Region before a scoped metric or evidence source is read."""
        if region not in self.regions:
            raise AccessDeniedError(f"Access Profile is not entitled to {region} data.")

    def as_effective_scope(self) -> EffectiveAccessScope:
        return EffectiveAccessScope(
            products=list(self.products),
            regions=list(self.regions),
            tenant_scope=self.tenant_scope,
            permitted_columns=list(self.permitted_columns),
        )

    def evidence_filter(
        self,
        product: str,
        region: str,
        *,
        seat_tier: str | None = None,
        metric_name: str | None = None,
    ) -> EvidenceAccessFilter:
        """Derive every document filter before the vector store is queried."""
        if product not in self.products:
            raise AccessDeniedError(f"Access Profile is not entitled to {product} evidence.")
        if region not in self.regions:
            raise AccessDeniedError(f"Access Profile is not entitled to {region} evidence.")
        region_tenants = (
            tenant_ids_for_segment(region, seat_tier)
            if seat_tier is not None
            else tenant_ids_for_region(region)
        )
        permitted_tenants = tuple(
            tenant_id for tenant_id in region_tenants if tenant_id in self.permitted_tenant_ids
        )
        if not permitted_tenants:
            raise AccessDeniedError(f"Access Profile has no permitted {region} Tenants.")
        identifier_entitlements = ("none", "direct") if self.permitted_identifiers else ("none",)
        return EvidenceAccessFilter(
            products=(product,),
            regions=(region,),
            tenant_ids=permitted_tenants,
            classifications=self.permitted_classifications,
            identifier_entitlements=identifier_entitlements,
            excluded_tenant_ids=tuple(
                tenant_id for tenant_id in _ALL_TENANT_IDS if tenant_id not in permitted_tenants
            ),
            seat_tiers=(seat_tier,) if seat_tier is not None else (),
            metric_names=(metric_name,) if metric_name is not None else (),
        )

    def graph_filter(
        self, product: str, region: str, *, seat_tier: str | None = None
    ) -> GraphAccessFilter:
        """Derive graph traversal constraints before a path is requested."""
        if product not in self.products:
            raise AccessDeniedError(f"Access Profile is not entitled to {product} graph paths.")
        if region not in self.regions:
            raise AccessDeniedError(f"Access Profile is not entitled to {region} graph paths.")
        region_tenants = (
            tenant_ids_for_segment(region, seat_tier)
            if seat_tier is not None
            else tenant_ids_for_region(region)
        )
        permitted_tenants = tuple(
            tenant_id for tenant_id in region_tenants if tenant_id in self.permitted_tenant_ids
        )
        if not permitted_tenants:
            raise AccessDeniedError(f"Access Profile has no permitted {region} Tenants.")
        identifier_entitlements = ("none", "direct") if self.permitted_identifiers else ("none",)
        return GraphAccessFilter(
            products=(product,),
            regions=(region,),
            tenant_ids=permitted_tenants,
            classifications=self.permitted_classifications,
            identifier_entitlements=identifier_entitlements,
            seat_tiers=(seat_tier,) if seat_tier is not None else (),
        )

    def permits_provisional_inputs(self, inputs: list[ProvisionalMetricInput]) -> bool:
        """Allow a provisional calculation only when every declared input is entitled."""
        return all(item.name in self.permitted_columns for item in inputs)

    def authorize_query_columns(self, group_by_names: tuple[str, ...]) -> None:
        """Reject MetricFlow plans that request columns outside this profile."""
        requested = tuple(_QUERY_COLUMN_NAMES.get(name, name) for name in group_by_names)
        unauthorized = tuple(
            column for column in requested if column not in self.permitted_query_columns
        )
        if unauthorized:
            columns = ", ".join(unauthorized)
            raise AccessDeniedError(f"Access Profile is not entitled to query columns: {columns}.")


_CANONICAL_DEFINITION_COLUMNS = (
    "metric_name",
    "definition",
    "formula",
    "grain",
    "time_rule",
    "semantic_version",
    "source_freshness",
    "paid_enablement_id",
)

_PROFILES = {
    "data_analyst": AccessProfile(
        products=("Jira", "Confluence"),
        regions=_ALL_REGIONS,
        tenant_scope="all permitted Tenants",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
        permitted_tenant_ids=_ALL_TENANT_IDS,
    ),
    "apac_regional_manager": AccessProfile(
        products=("Jira", "Confluence"),
        regions=("APAC",),
        tenant_scope="APAC Tenants only",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
        permitted_tenant_ids=tenant_ids_for_region("APAC"),
    ),
    "jira_product_manager": AccessProfile(
        products=("Jira",),
        regions=_ALL_REGIONS,
        tenant_scope="All permitted Jira Tenants",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
        permitted_tenant_ids=_ALL_TENANT_IDS,
    ),
    "confluence_product_manager": AccessProfile(
        products=("Confluence",),
        regions=_ALL_REGIONS,
        tenant_scope="All permitted Confluence Tenants",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
        permitted_tenant_ids=_ALL_TENANT_IDS,
    ),
    "customer_success_manager": AccessProfile(
        products=("Jira", "Confluence"),
        regions=("APAC",),
        tenant_scope="APAC 51-200 Seat Tier Tenant portfolio",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS + ("tenant_id",),
        permitted_tenant_ids=tuple(
            tenant_id
            for tenant_id in tenant_ids_for_region("APAC")
            if (int(tenant_id.rsplit("-", 1)[1]) - 1) % 4 == 2
        ),
        permitted_classifications=("internal", "restricted"),
        permitted_identifiers=("tenant_id",),
        permitted_query_columns=("product", "region"),
    ),
}


class UnknownAgentUserError(ValueError):
    """Raised when no Access Profile exists for an Agent User."""

    def __init__(self, message: str = "Unknown Agent User.", *, trace_id: str | None = None):
        super().__init__(message)
        self.trace_id = trace_id


class AccessDeniedError(ValueError):
    """Raised when an Access Profile is outside a metric's product scope."""

    def __init__(self, message: str, *, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def resolve_access_profile(agent_user_id: str) -> AccessProfile:
    try:
        return _PROFILES[agent_user_id]
    except KeyError as error:
        raise UnknownAgentUserError() from error

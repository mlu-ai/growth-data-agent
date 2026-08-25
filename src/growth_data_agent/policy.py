"""Resolve an Access Profile before the semantic gateway is invoked."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EffectiveAccessScope


@dataclass(frozen=True)
class AccessProfile:
    products: tuple[str, ...]
    regions: tuple[str, ...]
    tenant_scope: str
    permitted_columns: tuple[str, ...]

    def metricflow_where_constraints(self, metric_product: str) -> tuple[str, ...]:
        """Return fixed, profile-derived MetricFlow filters for a canonical metric.

        The service never accepts filter text from an Agent User. Product is a
        metric request property, while Region is an entitlement applied before
        MetricFlow plans SQL. Tenant scope is included in the effective scope;
        for the APAC profile, its permitted Tenant set is represented by the
        APAC Region constraint.
        """
        if metric_product not in self.products:
            raise AccessDeniedError(f"Access Profile is not entitled to {metric_product} data.")

        constraints = [f"product_user__product = '{metric_product}'"]
        if len(self.regions) != len(_ALL_REGIONS):
            regions = ", ".join(repr(region) for region in self.regions)
            constraints.append(f"product_user__region IN ({regions})")
        return tuple(constraints)

    def as_effective_scope(self) -> EffectiveAccessScope:
        return EffectiveAccessScope(
            products=list(self.products),
            regions=list(self.regions),
            tenant_scope=self.tenant_scope,
            permitted_columns=list(self.permitted_columns),
        )


_CANONICAL_DEFINITION_COLUMNS = (
    "metric_name",
    "definition",
    "formula",
    "grain",
    "time_rule",
    "semantic_version",
    "source_freshness",
)

_ALL_REGIONS = ("Americas", "APAC", "EMEA")

_PROFILES = {
    "data_analyst": AccessProfile(
        products=("Jira", "Confluence"),
        regions=_ALL_REGIONS,
        tenant_scope="all permitted Tenants",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
    ),
    "apac_regional_manager": AccessProfile(
        products=("Jira", "Confluence"),
        regions=("APAC",),
        tenant_scope="APAC Tenants only",
        permitted_columns=_CANONICAL_DEFINITION_COLUMNS,
    ),
}


class UnknownAgentUserError(ValueError):
    """Raised when no Access Profile exists for an Agent User."""


class AccessDeniedError(ValueError):
    """Raised when an Access Profile is outside a metric's product scope."""


def resolve_access_profile(agent_user_id: str) -> AccessProfile:
    try:
        return _PROFILES[agent_user_id]
    except KeyError as error:
        raise UnknownAgentUserError(f"Unknown Agent User: {agent_user_id}") from error

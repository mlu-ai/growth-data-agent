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

_PROFILES = {
    "data_analyst": AccessProfile(
        products=("Jira", "Confluence"),
        regions=("Americas", "APAC", "EMEA"),
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


def resolve_access_profile(agent_user_id: str) -> AccessProfile:
    try:
        return _PROFILES[agent_user_id]
    except KeyError as error:
        raise UnknownAgentUserError(f"Unknown Agent User: {agent_user_id}") from error

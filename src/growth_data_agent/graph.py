"""Entitlement-filtered traversal over the derived evidence graph."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class GraphNode(BaseModel):
    """A derived graph node with the metadata needed for pre-traversal policy."""

    node_id: str
    node_type: str
    label: str
    product: str
    region: str
    tenant_ids: list[str]
    classification: str
    identifier_entitlement: str


class GraphPath(BaseModel):
    """A bounded evidence path returned from an authorized graph traversal."""

    path_id: str
    nodes: list[GraphNode]


@dataclass(frozen=True)
class GraphAccessFilter:
    """All graph constraints derived from an Access Profile and known driver."""

    products: tuple[str, ...]
    regions: tuple[str, ...]
    tenant_ids: tuple[str, ...]
    classifications: tuple[str, ...]
    identifier_entitlements: tuple[str, ...]
    seat_tiers: tuple[str, ...] = ()

    def allows(self, path: GraphPath) -> bool:
        """Apply the policy to every node before a path reaches response generation."""
        return all(
            node.product in self.products
            and node.region in self.regions
            and set(node.tenant_ids).issubset(self.tenant_ids)
            and node.classification in self.classifications
            and node.identifier_entitlement in self.identifier_entitlements
            for node in path.nodes
        )


class EvidenceGraphStore(Protocol):
    def traverse(
        self,
        query: str,
        access_filter: GraphAccessFilter,
        *,
        limit: int,
    ) -> list[GraphPath]: ...


class InMemoryEvidenceGraphStore:
    """Deterministic graph boundary used by the local POC."""

    def __init__(self, paths: Iterable[GraphPath]):
        self._paths = tuple(paths)
        self.last_filter: GraphAccessFilter | None = None

    def traverse(
        self,
        query: str,
        access_filter: GraphAccessFilter,
        *,
        limit: int,
    ) -> list[GraphPath]:
        del query
        self.last_filter = access_filter
        # The Access Profile is the authorization authority. A real AGE adapter
        # would translate this already-derived filter into its query; this local
        # store simulates that source-side constraint and the service retains a
        # defensive boundary check after traversal.
        return [path for path in self._paths if access_filter.allows(path)][:limit]

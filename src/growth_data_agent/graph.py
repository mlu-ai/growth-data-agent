"""Entitlement-filtered traversal over the derived evidence graph."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel, Field, model_validator

from .datahub import DataHubEntityMetadata, _metric_name_for_model
from .evidence import EvidenceDocument


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
    seat_tiers: list[str] = Field(default_factory=list)
    graph_namespace: str | None = None


class GraphPath(BaseModel):
    """A bounded evidence path returned from an authorized graph traversal."""

    path_id: str
    nodes: list[GraphNode]

    @model_validator(mode="after")
    def propagate_segment_scope(self) -> GraphPath:
        """Keep the approved Seat Tier scope on every node in a derived path."""
        seat_tiers = {
            seat_tier
            for node in self.nodes
            for seat_tier in node.seat_tiers
        }
        if not seat_tiers:
            seat_tiers = {
                match
                for node in self.nodes
                for match in re.findall(r"(?:1-10|11-50|51-200|201\+)", node.label)
            }
        if seat_tiers:
            for node in self.nodes:
                if not node.seat_tiers:
                    node.seat_tiers = sorted(seat_tiers)
        return self


class EvidenceGraphUnavailableError(OSError):
    """Raised when the derived evidence graph cannot return a safe path."""


_AGE_PRELOAD_CONFIGURATION_ERROR = (
    "Apache AGE is unavailable for this application role. Configure AGE in the "
    "database's session_preload_libraries (for example, ALTER DATABASE ... SET "
    "session_preload_libraries = 'age'), grant the role usage on ag_catalog, and "
    "set APACHE_AGE_PRELOADED=true; do not make the application role a superuser."
)
_AGE_QUERY_ERROR = (
    "Apache AGE query failed after session setup. Verify that the configured graph "
    "exists and that the Cypher statement uses syntax supported by Apache AGE."
)
_AGE_MUTATION_ERROR = (
    "Apache AGE graph mutation failed after session setup. Verify that the application "
    "role owns the dedicated graph and has the required AGE function privileges."
)
_AGE_GRAPH_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z")


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
            and bool(node.tenant_ids)
            and set(node.tenant_ids).issubset(self.tenant_ids)
            and node.classification in self.classifications
            and node.identifier_entitlement in self.identifier_entitlements
            and (
                not self.seat_tiers
                or (
                    bool(node.seat_tiers)
                    and set(node.seat_tiers).issubset(self.seat_tiers)
                )
            )
            for node in path.nodes
        )


class EvidenceGraphStore(Protocol):
    def traverse(
        self,
        query: str,
        access_filter: GraphAccessFilter,
        *,
        limit: int,
        metric_name: str | None = None,
    ) -> list[GraphPath]: ...


class AgeGraphQueryExecutor(Protocol):
    """Execute a parameterized Cypher query against an Apache AGE graph."""

    def query(self, cypher: str, parameters: dict[str, object]) -> Iterable[GraphPath]: ...


class AgeGraphMutationExecutor(Protocol):
    """Execute a bounded write query for the derived AGE index."""

    def execute(self, cypher: str, parameters: dict[str, object]) -> None: ...


_AGE_EVIDENCE_CHAIN_QUERY = """
MATCH path = (metric)-[:EVIDENCE_CHAIN*3..4]->(target)
WHERE metric.node_type = $metric_node_type
  AND metric.node_id = $metric_name
  AND target.node_type IN $incident_or_team_node_types
  AND nodes(path)[1].node_type = $segment_node_type
  AND nodes(path)[2].node_type = $tenant_node_type
  AND (
    (size(nodes(path)) = 4 AND {node_filters_4})
    OR (size(nodes(path)) = 5 AND nodes(path)[3].node_type = $campaign_node_type
        AND nodes(path)[4].node_type = $team_node_type AND {node_filters_5})
  )
RETURN path
LIMIT {limit}
"""
_MAX_GRAPH_RESULT_LIMIT = 100


_AGE_CLEAR_GRAPH_QUERY = """
MATCH (existing:Evidence {graph_namespace: $graph_namespace})
DETACH DELETE existing
"""


_AGE_CREATE_NODES_QUERY = """
UNWIND $nodes AS node
CREATE (created:Evidence)
SET created = node
RETURN count(created) AS result
"""


_AGE_CREATE_EDGES_QUERY = """
UNWIND $edges AS edge
MATCH (source:Evidence {graph_namespace: $graph_namespace, node_key: edge.source_key})
MATCH (target:Evidence {graph_namespace: $graph_namespace, node_key: edge.target_key})
CREATE (source)-[:EVIDENCE_CHAIN {path_id: edge.path_id, position: edge.position}]->(target)
RETURN count(edge) AS result
"""


class ApacheAgeEvidenceGraphStore:
    """Query derived AGE chains after the Access Profile has produced filters."""

    def __init__(
        self,
        query_executor: AgeGraphQueryExecutor,
        *,
        graph_namespace: str = "growth-data-agent",
    ):
        self.query_executor = query_executor
        self.graph_namespace = graph_namespace
        self.last_filter: GraphAccessFilter | None = None
        self.last_query: str | None = None
        self.last_parameters: dict[str, object] | None = None

    def traverse(
        self,
        query: str,
        access_filter: GraphAccessFilter,
        *,
        limit: int,
        metric_name: str | None = None,
    ) -> list[GraphPath]:
        """Push every entitlement into AGE and defensively filter returned paths."""
        self.last_filter = access_filter
        query_limit = min(max(limit * 10, limit), _MAX_GRAPH_RESULT_LIMIT)
        self.last_query = _age_evidence_chain_query(query_limit)
        if not access_filter.tenant_ids or not metric_name:
            self.last_parameters = None
            return []
        self.last_parameters = {
            "query": query,
            "query_terms": _query_terms(query),
            "metric_name": metric_name,
            "metric_node_type": "metric",
            "incident_or_team_node_types": ["incident", "team"],
            "segment_node_type": "segment",
            "tenant_node_type": "tenant",
            "campaign_node_type": "campaign",
            "team_node_type": "team",
            "graph_namespace": self.graph_namespace,
            "products": list(access_filter.products),
            "regions": list(access_filter.regions),
            "tenant_ids": list(access_filter.tenant_ids),
            "classifications": list(access_filter.classifications),
            "identifier_entitlements": list(access_filter.identifier_entitlements),
            "seat_tiers": list(access_filter.seat_tiers),
        }
        paths = self.query_executor.query(self.last_query, self.last_parameters)
        return [
            path
            for path in paths
            if _is_supported_evidence_chain(path, metric_name=metric_name)
            and all(node.graph_namespace == self.graph_namespace for node in path.nodes)
            and access_filter.allows(path)
            and _path_matches_query(path, query)
        ][:limit]


class AgeGraphMaterializationResult(BaseModel):
    """Counts for one complete replacement of the derived AGE graph."""

    path_count: int = Field(ge=0)
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)


class ApacheAgeEvidenceGraphMaterializer:
    """Replace the AGE index from approved catalog and document metadata."""

    def __init__(self, mutation_executor: AgeGraphMutationExecutor):
        self.mutation_executor = mutation_executor

    def replace(
        self,
        catalog_entities: Iterable[DataHubEntityMetadata],
        documents: Iterable[EvidenceDocument],
        *,
        graph_namespace: str = "growth-data-agent",
    ) -> AgeGraphMaterializationResult:
        paths = DerivedEvidenceGraphBuilder().build(catalog_entities, documents)
        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []
        for path in paths:
            for node in path.nodes:
                node_key = _graph_node_key(node)
                nodes[node_key] = {
                    **node.model_dump(),
                    "node_key": node_key,
                    "graph_namespace": graph_namespace,
                }
            for position, (source, target) in enumerate(zip(path.nodes, path.nodes[1:])):
                edges.append(
                    {
                        "source_key": _graph_node_key(source),
                        "target_key": _graph_node_key(target),
                        "path_id": path.path_id,
                        "position": position,
                    }
                )
        if not nodes:
            self.mutation_executor.execute(
                _AGE_CLEAR_GRAPH_QUERY,
                {"graph_namespace": graph_namespace},
            )
            return AgeGraphMaterializationResult(
                path_count=0,
                node_count=0,
                edge_count=0,
            )
        self.mutation_executor.execute(
            _AGE_CLEAR_GRAPH_QUERY,
            {"graph_namespace": graph_namespace},
        )
        self.mutation_executor.execute(
            _AGE_CREATE_NODES_QUERY,
            {"nodes": list(nodes.values())},
        )
        self.mutation_executor.execute(
            _AGE_CREATE_EDGES_QUERY,
            {
                "graph_namespace": graph_namespace,
                "edges": edges,
            },
        )
        return AgeGraphMaterializationResult(
            path_count=len(paths),
            node_count=len(nodes),
            edge_count=len(edges),
        )


class PsycopgAgeGraphQueryExecutor:
    """Execute the AGE Cypher boundary in a read-only PostgreSQL transaction."""

    def __init__(
        self,
        database_url: str,
        *,
        graph_name: str = "growth_evidence",
        age_preloaded: bool = False,
    ):
        self.database_url = database_url
        self.graph_name = _validate_age_graph_name(graph_name)
        self.age_preloaded = age_preloaded

    def query(self, cypher: str, parameters: dict[str, object]) -> list[GraphPath]:
        try:
            with psycopg.connect(self.database_url) as connection:
                with connection.transaction():
                    connection.execute("SET TRANSACTION READ ONLY")
                    _configure_age_session(connection, age_preloaded=self.age_preloaded)
                    with connection.cursor() as cursor:
                        cursor.execute(
                            _age_cypher_statement(
                                self.graph_name,
                                cypher,
                                result_column="path",
                            ),
                            (json.dumps(parameters),),
                            prepare=True,
                        )
                        try:
                            return [_graph_path_from_age(row[0]) for row in cursor.fetchall()]
                        except (TypeError, ValueError, KeyError) as error:
                            raise EvidenceGraphUnavailableError(
                                "Apache AGE returned a malformed evidence path."
                            ) from error
        except EvidenceGraphUnavailableError:
            raise
        except psycopg.Error as error:
            raise EvidenceGraphUnavailableError(_AGE_QUERY_ERROR) from error


class PsycopgAgeGraphMutationExecutor:
    """Execute AGE mutations in a normal PostgreSQL transaction."""

    def __init__(
        self,
        database_url: str,
        *,
        graph_name: str = "growth_evidence",
        age_preloaded: bool = False,
    ):
        self.database_url = database_url
        self.graph_name = _validate_age_graph_name(graph_name)
        self.age_preloaded = age_preloaded

    def execute(self, cypher: str, parameters: dict[str, object]) -> None:
        try:
            with psycopg.connect(self.database_url) as connection:
                with connection.transaction():
                    _configure_age_session(connection, age_preloaded=self.age_preloaded)
                    with connection.cursor() as cursor:
                        try:
                            with connection.transaction():
                                cursor.execute(
                                    sql.SQL("SELECT ag_catalog.create_graph({graph})").format(
                                        graph=sql.Literal(self.graph_name)
                                    )
                                )
                        except psycopg.errors.InvalidSchemaName as error:
                            if error.diag.message_primary != (
                                f'graph "{self.graph_name}" already exists'
                            ):
                                raise
                        cursor.execute(
                            _age_cypher_statement(
                                self.graph_name,
                                cypher,
                                result_column="result",
                            ),
                            (json.dumps(parameters),),
                            prepare=True,
                        )
                        cursor.fetchall()
        except psycopg.Error as error:
            raise EvidenceGraphUnavailableError(_AGE_MUTATION_ERROR) from error


def _configure_age_session(connection, *, age_preloaded: bool) -> None:
    """Prepare AGE without requiring LOAD when the server preloads its library."""
    try:
        if not age_preloaded:
            connection.execute("LOAD 'age'")
        connection.execute('SET search_path = ag_catalog, "$user", public')
    except psycopg.Error as error:
        raise EvidenceGraphUnavailableError(_AGE_PRELOAD_CONFIGURATION_ERROR) from error


def _validate_age_graph_name(graph_name: str) -> str:
    """Keep AGE graph names strict before embedding the required name constant."""
    if not _AGE_GRAPH_NAME_PATTERN.fullmatch(graph_name):
        raise ValueError(
            "APACHE_AGE_GRAPH_NAME must start with a letter or underscore and contain "
            "only letters, digits, and underscores (maximum 63 characters)."
        )
    return graph_name


def _age_cypher_statement(graph_name: str, cypher: str, *, result_column: str) -> sql.Composed:
    """Render AGE's name/cstring constants while keeping params parameterized."""
    return sql.SQL(
        "SELECT * FROM ag_catalog.cypher({graph}, {query}, %s) "
        "AS ({result} ag_catalog.agtype)"
    ).format(
        graph=sql.Literal(_validate_age_graph_name(graph_name)),
        query=_age_cstring_literal(cypher),
        result=sql.Identifier(result_column),
    )


def _age_cstring_literal(value: str) -> sql.SQL:
    """Render a cstring as a safe dollar-quoted SQL constant for AGE."""
    for tag in ("gda", "gda1", "gda2", "gda3", "gda4"):
        delimiter = f"${tag}$"
        if delimiter not in value:
            return sql.SQL(f"{delimiter}{value}{delimiter}")
    raise ValueError("AGE Cypher query contains an unsupported dollar-quote delimiter.")


def apache_age_preloaded_from_environment() -> bool:
    """Return whether the database/session administrator preloaded Apache AGE."""
    return os.environ.get("APACHE_AGE_PRELOADED", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DerivedEvidenceGraphBuilder:
    """Derive graph paths from approved catalog and document-ingestion metadata."""

    def build(
        self,
        catalog_entities: Iterable[DataHubEntityMetadata],
        documents: Iterable[EvidenceDocument],
    ) -> tuple[GraphPath, ...]:
        entities = tuple(catalog_entities)
        metrics_by_name = {
            entity.entity_name: entity
            for entity in entities
            if entity.entity_type == "metric"
        }
        for entity in entities:
            if entity.entity_type != "model":
                continue
            metric_name = _metric_name_for_model(entity.entity_name)
            if metric_name is not None:
                metrics_by_name.setdefault(metric_name, entity)
        paths: list[GraphPath] = []
        for document in documents:
            metric_name = document.metric_name
            if metric_name is None:
                product_metrics = [
                    entity
                    for entity in metrics_by_name.values()
                    if entity.product == document.product
                ]
                metric = product_metrics[0] if len(product_metrics) == 1 else None
            else:
                metric = metrics_by_name.get(metric_name)
            if metric is None:
                continue
            if metric_name is None:
                metric_name = next(
                    name for name, candidate in metrics_by_name.items() if candidate is metric
                )
            seat_tier = _seat_tier_from_scope(document.tenant_scope)
            classification = document.classification
            direct_single_tenant = (
                document.identifier_entitlement == "direct"
                and len(document.tenant_ids) == 1
            )
            tenant_label = (
                document.tenant_ids[0]
                if direct_single_tenant
                else f"{document.tenant_scope} cohort"
            )
            tenant_id = (
                document.tenant_ids[0]
                if direct_single_tenant
                else f"tenant-cohort-{_slug(document.tenant_scope)}"
            )
            nodes = [
                _derived_node(
                    node_id=metric_name,
                    node_type="metric",
                    label=metric_name.replace("_", " ").title(),
                    product=document.product,
                    region=document.region,
                    tenant_ids=document.tenant_ids,
                    classification=metric.classification,
                    identifier_entitlement="none",
                    seat_tier=seat_tier,
                ),
                _derived_node(
                    node_id=f"segment-{_slug(document.tenant_scope)}",
                    node_type="segment",
                    label=document.tenant_scope,
                    product=document.product,
                    region=document.region,
                    tenant_ids=document.tenant_ids,
                    classification=classification,
                    identifier_entitlement=document.identifier_entitlement,
                    seat_tier=seat_tier,
                ),
                _derived_node(
                    node_id=tenant_id,
                    node_type="tenant",
                    label=tenant_label,
                    product=document.product,
                    region=document.region,
                    tenant_ids=document.tenant_ids,
                    classification=classification,
                    identifier_entitlement=document.identifier_entitlement,
                    seat_tier=seat_tier,
                ),
            ]
            document_node_type = "team" if "campaign" in document.document_id else "incident"
            nodes.append(
                _derived_node(
                    node_id=document.document_id,
                    node_type=document_node_type,
                    label=(
                        document.accountable_team or f"{document.product} Evidence Team"
                        if document_node_type == "team"
                        else document.title
                    ),
                    product=document.product,
                    region=document.region,
                    tenant_ids=document.tenant_ids,
                    classification=classification,
                    identifier_entitlement=document.identifier_entitlement,
                    seat_tier=seat_tier,
                )
            )
            path_id = (
                f"{document.document_id}-identifier-chain"
                if direct_single_tenant
                else f"{document.document_id}-chain"
            )
            paths.append(GraphPath(path_id=path_id, nodes=nodes))
        return tuple(paths)


class InMemoryEvidenceGraphStore:
    """Deterministic graph boundary used by the local POC."""

    def __init__(self, paths: Iterable[GraphPath], *, graph_namespace: str = "growth-data-agent"):
        self._paths = tuple(paths)
        self.graph_namespace = graph_namespace
        self.last_filter: GraphAccessFilter | None = None

    def traverse(
        self,
        query: str,
        access_filter: GraphAccessFilter,
        *,
        limit: int,
        metric_name: str | None = None,
    ) -> list[GraphPath]:
        del query
        self.last_filter = access_filter
        # The Access Profile is the authorization authority. A real AGE adapter
        # would translate this already-derived filter into its query; this local
        # store simulates that source-side constraint and the service retains a
        # defensive boundary check after traversal.
        return [
            path
            for path in self._paths
            if (metric_name is None or _is_supported_evidence_chain(path, metric_name=metric_name))
            and all(node.graph_namespace == self.graph_namespace for node in path.nodes)
            and access_filter.allows(path)
        ][:limit]


def _query_terms(query: str) -> list[str]:
    terms = [term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 2]
    return terms or [""]


def _age_evidence_chain_query(limit: int) -> str:
    """Bound the AGE result limit as a SQL/Cypher literal, not an external variable."""
    if not 1 <= limit <= _MAX_GRAPH_RESULT_LIMIT:
        raise ValueError(f"AGE graph result limit must be between 1 and {_MAX_GRAPH_RESULT_LIMIT}.")
    return _AGE_EVIDENCE_CHAIN_QUERY.format(
        limit=limit,
        node_filters_4=" AND ".join(_age_node_access_filter(index) for index in range(4)),
        node_filters_5=" AND ".join(_age_node_access_filter(index) for index in range(5)),
    )


def _age_node_access_filter(index: int) -> str:
    """Build direct node predicates; AGE 1.8 cannot plan all()/any() path predicates."""
    node = f"nodes(path)[{index}]"
    return " AND ".join(
        (
            f"{node}.graph_namespace = $graph_namespace",
            f"{node}.product IN $products",
            f"{node}.region IN $regions",
            f"size({node}.tenant_ids) > 0",
            f"{node}.tenant_ids[0] IN $tenant_ids",
            f"{node}.classification IN $classifications",
            f"{node}.identifier_entitlement IN $identifier_entitlements",
        )
    )


def _path_matches_query(path: GraphPath, query: str) -> bool:
    """Apply text matching after bounded retrieval when AGE lacks any()/all()."""
    terms = _query_terms(query)
    return any(term in node.label.casefold() for node in path.nodes for term in terms)


def _is_supported_evidence_chain(path: GraphPath, *, metric_name: str | None = None) -> bool:
    node_types = [node.node_type for node in path.nodes]
    shape_is_supported = (
        (
            len(node_types) == 4
            and node_types[:3] == ["metric", "segment", "tenant"]
            and node_types[3] in {"incident", "team"}
        )
        or (
            len(node_types) == 5
            and node_types[:4] == ["metric", "segment", "tenant", "campaign"]
            and node_types[4] == "team"
        )
    )
    return (
        bool(path.nodes)
        and (metric_name is None or path.nodes[0].node_id == metric_name)
        and shape_is_supported
    )


def _derived_node(
    *,
    node_id: str,
    node_type: str,
    label: str,
    product: str,
    region: str,
    tenant_ids: list[str],
    classification: str,
    identifier_entitlement: str,
    seat_tier: str | None,
    graph_namespace: str = "growth-data-agent",
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        product=product,
        region=region,
        tenant_ids=list(tenant_ids),
        classification=classification,
        identifier_entitlement=identifier_entitlement,
        seat_tiers=[seat_tier] if seat_tier else [],
        graph_namespace=graph_namespace,
    )


def _seat_tier_from_scope(scope: str) -> str | None:
    match = re.search(r"(?:1-10|11-50|51-200|201\+)", scope)
    return match.group(0) if match else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _graph_node_key(node: GraphNode) -> str:
    tenant_scope = ",".join(sorted(node.tenant_ids))
    seat_scope = ",".join(sorted(node.seat_tiers))
    return ":".join(
        (
            node.product,
            node.region,
            node.node_type,
            node.node_id,
            node.classification,
            node.identifier_entitlement,
            tenant_scope,
            seat_scope,
        )
    )


def _graph_path_from_age(value: object) -> GraphPath:
    """Decode AGE's alternating vertex/edge agtype path into governed nodes."""
    if isinstance(value, str):
        value = json.loads(_strip_agtype_annotations(value))
    if isinstance(value, dict):
        raw_nodes = value.get("nodes") or value.get("vertices") or value.get("path")
    else:
        raw_nodes = value
    if not isinstance(raw_nodes, list):
        raise ValueError("AGE returned a path without a node list.")
    nodes: list[GraphNode] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError("AGE returned a malformed graph node.")
        if "start_id" in raw_node or "startid" in raw_node:
            continue
        properties = raw_node.get("properties") or raw_node
        properties = dict(properties)
        properties.setdefault("node_id", str(raw_node.get("id", "")))
        properties.setdefault("node_type", raw_node.get("label", ""))
        properties.setdefault("label", raw_node.get("label", properties["node_id"]))
        nodes.append(GraphNode.model_validate(properties))
    if len(nodes) < 4:
        raise ValueError("AGE returned a path shorter than the governed evidence chain.")
    path_id = nodes[0].node_id + "-" + nodes[-1].node_id
    return GraphPath(path_id=path_id, nodes=nodes)


def _strip_agtype_annotations(value: str) -> str:
    return re.sub(r"::(?:path|vertex|edge)\b", "", value.strip())

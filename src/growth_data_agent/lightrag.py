"""Fail-closed LightRAG retrieval over an already-authorized evidence scope.

LightRAG is deliberately kept behind this small internal contract. It may
retrieve bounded evidence references, but it is not a semantic, permission, or
causal authority and it never generates answer text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Final, Literal, Protocol, TypeVar, cast, final

from pydantic import BaseModel, Field

from .evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    EvidenceLifecycleState,
    EvidencePrincipalGrant,
    QdrantEvidenceStore,
    _provenance_for,
    _vectorize,
)
from .graph import (
    ApacheAgeEvidenceGraphStore,
    GraphAccessFilter,
    GraphNode,
    GraphPath,
    InMemoryEvidenceGraphStore,
)

_MAX_LIGHTRAG_RESULTS = 3
_REFERENCE_KINDS = Literal["chunk", "entity", "relation"]
_AUTHORIZED_SCOPE_TOKEN: Final = object()
_LIGHTRAG_CAPABILITY_TOKEN: Final = object()
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class LightRAGAuthorizationError(PermissionError):
    """Raised when LightRAG authorization cannot be proven at retrieval time."""


class LightRAGRetrievalError(RuntimeError):
    """Raised when LightRAG cannot return a safe bounded reference set."""


class LightRAGEvidenceReference(BaseModel):
    """A bounded model-facing reference with exact source and policy metadata."""

    reference_id: str = Field(min_length=1)
    reference_kind: _REFERENCE_KINDS
    source_document_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    revision_fingerprint: str | None = None
    source_page_id: str | None = None
    metric_name: str | None = None
    product: str = Field(min_length=1)
    region: str = Field(min_length=1)
    tenant_ids: list[str] = Field(min_length=1)
    tenant_scope: str = Field(min_length=1)
    classification: str = Field(min_length=1)
    identifier_entitlement: str = Field(min_length=1)
    access_groups: list[str] = Field(default_factory=list)
    direct_principal_grants: list[EvidencePrincipalGrant] = Field(default_factory=list)
    policy_expires_at: datetime
    lifecycle_state: EvidenceLifecycleState = EvidenceLifecycleState.ACTIVE
    related_entity_references: list[LightRAGEvidenceReference] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)

    @classmethod
    def from_document(
        cls,
        document: EvidenceDocument,
        *,
        reference_kind: _REFERENCE_KINDS = "chunk",
        reference_id: str | None = None,
    ) -> LightRAGEvidenceReference:
        """Create a reference without dropping provenance from an evidence revision."""
        provenance = _provenance_for(document)
        return cls(
            reference_id=reference_id or f"{reference_kind}:{provenance.chunk_id}",
            reference_kind=reference_kind,
            source_document_id=provenance.source_document_id,
            source_url=provenance.source_url,
            source_revision=provenance.source_revision,
            chunk_id=provenance.chunk_id,
            revision_fingerprint=document.revision_fingerprint,
            source_page_id=document.source_page_id,
            metric_name=document.metric_name,
            product=document.product,
            region=document.region,
            tenant_ids=list(document.tenant_ids),
            tenant_scope=document.tenant_scope,
            classification=document.classification,
            identifier_entitlement=document.identifier_entitlement,
            access_groups=list(document.access_groups),
            direct_principal_grants=list(document.direct_principal_grants),
            policy_expires_at=document.policy_expires_at,
            lifecycle_state=document.lifecycle_state,
        )


class LightRAGChunkRecord(BaseModel):
    """A chunk stored in LightRAG's vector retrieval index."""

    reference: LightRAGEvidenceReference
    text: str = Field(min_length=1)
    embedding: list[float] | None = None


class LightRAGEntityRecord(BaseModel):
    """An entity stored in LightRAG's graph retrieval index."""

    reference: LightRAGEvidenceReference
    name: str = Field(min_length=1)
    description: str = ""


class LightRAGRelationRecord(BaseModel):
    """A relation stored in LightRAG's graph retrieval index."""

    reference: LightRAGEvidenceReference
    source_entity: LightRAGEvidenceReference
    target_entity: LightRAGEvidenceReference
    description: str = ""


class LightRAGEvidenceChain(BaseModel):
    """Bounded typed LightRAG output; it contains evidence records, never answer prose."""

    supporting_chunks: list[LightRAGChunkRecord] = Field(max_length=_MAX_LIGHTRAG_RESULTS)
    entities: list[LightRAGEntityRecord] = Field(max_length=_MAX_LIGHTRAG_RESULTS)
    relations: list[LightRAGRelationRecord] = Field(max_length=_MAX_LIGHTRAG_RESULTS)
    references: list[LightRAGEvidenceReference] = Field(max_length=_MAX_LIGHTRAG_RESULTS)


class _ReferenceRecord(Protocol):
    reference: LightRAGEvidenceReference


ReferenceRecordT = TypeVar("ReferenceRecordT", bound=_ReferenceRecord)


@dataclass(frozen=True)
class LightRAGStoreCall:
    """Auditable record of one governed vector or graph retrieval operation."""

    kind: Literal["chunk_vector", "entity_graph", "relation_graph"]
    query: str
    authorized_reference_ids: frozenset[str]
    returned_reference_ids: frozenset[str]


class LightRAGAuthorizedView:
    """Backend-created capability exposing only already-authorized retrieval methods."""

    __slots__ = (
        "_capability",
        "_owner",
        "_chunks",
        "_entities",
        "_relations",
        "_chunk_retriever",
        "_entity_retriever",
        "_relation_retriever",
    )

    def __init__(
        self,
        owner: object,
        capability: _LightRAGAuthorizationCapability,
        *,
        chunks: tuple[LightRAGChunkRecord, ...] = (),
        entities: tuple[LightRAGEntityRecord, ...] = (),
        relations: tuple[LightRAGRelationRecord, ...] = (),
        chunk_retriever: Callable[[str, int], list[LightRAGChunkRecord]] | None = None,
        entity_retriever: Callable[[str, int], list[LightRAGEntityRecord]] | None = None,
        relation_retriever: Callable[[str, int], list[LightRAGRelationRecord]] | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _LIGHTRAG_CAPABILITY_TOKEN:
            raise TypeError("LightRAG authorized views are backend-created capabilities.")
        if capability.owner is not owner:
            raise LightRAGAuthorizationError("LightRAG view owner does not match its capability.")
        self._owner = owner
        self._capability = capability
        self._chunks = chunks
        self._entities = entities
        self._relations = relations
        self._chunk_retriever = chunk_retriever
        self._entity_retriever = entity_retriever
        self._relation_retriever = relation_retriever

    def proves(self, capability: _LightRAGAuthorizationCapability) -> bool:
        return self._capability is capability and self._owner is capability.owner

    def retrieve_chunk_vectors(self, query: str, *, limit: int) -> list[LightRAGChunkRecord]:
        result = (
            self._chunk_retriever(query, _bounded_limit(limit))
            if self._chunk_retriever is not None
            else _rank_chunk_records(query, self._chunks, limit)
        )
        _record_store_call(self._owner, "chunk_vector", query, self._chunks, result)
        return result

    def retrieve_entity_graph(self, query: str, *, limit: int) -> list[LightRAGEntityRecord]:
        result = (
            self._entity_retriever(query, _bounded_limit(limit))
            if self._entity_retriever is not None
            else _rank_graph_records(query, self._entities, lambda record: record.name, limit)
        )
        _record_store_call(self._owner, "entity_graph", query, self._entities, result)
        return result

    def retrieve_relation_graph(
        self, query: str, *, limit: int
    ) -> list[LightRAGRelationRecord]:
        result = (
            self._relation_retriever(query, _bounded_limit(limit))
            if self._relation_retriever is not None
            else _rank_graph_records(
                query,
                self._relations,
                lambda record: (
                    f"{record.source_entity.reference_id} "
                    f"{record.target_entity.reference_id} {record.description}"
                ),
                limit,
            )
        )
        _record_store_call(self._owner, "relation_graph", query, self._relations, result)
        return result


class LightRAGRetrievalStore:
    """Nominal internal store seam; only sealed concrete implementations are admitted."""

    def issue_capability(
        self,
        scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> _LightRAGAuthorizationCapability:
        raise LightRAGAuthorizationError("LightRAG capability issuance is not implemented.")

    def authorized_view(
        self,
        capability: _LightRAGAuthorizationCapability,
    ) -> LightRAGAuthorizedView:
        raise LightRAGAuthorizationError("LightRAG authorized retrieval is not implemented.")


class InMemoryLightRAGStore(LightRAGRetrievalStore):
    """Concrete deterministic LightRAG vector/graph store for the local POC.

    This is a retrieval seam, not a tuple fixture: chunks use vector
    similarity, while entities and relations use graph-record search. The
    store builds an authorized view before any query is run.
    """

    __slots__ = ("_chunks", "_entities", "_relations", "calls")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("InMemoryLightRAGStore authorization cannot be bypassed by subclassing.")

    def __init__(
        self,
        *,
        chunks: Iterable[LightRAGChunkRecord] = (),
        entities: Iterable[LightRAGEntityRecord] = (),
        relations: Iterable[LightRAGRelationRecord] = (),
    ) -> None:
        self._chunks = tuple(record.model_copy(deep=True) for record in chunks)
        self._entities = tuple(record.model_copy(deep=True) for record in entities)
        self._relations = tuple(record.model_copy(deep=True) for record in relations)
        self.calls: list[LightRAGStoreCall] = []

    def authorized_view(
        self,
        capability: _LightRAGAuthorizationCapability,
    ) -> LightRAGAuthorizedView:
        _require_capability(self, capability)
        capability.scope.revalidate(capability.access_filter)
        return LightRAGAuthorizedView(
            self,
            capability,
            chunks=self._authorized_records(self._chunks, capability.scope),
            entities=self._authorized_records(self._entities, capability.scope),
            relations=self._authorized_relations(self._relations, capability.scope),
            _token=_LIGHTRAG_CAPABILITY_TOKEN,
        )

    def issue_capability(
        self,
        scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> _LightRAGAuthorizationCapability:
        if not isinstance(scope, AuthorizedEvidenceRevisionSet) or not isinstance(
            access_filter, EvidenceAccessFilter
        ):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set and access filter."
            )
        scope.revalidate(access_filter)
        return _LightRAGAuthorizationCapability._issue(
            self, scope, access_filter, _LIGHTRAG_CAPABILITY_TOKEN
        )

    @staticmethod
    def _authorized_records(
        records: Iterable[ReferenceRecordT],
        authorized_scope: AuthorizedEvidenceRevisionSet,
    ) -> tuple[ReferenceRecordT, ...]:
        return tuple(
            record
            for record in records
            if authorized_scope.allows_reference(record.reference)
        )

    @staticmethod
    def _authorized_relations(
        records: Iterable[LightRAGRelationRecord],
        authorized_scope: AuthorizedEvidenceRevisionSet,
    ) -> tuple[LightRAGRelationRecord, ...]:
        return tuple(
            record
            for record in records
            if (
                authorized_scope.allows_reference(record.reference)
                and record.reference.related_entity_references
                == [record.source_entity, record.target_entity]
                and _is_authorized_entity(record.source_entity, authorized_scope)
                and _is_authorized_entity(record.target_entity, authorized_scope)
            )
        )

    def _record_call(
        self,
        kind: Literal["chunk_vector", "entity_graph", "relation_graph"],
        query: str,
        authorized_records: Iterable[_ReferenceRecord],
        returned_records: Iterable[_ReferenceRecord],
    ) -> None:
        self.calls.append(
            LightRAGStoreCall(
                kind=kind,
                query=query,
                authorized_reference_ids=frozenset(
                    record.reference.reference_id for record in authorized_records
                ),
                returned_reference_ids=frozenset(
                    record.reference.reference_id for record in returned_records
                ),
            )
        )


@dataclass(frozen=True, init=False)
class AuthorizedEvidenceRevisionSet:
    """Opaque active Evidence Revisions authorized before LightRAG retrieval."""

    _revisions: tuple[EvidenceDocument, ...]
    _revision_source: Callable[[EvidenceAccessFilter], Iterable[EvidenceDocument]] | None

    def __init__(
        self,
        revisions: Iterable[EvidenceDocument],
        *,
        _token: object | None = None,
        _revision_source: Callable[
            [EvidenceAccessFilter], Iterable[EvidenceDocument]
        ] | None = None,
    ) -> None:
        if _token is not _AUTHORIZED_SCOPE_TOKEN:
            raise TypeError(
                "AuthorizedEvidenceRevisionSet must be created by its authorization factory."
            )
        snapshot = tuple(document.model_copy(deep=True) for document in revisions)
        if any(
            document.lifecycle_state is not EvidenceLifecycleState.ACTIVE
            for document in snapshot
        ):
            raise LightRAGAuthorizationError(
                "The LightRAG scope contains an inactive Evidence Revision."
            )
        keys = [
            (
                _provenance_for(document).source_document_id,
                document.source_revision,
                _provenance_for(document).chunk_id,
            )
            for document in snapshot
        ]
        if len(keys) != len(set(keys)):
            raise LightRAGAuthorizationError(
                "The LightRAG scope contains duplicate Evidence Revision references."
            )
        object.__setattr__(self, "_revisions", snapshot)
        object.__setattr__(self, "_revision_source", _revision_source)

    @property
    def revisions(self) -> tuple[EvidenceDocument, ...]:
        """Return defensive copies so callers cannot mutate the authorization proof."""
        return tuple(document.model_copy(deep=True) for document in self._revisions)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[EvidenceDocument],
        access_filter: EvidenceAccessFilter,
        *,
        revision_source: Callable[[EvidenceAccessFilter], Iterable[EvidenceDocument]] | None = None,
    ) -> AuthorizedEvidenceRevisionSet:
        revisions = tuple(documents)
        if any(
            document.lifecycle_state is not EvidenceLifecycleState.ACTIVE
            or not access_filter.allows(document)
            for document in revisions
        ):
            raise LightRAGAuthorizationError(
                "The LightRAG scope contains an inactive or unauthorized Evidence Revision."
            )
        return cls(
            revisions,
            _token=_AUTHORIZED_SCOPE_TOKEN,
            _revision_source=revision_source,
        )

    def revalidate(self, access_filter: EvidenceAccessFilter) -> None:
        """Recheck current lifecycle and policy state immediately before retrieval."""
        if not isinstance(access_filter, EvidenceAccessFilter):
            raise LightRAGAuthorizationError(
                "LightRAG requires the authenticated current Evidence access filter."
            )
        if self._revision_source is not None:
            try:
                current_revisions = tuple(self._revision_source(access_filter))
            except Exception as error:
                raise LightRAGAuthorizationError(
                    "The current Evidence Revision authorization source is unavailable."
                ) from error
            current_by_key = {
                _revision_key(document): document for document in current_revisions
            }
            if any(
                current_by_key.get(_revision_key(document)) is None
                or current_by_key[_revision_key(document)].model_dump(mode="json")
                != document.model_dump(mode="json")
                for document in self._revisions
            ):
                raise LightRAGAuthorizationError(
                    "The LightRAG authorization scope is stale or has been revoked."
                )
        if not self._revisions or any(
            document.lifecycle_state is not EvidenceLifecycleState.ACTIVE
            or not access_filter.allows(document)
            for document in self._revisions
        ):
            raise LightRAGAuthorizationError(
                "The LightRAG authorization scope is no longer active or authorized."
            )

    def allows_reference(self, reference: LightRAGEvidenceReference) -> bool:
        """Require reference provenance and policy metadata to match the scope exactly."""
        if reference.lifecycle_state is not EvidenceLifecycleState.ACTIVE:
            return False
        for document in self._revisions:
            provenance = _provenance_for(document)
            canonical_prefix = {
                "chunk": f"chunk:{provenance.chunk_id}",
                "entity": f"entity:{provenance.source_document_id}",
                "relation": f"relation:{provenance.source_document_id}",
            }[reference.reference_kind]
            if (
                (
                    reference.reference_id == canonical_prefix
                    or reference.reference_id.startswith(f"{canonical_prefix}:")
                )
                and (
                    reference.reference_kind == "relation"
                    or not reference.related_entity_references
                )
                and provenance.source_document_id == reference.source_document_id
                and provenance.source_url == reference.source_url
                and provenance.source_revision == reference.source_revision
                and provenance.chunk_id == reference.chunk_id
                and document.revision_fingerprint == reference.revision_fingerprint
                and document.source_page_id == reference.source_page_id
                and document.metric_name == reference.metric_name
                and document.product == reference.product
                and document.region == reference.region
                and set(document.tenant_ids) == set(reference.tenant_ids)
                and document.tenant_scope == reference.tenant_scope
                and document.classification == reference.classification
                and document.identifier_entitlement == reference.identifier_entitlement
                and document.access_groups == reference.access_groups
                and document.direct_principal_grants == reference.direct_principal_grants
                and document.policy_expires_at == reference.policy_expires_at
                and (
                    reference.reference_kind != "relation"
                    or (
                        len(reference.related_entity_references) == 2
                        and all(
                            _is_authorized_entity(entity, self)
                            for entity in reference.related_entity_references
                        )
                    )
                )
            ):
                return True
        return False


@dataclass(frozen=True, init=False)
class _LightRAGAuthorizationCapability:
    """Opaque backend-issued proof binding one scope to one concrete store."""

    owner: object
    scope: AuthorizedEvidenceRevisionSet
    access_filter: EvidenceAccessFilter
    _token: object

    @classmethod
    def _issue(
        cls,
        owner: object,
        scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        token: object,
    ) -> _LightRAGAuthorizationCapability:
        if token is not _LIGHTRAG_CAPABILITY_TOKEN:
            raise LightRAGAuthorizationError("LightRAG capability issuance is backend-only.")
        capability = object.__new__(cls)
        object.__setattr__(capability, "owner", owner)
        object.__setattr__(capability, "scope", scope)
        object.__setattr__(capability, "access_filter", access_filter)
        object.__setattr__(capability, "_token", token)
        return capability


def _require_capability(
    owner: object,
    capability: _LightRAGAuthorizationCapability,
) -> None:
    if (
        not isinstance(capability, _LightRAGAuthorizationCapability)
        or capability._token is not _LIGHTRAG_CAPABILITY_TOKEN
        or capability.owner is not owner
        or not isinstance(capability.scope, AuthorizedEvidenceRevisionSet)
        or not isinstance(capability.access_filter, EvidenceAccessFilter)
    ):
        raise LightRAGAuthorizationError(
            "LightRAG store requires a backend-issued authorization capability."
        )


class QdrantAGELightRAGStore(LightRAGRetrievalStore):
    """Concrete bridge from the repository's Qdrant and AGE stores to LightRAG."""

    __slots__ = ("qdrant_store", "graph_store", "calls")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("Qdrant/AGE LightRAG authorization cannot be bypassed by subclassing.")

    def __init__(
        self,
        qdrant_store: QdrantEvidenceStore,
        graph_store: ApacheAgeEvidenceGraphStore | InMemoryEvidenceGraphStore,
    ) -> None:
        if type(qdrant_store) is not QdrantEvidenceStore:
            raise LightRAGAuthorizationError(
                "LightRAG requires the concrete Qdrant evidence store."
            )
        if type(graph_store) not in {ApacheAgeEvidenceGraphStore, InMemoryEvidenceGraphStore}:
            raise LightRAGAuthorizationError(
                "LightRAG requires the concrete AGE-compatible graph store."
            )
        self.qdrant_store = qdrant_store
        self.graph_store = graph_store
        self.calls: list[LightRAGStoreCall] = []

    def authorized_view(
        self,
        capability: _LightRAGAuthorizationCapability,
    ) -> LightRAGAuthorizedView:
        _require_capability(self, capability)
        capability.scope.revalidate(capability.access_filter)
        scope_documents = capability.scope.revisions
        scope_ids = tuple(
            _provenance_for(document).source_document_id for document in scope_documents
        )
        chunk_records = tuple(
            LightRAGChunkRecord(
                reference=LightRAGEvidenceReference.from_document(document),
                text=document.text,
            )
            for document in scope_documents
        )
        revision_keys = _scope_revision_keys(capability.scope)
        return LightRAGAuthorizedView(
            self,
            capability,
            chunks=chunk_records,
            chunk_retriever=lambda query, limit: self._retrieve_chunks(
                query, capability, scope_ids, revision_keys, limit
            ),
            entity_retriever=lambda query, limit: self._retrieve_entities(
                query, capability, limit
            ),
            relation_retriever=lambda query, limit: self._retrieve_relations(
                query, capability, limit
            ),
            _token=_LIGHTRAG_CAPABILITY_TOKEN,
        )

    def issue_capability(
        self,
        scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> _LightRAGAuthorizationCapability:
        if not isinstance(scope, AuthorizedEvidenceRevisionSet) or not isinstance(
            access_filter, EvidenceAccessFilter
        ):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set and access filter."
            )
        scope.revalidate(access_filter)
        return _LightRAGAuthorizationCapability._issue(
            self, scope, access_filter, _LIGHTRAG_CAPABILITY_TOKEN
        )

    def _retrieve_chunks(
        self,
        query: str,
        capability: _LightRAGAuthorizationCapability,
        scope_ids: tuple[str, ...],
        revision_keys: tuple[tuple[str, str, str], ...],
        limit: int,
    ) -> list[LightRAGChunkRecord]:
        capability.scope.revalidate(capability.access_filter)
        documents = self.qdrant_store.retrieve_scoped(
            query,
            capability.access_filter,
            scope_ids,
            limit=limit,
            authorized_revision_keys=revision_keys,
        )
        return [
            LightRAGChunkRecord(
                reference=LightRAGEvidenceReference.from_document(document),
                text=document.text,
            )
            for document in documents
            if capability.scope.allows_reference(
                LightRAGEvidenceReference.from_document(document)
            )
        ]

    def _graph_paths(
        self,
        query: str,
        capability: _LightRAGAuthorizationCapability,
        limit: int,
    ) -> list[GraphPath]:
        capability.scope.revalidate(capability.access_filter)
        scope_ids = tuple(
            _provenance_for(document).source_document_id
            for document in capability.scope.revisions
        )
        graph_filter = GraphAccessFilter(
            products=capability.access_filter.products,
            regions=capability.access_filter.regions,
            tenant_ids=capability.access_filter.tenant_ids,
            classifications=capability.access_filter.classifications,
            identifier_entitlements=capability.access_filter.identifier_entitlements,
            seat_tiers=capability.access_filter.seat_tiers,
            groups=capability.access_filter.groups,
            agent_user_id=capability.access_filter.agent_user_id,
            as_of=capability.access_filter.as_of,
            authorized_document_ids=scope_ids,
            authorized_revision_keys=_scope_revision_keys(capability.scope),
        )
        paths: list[GraphPath] = []
        metric_names = {
            document.metric_name
            for document in capability.scope.revisions
            if document.metric_name is not None
        }
        for metric_name in sorted(metric_names):
            paths.extend(
                self.graph_store.traverse(
                    query,
                    graph_filter,
                    limit=limit,
                    metric_name=metric_name,
                )
            )
        return paths

    def _retrieve_entities(
        self,
        query: str,
        capability: _LightRAGAuthorizationCapability,
        limit: int,
    ) -> list[LightRAGEntityRecord]:
        records: list[LightRAGEntityRecord] = []
        seen: set[str] = set()
        for path in self._graph_paths(query, capability, limit):
            node = path.nodes[-1]
            reference = _graph_reference(node, capability.scope, "entity")
            if reference is not None and reference.reference_id not in seen:
                seen.add(reference.reference_id)
                records.append(
                    LightRAGEntityRecord(
                        reference=reference,
                        name=node.label,
                        description=" ".join(candidate.label for candidate in path.nodes),
                    )
                )
        return records[:_bounded_limit(limit)]

    def _retrieve_relations(
        self,
        query: str,
        capability: _LightRAGAuthorizationCapability,
        limit: int,
    ) -> list[LightRAGRelationRecord]:
        records: list[LightRAGRelationRecord] = []
        seen: set[str] = set()
        for path in self._graph_paths(query, capability, limit):
            if len(path.nodes) < 2:
                continue
            source = _graph_reference(path.nodes[-2], capability.scope, "entity")
            target = _graph_reference(path.nodes[-1], capability.scope, "entity")
            relation = _graph_reference(path.nodes[-1], capability.scope, "relation")
            if source is None or target is None or relation is None:
                continue
            relation = relation.model_copy(update={"related_entity_references": [source, target]})
            if relation.reference_id not in seen:
                seen.add(relation.reference_id)
                records.append(
                    LightRAGRelationRecord(
                        reference=relation,
                        source_entity=source,
                        target_entity=target,
                        description=" ".join(node.label for node in path.nodes),
                    )
                )
        return records[:_bounded_limit(limit)]

    def _record_call(
        self,
        kind: Literal["chunk_vector", "entity_graph", "relation_graph"],
        query: str,
        authorized_records: Iterable[_ReferenceRecord],
        returned_records: Iterable[_ReferenceRecord],
    ) -> None:
        self.calls.append(
            LightRAGStoreCall(
                kind=kind,
                query=query,
                authorized_reference_ids=frozenset(
                    record.reference.reference_id for record in authorized_records
                ),
                returned_reference_ids=frozenset(
                    record.reference.reference_id for record in returned_records
                ),
            )
        )


class AuthorizedLightRAGIndex:
    """Read-only LightRAG view that authorizes each operation before lookup."""

    def __init__(
        self,
        store: LightRAGRetrievalStore,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> None:
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before indexing."
            )
        if not _is_supported_store(store):
            raise LightRAGAuthorizationError(
                "LightRAG index cannot prove pre-retrieval filtering for this store."
            )
        if not isinstance(access_filter, EvidenceAccessFilter):
            raise LightRAGAuthorizationError(
                "LightRAG requires the authenticated current Evidence access filter."
            )
        self._store = store
        self.scope = authorized_scope
        self.access_filter = access_filter

    def retrieve_chunks(self, query: str, *, limit: int) -> list[LightRAGChunkRecord]:
        self.scope.revalidate(self.access_filter)
        return self._authorized_view().retrieve_chunk_vectors(
            query, limit=_bounded_limit(limit)
        )

    def retrieve_entities(self, query: str, *, limit: int) -> list[LightRAGEntityRecord]:
        self.scope.revalidate(self.access_filter)
        return self._authorized_view().retrieve_entity_graph(
            query, limit=_bounded_limit(limit)
        )

    def retrieve_relations(self, query: str, *, limit: int) -> list[LightRAGRelationRecord]:
        self.scope.revalidate(self.access_filter)
        return self._authorized_view().retrieve_relation_graph(
            query, limit=_bounded_limit(limit)
        )

    def _authorized_view(self) -> LightRAGAuthorizedView:
        capability = self._store.issue_capability(self.scope, self.access_filter)
        view = self._store.authorized_view(capability)
        if not isinstance(view, LightRAGAuthorizedView) or not view.proves(capability):
            raise LightRAGAuthorizationError(
                "LightRAG authorized view cannot prove backend-created enforcement."
            )
        return view


class LightRAGBackend:
    """Closed LightRAG backend whose only path uses the governed retrieval view."""

    __slots__ = (
        "_store",
        "_last_query",
        "_last_scope",
        "_last_candidate_references",
        "_last_chain",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("LightRAGBackend retrieve entrypoint cannot be bypassed by subclassing.")

    def __init__(self, store: LightRAGRetrievalStore) -> None:
        if not _is_supported_store(store):
            raise LightRAGAuthorizationError(
                "LightRAG backend cannot prove graph/vector retrieval enforcement."
            )
        self._store = store

    @final
    def retrieve(
        self,
        query: str,
        *,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        limit: int,
    ) -> Iterable[LightRAGEvidenceReference]:
        return self.retrieve_chain(
            query,
            authorized_scope=authorized_scope,
            access_filter=access_filter,
            limit=limit,
        ).references

    @final
    def retrieve_chain(
        self,
        query: str,
        *,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        limit: int,
    ) -> LightRAGEvidenceChain:
        """Retrieve typed chunks and graph records without generating an answer."""
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before retrieval."
            )
        if not isinstance(access_filter, EvidenceAccessFilter):
            raise LightRAGAuthorizationError(
                "LightRAG requires the authenticated current Evidence access filter."
            )
        if limit < 1:
            raise ValueError("LightRAG result limit must be positive.")
        if not authorized_scope.revisions:
            return LightRAGEvidenceChain(
                supporting_chunks=[], entities=[], relations=[], references=[]
            )

        index = AuthorizedLightRAGIndex(self._store, authorized_scope, access_filter)
        self._last_query = query
        self._last_scope = authorized_scope
        chunks = index.retrieve_chunks(query, limit=1)
        entities = index.retrieve_entities(query, limit=1)
        relations = index.retrieve_relations(query, limit=1)
        records = [*chunks, *entities, *relations]
        if any(
            record.reference.related_entity_references
            != [record.source_entity, record.target_entity]
            for record in records
            if isinstance(record, LightRAGRelationRecord)
        ):
            raise LightRAGAuthorizationError(
                "LightRAG returned a relation with unverifiable graph endpoint provenance."
            )
        result_limit = _bounded_limit(limit)
        ranked_records = []
        for rank, record in enumerate(records[:result_limit], start=1):
            ranked_reference = record.reference.model_copy(update={"rank": rank})
            ranked_records.append(record.model_copy(update={"reference": ranked_reference}))
        chain = LightRAGEvidenceChain(
            supporting_chunks=[
                cast(LightRAGChunkRecord, record)
                for record in ranked_records
                if isinstance(record, LightRAGChunkRecord)
            ],
            entities=[
                cast(LightRAGEntityRecord, record)
                for record in ranked_records
                if isinstance(record, LightRAGEntityRecord)
            ],
            relations=[
                cast(LightRAGRelationRecord, record)
                for record in ranked_records
                if isinstance(record, LightRAGRelationRecord)
            ],
            references=[record.reference for record in ranked_records],
        )
        self._last_chain = chain.model_copy(deep=True)
        self._last_candidate_references = tuple(
            reference.model_copy(deep=True) for reference in chain.references
        )
        return chain

    @property
    def last_scope(self) -> AuthorizedEvidenceRevisionSet | None:
        return getattr(self, "_last_scope", None)

    @property
    def last_query(self) -> str | None:
        return getattr(self, "_last_query", None)

    @property
    def last_candidate_references(self) -> tuple[LightRAGEvidenceReference, ...]:
        return getattr(self, "_last_candidate_references", ())

    @property
    def last_chain(self) -> LightRAGEvidenceChain | None:
        return getattr(self, "_last_chain", None)


class LightRAGEvidenceAdapter:
    """Call a controlled pre-authorized LightRAG backend and expose typed evidence only."""

    __slots__ = ("_backend", "_max_results")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("LightRAG evidence adapter cannot be bypassed by subclassing.")

    def __init__(
        self,
        backend: object,
        *,
        max_results: int = _MAX_LIGHTRAG_RESULTS,
    ) -> None:
        if max_results < 1:
            raise ValueError("LightRAG max_results must be positive.")
        if not isinstance(backend, LightRAGBackend):
            raise LightRAGAuthorizationError(
                "LightRAG backend cannot prove pre-retrieval authorization enforcement."
            )
        self._backend = backend
        self._max_results = min(max_results, _MAX_LIGHTRAG_RESULTS)

    def retrieve(
        self,
        query: str,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        *,
        limit: int = _MAX_LIGHTRAG_RESULTS,
    ) -> list[LightRAGEvidenceReference]:
        return self.retrieve_chain(
            query,
            authorized_scope,
            access_filter,
            limit=limit,
        ).references

    def retrieve_chain(
        self,
        query: str,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        *,
        limit: int = _MAX_LIGHTRAG_RESULTS,
    ) -> LightRAGEvidenceChain:
        """Retrieve bounded references only after current authorization is revalidated."""
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before retrieval."
            )
        if not isinstance(access_filter, EvidenceAccessFilter):
            raise LightRAGAuthorizationError(
                "LightRAG requires the authenticated current Evidence access filter."
            )
        if limit < 1:
            raise ValueError("LightRAG result limit must be positive.")
        if not authorized_scope.revisions:
            return LightRAGEvidenceChain(
                supporting_chunks=[], entities=[], relations=[], references=[]
            )
        authorized_scope.revalidate(access_filter)
        result_limit = min(limit, self._max_results)
        try:
            chain = self._backend.retrieve_chain(
                query,
                authorized_scope=authorized_scope,
                access_filter=access_filter,
                limit=result_limit,
            )
        except LightRAGAuthorizationError:
            raise
        except Exception as error:
            raise LightRAGRetrievalError(
                "LightRAG retrieval failed closed before producing model context."
            ) from error

        return validate_authorized_lightrag_chain(chain, authorized_scope, access_filter)


def require_governed_lightrag_adapter(adapter: object) -> LightRAGEvidenceAdapter:
    """Admit only the sealed adapter bound to a concrete governed backend."""
    if type(adapter) is not LightRAGEvidenceAdapter:
        raise LightRAGAuthorizationError(
            "LightRAG requires the concrete governed evidence adapter."
        )
    concrete_adapter = cast(LightRAGEvidenceAdapter, adapter)
    backend = concrete_adapter._backend
    if type(backend) is not LightRAGBackend or not _is_supported_store(backend._store):
        raise LightRAGAuthorizationError(
            "LightRAG adapter cannot prove backend-enforced retrieval."
        )
    return concrete_adapter


def require_bound_qdrant_age_stores(
    adapter: object,
    evidence_store: object,
    graph_store: object,
) -> QdrantAGELightRAGStore:
    """Require direct-identifier stores to be the adapter's exact governed backends."""
    concrete_adapter = require_governed_lightrag_adapter(adapter)
    backend_store = concrete_adapter._backend._store
    if (
        type(backend_store) is not QdrantAGELightRAGStore
        or backend_store.qdrant_store is not evidence_store
        or backend_store.graph_store is not graph_store
    ):
        raise LightRAGAuthorizationError(
            "Direct-identifier retrieval requires the adapter-bound Qdrant and AGE stores."
        )
    return backend_store


def validate_authorized_lightrag_references(
    references: Iterable[object],
    authorized_scope: AuthorizedEvidenceRevisionSet,
    access_filter: EvidenceAccessFilter,
) -> list[LightRAGEvidenceReference]:
    """Revalidate and prove every adapter reference belongs to the exact scope."""
    authorized_scope.revalidate(access_filter)
    try:
        candidate_references = list(references)
    except Exception as error:
        raise LightRAGAuthorizationError(
            "LightRAG returned an unreadable reference set for model context."
        ) from error
    validated: list[LightRAGEvidenceReference] = []
    for reference in candidate_references:
        if not isinstance(reference, LightRAGEvidenceReference):
            raise LightRAGAuthorizationError(
                "LightRAG returned a non-reference value for model context."
            )
        if not authorized_scope.allows_reference(reference):
            raise LightRAGAuthorizationError(
                "LightRAG returned a reference outside the authorized Evidence Revision set."
            )
        validated.append(reference.model_copy(deep=True))
    return validated


def validate_authorized_lightrag_chain(
    chain: object,
    authorized_scope: AuthorizedEvidenceRevisionSet,
    access_filter: EvidenceAccessFilter,
) -> LightRAGEvidenceChain:
    """Independently validate every chain record before it reaches a response."""
    if not isinstance(chain, LightRAGEvidenceChain):
        raise LightRAGAuthorizationError("LightRAG returned an unreadable evidence chain.")
    validated_references = validate_authorized_lightrag_references(
        chain.references, authorized_scope, access_filter
    )
    if [reference.rank for reference in validated_references] != list(
        range(1, len(validated_references) + 1)
    ):
        raise LightRAGAuthorizationError("LightRAG evidence-chain ranks are invalid.")
    records: list[_ReferenceRecord] = [
        *chain.supporting_chunks,
        *chain.entities,
        *chain.relations,
    ]
    reference_by_id = {reference.reference_id: reference for reference in validated_references}
    if len(reference_by_id) != len(validated_references):
        raise LightRAGAuthorizationError("LightRAG evidence-chain references are duplicated.")
    record_ids = {record.reference.reference_id for record in records}
    if len(records) > _MAX_LIGHTRAG_RESULTS or len(record_ids) != len(records):
        raise LightRAGAuthorizationError("LightRAG evidence-chain record bound is exceeded.")
    record_reference_ids = {record.reference.reference_id for record in records}
    if record_reference_ids != set(reference_by_id):
        raise LightRAGAuthorizationError(
            "LightRAG evidence-chain records and references do not match."
        )
    for record in records:
        reference = record.reference
        if reference.reference_id not in reference_by_id:
            raise LightRAGAuthorizationError(
                "LightRAG evidence-chain record is not represented by a validated reference."
            )
        if reference_by_id[reference.reference_id] != reference:
            raise LightRAGAuthorizationError(
                "LightRAG evidence-chain record provenance differs from its reference."
            )
        if isinstance(record, LightRAGChunkRecord):
            expected_document = next(
                (
                    document
                    for document in authorized_scope.revisions
                    if authorized_scope.allows_reference(
                        reference.model_copy(update={"rank": None})
                    )
                    and _revision_key(document)
                    == (reference.source_document_id, reference.source_revision, reference.chunk_id)
                ),
                None,
            )
            if expected_document is None or record.text != expected_document.text:
                raise LightRAGAuthorizationError(
                    "LightRAG evidence-chain chunk content is not from the authorized revision."
                )
        expected_kind = (
            "chunk"
            if isinstance(record, LightRAGChunkRecord)
            else "entity"
            if isinstance(record, LightRAGEntityRecord)
            else "relation"
        )
        if reference.reference_kind != expected_kind:
            raise LightRAGAuthorizationError("LightRAG evidence-chain reference kind is invalid.")
    for relation in chain.relations:
        if relation.reference.related_entity_references != [
            relation.source_entity,
            relation.target_entity,
        ] or any(
            not _is_authorized_entity(entity, authorized_scope)
            for entity in [relation.source_entity, relation.target_entity]
        ):
            raise LightRAGAuthorizationError(
                "LightRAG evidence-chain relation endpoint provenance is invalid."
            )
    return chain.model_copy(deep=True)


def _is_authorized_entity(
    reference: LightRAGEvidenceReference,
    authorized_scope: AuthorizedEvidenceRevisionSet,
) -> bool:
    return (
        reference.reference_kind == "entity"
        and reference.reference_id.startswith("entity:")
        and authorized_scope.allows_reference(reference)
    )


def _is_supported_store(store: object) -> bool:
    return type(store) in {InMemoryLightRAGStore, QdrantAGELightRAGStore}


def _scope_revision_keys(
    scope: AuthorizedEvidenceRevisionSet,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            provenance.source_document_id,
            provenance.source_revision,
            provenance.chunk_id,
        )
        for document in scope.revisions
        for provenance in [_provenance_for(document)]
    )


def _revision_key(document: EvidenceDocument) -> tuple[str, str, str]:
    provenance = _provenance_for(document)
    return (
        provenance.source_document_id,
        provenance.source_revision,
        provenance.chunk_id,
    )


def _record_store_call(
    owner: object,
    kind: Literal["chunk_vector", "entity_graph", "relation_graph"],
    query: str,
    authorized_records: Iterable[_ReferenceRecord],
    returned_records: Iterable[_ReferenceRecord],
) -> None:
    recorder = getattr(owner, "_record_call", None)
    if recorder is None:
        raise LightRAGAuthorizationError("LightRAG store cannot prove retrieval auditing.")
    recorder(kind, query, authorized_records, returned_records)


def _rank_chunk_records(
    query: str,
    records: Iterable[LightRAGChunkRecord],
    limit: int,
) -> list[LightRAGChunkRecord]:
    query_vector = _vectorize(query)
    ranked = sorted(
        (
            (
                _cosine_similarity(query_vector, record.embedding or _vectorize(record.text)),
                record,
            )
            for record in records
            if _lexical_match(query, record.text)
        ),
        key=lambda item: (-item[0], item[1].reference.reference_id),
    )[:_bounded_limit(limit)]
    return [record.model_copy(deep=True) for _, record in ranked]


def _rank_graph_records(
    query: str,
    records: Iterable[ReferenceRecordT],
    text_for_record: Callable[[ReferenceRecordT], str],
    limit: int,
) -> list[ReferenceRecordT]:
    ranked = sorted(
        (
            (_lexical_score(query, text_for_record(record)), record)
            for record in records
            if _lexical_match(query, text_for_record(record))
        ),
        key=lambda item: (-item[0], item[1].reference.reference_id),
    )[:_bounded_limit(limit)]
    return [record for _, record in ranked]


def _graph_reference(
    node: GraphNode,
    scope: AuthorizedEvidenceRevisionSet,
    reference_kind: _REFERENCE_KINDS,
) -> LightRAGEvidenceReference | None:
    """Join AGE node provenance to an authorized revision; labels are never authority."""
    if (
        node.source_document_id is None
        or node.source_url is None
        or node.source_revision is None
        or node.chunk_id is None
        or node.lifecycle_state is None
        or node.policy_expires_at is None
    ):
        return None
    document = next(
        (
            candidate
            for candidate in scope.revisions
            if _provenance_for(candidate).source_document_id == node.source_document_id
            and _provenance_for(candidate).source_url == node.source_url
            and candidate.source_revision == node.source_revision
            and _provenance_for(candidate).chunk_id == node.chunk_id
            and candidate.revision_fingerprint == node.revision_fingerprint
            and candidate.source_page_id == node.source_page_id
            and candidate.metric_name == node.metric_name
            and candidate.access_groups == node.access_groups
            and candidate.direct_principal_grants == node.direct_principal_grants
            and candidate.lifecycle_state is node.lifecycle_state
            and candidate.policy_expires_at == node.policy_expires_at
        ),
        None,
    )
    if document is None:
        return None
    return LightRAGEvidenceReference.from_document(
        document,
        reference_kind=reference_kind,
        reference_id=f"{reference_kind}:{document.document_id}:{node.node_id}",
    )


def _bounded_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("LightRAG result limit must be positive.")
    return min(limit, _MAX_LIGHTRAG_RESULTS)


def _lexical_score(query: str, text: str) -> int:
    query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
    text_tokens = set(_TOKEN_PATTERN.findall(text.casefold()))
    return len(query_tokens & text_tokens)


def _lexical_match(query: str, text: str) -> bool:
    return _lexical_score(query, text) > 0


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    magnitude = sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / magnitude if magnitude else 0.0

"""Fail-closed LightRAG retrieval over an already-authorized evidence scope.

LightRAG is deliberately kept behind this small internal contract. It may
retrieve bounded evidence references, but it is not a semantic, permission, or
causal authority and it never generates answer text.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Final, Literal, Protocol, TypeVar, final

from pydantic import BaseModel, Field

from .evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    EvidenceLifecycleState,
    EvidencePrincipalGrant,
    _provenance_for,
    _vectorize,
)

_MAX_LIGHTRAG_RESULTS = 3
_REFERENCE_KINDS = Literal["chunk", "entity", "relation"]
_AUTHORIZED_SCOPE_TOKEN: Final = object()
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


class LightRAGAuthorizedView(ABC):
    """Already-authorized vector and graph operations exposed to the backend."""

    @abstractmethod
    def retrieve_chunk_vectors(self, query: str, *, limit: int) -> list[LightRAGChunkRecord]: ...

    @abstractmethod
    def retrieve_entity_graph(self, query: str, *, limit: int) -> list[LightRAGEntityRecord]: ...

    @abstractmethod
    def retrieve_relation_graph(
        self, query: str, *, limit: int
    ) -> list[LightRAGRelationRecord]: ...


class LightRAGRetrievalStore(ABC):
    """Store contract compatible with Qdrant/vector and AGE/graph adapters."""

    @abstractmethod
    def authorized_view(
        self,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> LightRAGAuthorizedView: ...


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
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
    ) -> LightRAGAuthorizedView:
        authorized_scope.revalidate(access_filter)
        return _InMemoryLightRAGAuthorizedView(
            self,
            self._authorized_records(self._chunks, authorized_scope),
            self._authorized_records(self._entities, authorized_scope),
            self._authorized_relations(self._relations, authorized_scope),
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


class _InMemoryLightRAGAuthorizedView(LightRAGAuthorizedView):
    """Private authorized view whose records cannot be queried before filtering."""

    __slots__ = ("_store", "_chunks", "_entities", "_relations")

    def __init__(
        self,
        store: InMemoryLightRAGStore,
        chunks: tuple[LightRAGChunkRecord, ...],
        entities: tuple[LightRAGEntityRecord, ...],
        relations: tuple[LightRAGRelationRecord, ...],
    ) -> None:
        self._store = store
        self._chunks = chunks
        self._entities = entities
        self._relations = relations

    def retrieve_chunk_vectors(self, query: str, *, limit: int) -> list[LightRAGChunkRecord]:
        query_vector = _vectorize(query)
        ranked = sorted(
            (
                (
                    _cosine_similarity(
                        query_vector,
                        record.embedding or _vectorize(record.text),
                    ),
                    record,
                )
                for record in self._chunks
                if _lexical_match(query, record.text)
            ),
            key=lambda item: (-item[0], item[1].reference.reference_id),
        )[:_bounded_limit(limit)]
        result = [record.model_copy(deep=True) for _, record in ranked]
        self._store._record_call("chunk_vector", query, self._chunks, result)
        return result

    def retrieve_entity_graph(self, query: str, *, limit: int) -> list[LightRAGEntityRecord]:
        ranked = self._rank_graph_records(query, self._entities, lambda record: record.name, limit)
        self._store._record_call("entity_graph", query, self._entities, ranked)
        return ranked

    def retrieve_relation_graph(self, query: str, *, limit: int) -> list[LightRAGRelationRecord]:
        ranked = self._rank_graph_records(
            query,
            self._relations,
            lambda record: (
                f"{record.source_entity.reference_id} "
                f"{record.target_entity.reference_id} {record.description}"
            ),
            limit,
        )
        self._store._record_call("relation_graph", query, self._relations, ranked)
        return ranked

    @staticmethod
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


@dataclass(frozen=True, init=False)
class AuthorizedEvidenceRevisionSet:
    """Opaque active Evidence Revisions authorized before LightRAG retrieval."""

    _revisions: tuple[EvidenceDocument, ...]

    def __init__(
        self,
        revisions: Iterable[EvidenceDocument],
        *,
        _token: object | None = None,
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

    @property
    def revisions(self) -> tuple[EvidenceDocument, ...]:
        """Return defensive copies so callers cannot mutate the authorization proof."""
        return tuple(document.model_copy(deep=True) for document in self._revisions)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[EvidenceDocument],
        access_filter: EvidenceAccessFilter,
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
        return cls(revisions, _token=_AUTHORIZED_SCOPE_TOKEN)

    def revalidate(self, access_filter: EvidenceAccessFilter) -> None:
        """Recheck current lifecycle and policy state immediately before retrieval."""
        if not isinstance(access_filter, EvidenceAccessFilter):
            raise LightRAGAuthorizationError(
                "LightRAG requires the authenticated current Evidence access filter."
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
        if (
            reference.lifecycle_state is not EvidenceLifecycleState.ACTIVE
            or not reference.reference_id.startswith(f"{reference.reference_kind}:")
        ):
            return False
        for document in self._revisions:
            provenance = _provenance_for(document)
            if (
                provenance.source_document_id == reference.source_document_id
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
        if not isinstance(store, LightRAGRetrievalStore):
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
        view = self._store.authorized_view(self.scope, self.access_filter)
        if not isinstance(view, LightRAGAuthorizedView):
            raise LightRAGAuthorizationError(
                "LightRAG authorized view cannot prove all retrieval operations."
            )
        return view


class LightRAGBackend:
    """Closed LightRAG backend whose only path uses the governed retrieval view."""

    __slots__ = ("_store", "_last_query", "_last_scope", "_last_candidate_references")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("LightRAGBackend retrieve entrypoint cannot be bypassed by subclassing.")

    def __init__(self, store: LightRAGRetrievalStore) -> None:
        if not isinstance(store, LightRAGRetrievalStore):
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
            return []

        index = AuthorizedLightRAGIndex(self._store, authorized_scope, access_filter)
        self._last_query = query
        self._last_scope = authorized_scope
        records = [
            *index.retrieve_chunks(query, limit=1),
            *index.retrieve_entities(query, limit=1),
            *index.retrieve_relations(query, limit=1),
        ]
        if any(
            record.reference.related_entity_references
            != [record.source_entity, record.target_entity]
            for record in records
            if isinstance(record, LightRAGRelationRecord)
        ):
            raise LightRAGAuthorizationError(
                "LightRAG returned a relation with unverifiable graph endpoint provenance."
            )
        references = [record.reference for record in records]
        self._last_candidate_references = tuple(
            reference.model_copy(deep=True) for reference in references[:_bounded_limit(limit)]
        )
        return [reference.model_copy(deep=True) for reference in self._last_candidate_references]

    @property
    def last_scope(self) -> AuthorizedEvidenceRevisionSet | None:
        return getattr(self, "_last_scope", None)

    @property
    def last_query(self) -> str | None:
        return getattr(self, "_last_query", None)

    @property
    def last_candidate_references(self) -> tuple[LightRAGEvidenceReference, ...]:
        return getattr(self, "_last_candidate_references", ())


class LightRAGEvidenceAdapter:
    """Call a controlled pre-authorized LightRAG backend and expose references only."""

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
            return []
        authorized_scope.revalidate(access_filter)
        result_limit = min(limit, self._max_results)
        try:
            raw_references = list(
                self._backend.retrieve(
                    query,
                    authorized_scope=authorized_scope,
                    access_filter=access_filter,
                    limit=result_limit,
                )
            )
        except LightRAGAuthorizationError:
            raise
        except Exception as error:
            raise LightRAGRetrievalError(
                "LightRAG retrieval failed closed before producing model context."
            ) from error

        references = raw_references[:result_limit]
        for reference in references:
            if not isinstance(reference, LightRAGEvidenceReference):
                raise LightRAGAuthorizationError(
                    "LightRAG returned a non-reference value for model context."
                )
            if not authorized_scope.allows_reference(reference):
                raise LightRAGAuthorizationError(
                    "LightRAG returned a reference outside the authorized Evidence Revision set."
                )
        return [reference.model_copy(deep=True) for reference in references]


def _is_authorized_entity(
    reference: LightRAGEvidenceReference,
    authorized_scope: AuthorizedEvidenceRevisionSet,
) -> bool:
    return (
        reference.reference_kind == "entity"
        and reference.reference_id.startswith("entity:")
        and authorized_scope.allows_reference(reference)
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

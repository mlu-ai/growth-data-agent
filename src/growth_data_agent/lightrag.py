"""Fail-closed LightRAG retrieval over an already-authorized evidence scope.

LightRAG is deliberately kept behind this small internal contract. It may
retrieve bounded evidence references, but it is not a semantic, permission, or
causal authority and it never generates answer text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, final

from pydantic import BaseModel, Field

from .evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    EvidenceLifecycleState,
    EvidencePrincipalGrant,
    _provenance_for,
)

_MAX_LIGHTRAG_RESULTS = 3
_REFERENCE_KINDS = Literal["chunk", "entity", "relation"]
_AUTHORIZED_SCOPE_TOKEN: Final = object()


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

    @property
    def revision_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (_provenance_for(document).source_document_id, document.source_revision)
            for document in self._revisions
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
            ):
                return True
        return False


class AuthorizedLightRAGIndex:
    """Read-only LightRAG index view containing only one authorized revision set."""

    def __init__(
        self,
        references: Iterable[LightRAGEvidenceReference],
        authorized_scope: AuthorizedEvidenceRevisionSet,
    ) -> None:
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before indexing."
            )
        self.scope = authorized_scope
        self._references = tuple(
            reference.model_copy(deep=True)
            for reference in references
            if (
                (reference.source_document_id, reference.source_revision)
                in authorized_scope.revision_keys
                and authorized_scope.allows_reference(reference)
            )
        )

    @property
    def references(self) -> tuple[LightRAGEvidenceReference, ...]:
        """Return defensive copies of the pre-authorized reference candidates."""
        return tuple(reference.model_copy(deep=True) for reference in self._references)

    def retrieve(
        self,
        *,
        reference_kind: _REFERENCE_KINDS | None = None,
        limit: int,
    ) -> list[LightRAGEvidenceReference]:
        """Return only candidates selected after source-revision authorization."""
        if limit < 1:
            raise ValueError("LightRAG result limit must be positive.")
        result_limit = min(limit, _MAX_LIGHTRAG_RESULTS)
        references = (
            self._references
            if reference_kind is None
            else tuple(
                reference
                for reference in self._references
                if reference.reference_kind == reference_kind
            )
        )
        return [reference.model_copy(deep=True) for reference in references[:result_limit]]


class LightRAGBackend:
    """Closed LightRAG index whose only retrieval path creates a scoped view."""

    __slots__ = ("_references", "_last_query", "_last_scope", "_last_candidate_references")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("LightRAGBackend retrieve entrypoint cannot be bypassed by subclassing.")

    def __init__(self, references: Iterable[LightRAGEvidenceReference]) -> None:
        self._references = tuple(reference.model_copy(deep=True) for reference in references)

    @final
    def retrieve(
        self,
        query: str,
        *,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        limit: int,
    ) -> Iterable[LightRAGEvidenceReference]:
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before retrieval."
            )
        if not authorized_scope.revisions:
            return []
        authorized_index = AuthorizedLightRAGIndex(self._references, authorized_scope)
        self._last_query = query
        self._last_scope = authorized_scope
        self._last_candidate_references = authorized_index.references
        return authorized_index.retrieve(limit=limit)

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
    """Call a controlled pre-authorized LightRAG backend and expose safe references only."""

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
        *,
        limit: int = _MAX_LIGHTRAG_RESULTS,
    ) -> list[LightRAGEvidenceReference]:
        """Retrieve bounded references only after an active authorized scope exists."""
        if not isinstance(authorized_scope, AuthorizedEvidenceRevisionSet):
            raise LightRAGAuthorizationError(
                "LightRAG requires an authorized Evidence Revision set before retrieval."
            )
        if limit < 1:
            raise ValueError("LightRAG result limit must be positive.")
        if not authorized_scope.revisions:
            return []
        result_limit = min(limit, self._max_results)
        try:
            raw_references = list(
                self._backend.retrieve(
                    query,
                    authorized_scope=authorized_scope,
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
        return references

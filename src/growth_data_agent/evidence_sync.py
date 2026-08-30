"""Governed Confluence evidence normalization and synchronization contracts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from hashlib import sha256
from typing import Protocol

from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import QdrantClient, models

from .contracts import EvidenceSupportStatus
from .evidence import (
    EmbeddingProvider,
    EvidenceDocument,
    EvidenceLifecycleState,
    EvidencePrincipalGrant,
    HashEmbeddingProvider,
    _evidence_node,
)


class EvidenceRevisionValidationError(ValueError):
    """Raised when a source revision cannot safely enter the evidence index."""


class QdrantConfigurationError(RuntimeError):
    """Raised when the production Qdrant connection is not configured."""


class SourceAccessMetadata(BaseModel):
    """Source policy metadata required before evidence can be indexed."""

    classification: str = Field(min_length=1)
    identifier_entitlement: str = Field(min_length=1)
    access_groups: list[str] = Field(default_factory=list)
    direct_principal_grants: list[EvidencePrincipalGrant] = Field(default_factory=list)
    policy_expires_at: datetime


class ConfluenceEvidenceChunk(BaseModel):
    """One source chunk with stable provenance within its source page revision."""

    chunk_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class ConfluenceEvidenceRevision(BaseModel):
    """Normalized source record shared by live-shaped and synthetic adapters."""

    source_page_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    lifecycle_state: EvidenceLifecycleState = EvidenceLifecycleState.ACTIVE
    metric_name: str | None = None
    title: str = Field(min_length=1)
    product: str = Field(min_length=1)
    region: str = Field(min_length=1)
    tenant_ids: list[str] = Field(min_length=1)
    tenant_scope: str = Field(min_length=1)
    relevant_date: date
    freshness: datetime
    support_status: EvidenceSupportStatus
    support_explanation: str = Field(min_length=1)
    chunks: list[ConfluenceEvidenceChunk] = Field(default_factory=list)
    source_access: SourceAccessMetadata
    embedding_model: str = Field(min_length=1)
    embedding_version: str = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def require_http_source_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise EvidenceRevisionValidationError("source_url must be an HTTP(S) URL.")
        return value

    @model_validator(mode="after")
    def validate_lifecycle_content(self) -> ConfluenceEvidenceRevision:
        if self.lifecycle_state is EvidenceLifecycleState.ACTIVE and not self.chunks:
            raise EvidenceRevisionValidationError(
                "Active evidence revisions require at least one content chunk."
            )
        if self.lifecycle_state is not EvidenceLifecycleState.ACTIVE and self.chunks:
            raise EvidenceRevisionValidationError(
                "Deleted or inaccessible revisions cannot carry retrievable content."
            )
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise EvidenceRevisionValidationError("Chunk provenance IDs must be unique.")
        return self


class ConfluenceEvidenceSource(Protocol):
    def iter_revisions(self) -> Iterable[ConfluenceEvidenceRevision]: ...


class EvidenceQdrantError(RuntimeError):
    """Raised when Qdrant cannot complete a synchronization operation."""


class EvidenceEmbeddingError(RuntimeError):
    """Raised when the configured embedding provider cannot embed content."""


class EvidenceSyncResult(BaseModel):
    """Counts from one source synchronization pass."""

    indexed_revision_count: int = 0
    skipped_revision_count: int = 0
    removed_revision_count: int = 0


def qdrant_client_from_environment() -> QdrantClient:
    """Build the external Qdrant client required by operator synchronization."""
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise QdrantConfigurationError("QDRANT_URL is required for external Qdrant.")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))


def qdrant_collection_from_environment() -> str:
    return os.environ.get("QDRANT_COLLECTION", "growth_evidence")


class ConfluenceEvidenceSourceAdapter:
    """Normalize raw live-shaped source records at the source boundary."""

    def __init__(
        self,
        fetch_revisions: Callable[
            [], Iterable[ConfluenceEvidenceRevision | Mapping[str, object]]
        ],
    ) -> None:
        self._fetch_revisions = fetch_revisions

    def iter_revisions(self) -> Iterable[ConfluenceEvidenceRevision]:
        for raw_revision in self._fetch_revisions():
            yield ConfluenceEvidenceRevision.model_validate(raw_revision)


class QdrantEvidenceSynchronizer:
    """Synchronize normalized Confluence revisions into an external Qdrant collection."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str = "growth_evidence",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider or HashEmbeddingProvider()
        self._vector_store = None

    def sync(self, source: ConfluenceEvidenceSource) -> EvidenceSyncResult:
        """Apply the latest source revision for each page without indexing invalid content."""
        indexed = 0
        skipped = 0
        removed = 0
        revisions = []
        for raw_revision in source.iter_revisions():
            revision = ConfluenceEvidenceRevision.model_validate(raw_revision)
            self._validate_embedding_metadata(revision)
            revisions.append(revision)
        for revision in revisions:
            existing = self._existing_page_points(revision.source_page_id)
            if self._is_unchanged(revision, existing):
                skipped += 1
                continue

            staging_lifecycle_state = (
                EvidenceLifecycleState.INACCESSIBLE
                if revision.lifecycle_state is EvidenceLifecycleState.ACTIVE
                else revision.lifecycle_state
            )
            nodes = self._nodes_for_revision(
                revision,
                lifecycle_state=staging_lifecycle_state,
            )
            new_point_ids = [node.node_id for node in nodes]
            existing_point_ids = [point.id for point in existing]
            new_point_id_set = set(new_point_ids)
            stale_existing_point_ids = [
                point_id for point_id in existing_point_ids if point_id not in new_point_id_set
            ]
            add_attempted = False
            uploaded = False
            promotion_attempted = False
            try:
                if stale_existing_point_ids:
                    self._client.set_payload(
                        self._collection_name,
                        payload={
                            "lifecycle_state": EvidenceLifecycleState.INACCESSIBLE.value
                        },
                        points=stale_existing_point_ids,
                    )
                vector_store = self._vector_store or self._new_vector_store()
                self._vector_store = vector_store
                add_attempted = True
                vector_store.add(nodes)
                uploaded = True
                if stale_existing_point_ids:
                    self._client.delete(
                        self._collection_name,
                        models.PointIdsList(points=stale_existing_point_ids),
                    )
                if revision.lifecycle_state is EvidenceLifecycleState.ACTIVE:
                    promotion_attempted = True
                    self._client.set_payload(
                        self._collection_name,
                        payload={"lifecycle_state": EvidenceLifecycleState.ACTIVE.value},
                        points=new_point_ids,
                    )
            except Exception as error:
                if add_attempted and not uploaded:
                    cleanup_succeeded = self._rollback_new_points(new_point_ids)
                    if stale_existing_point_ids and cleanup_succeeded:
                        self._restore_points(stale_existing_point_ids)
                elif promotion_attempted:
                    try:
                        self._quarantine_points(new_point_ids)
                    except Exception as quarantine_error:
                        raise EvidenceQdrantError(
                            f"Qdrant synchronization failed and could not quarantine the "
                            f"new revision for source page {revision.source_page_id}."
                        ) from quarantine_error
                raise EvidenceQdrantError(
                    f"Qdrant synchronization failed for source page {revision.source_page_id}."
                ) from error

            if stale_existing_point_ids:
                removed += 1
            if revision.lifecycle_state is EvidenceLifecycleState.ACTIVE:
                indexed += 1
        return EvidenceSyncResult(
            indexed_revision_count=indexed,
            skipped_revision_count=skipped,
            removed_revision_count=removed,
        )

    def _validate_embedding_metadata(self, revision: ConfluenceEvidenceRevision) -> None:
        if (
            revision.embedding_model != self._embedding_provider.model_name
            or revision.embedding_version != self._embedding_provider.model_version
        ):
            raise EvidenceRevisionValidationError(
                "Revision embedding metadata does not match the configured embedding provider."
            )

    def _existing_page_points(self, source_page_id: str):
        try:
            if not self._client.collection_exists(self._collection_name):
                return []
            points, next_offset = self._client.scroll(
                self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_page_id",
                            match=models.MatchValue(value=source_page_id),
                        )
                    ]
                ),
                limit=10_000,
                with_payload=True,
            )
            all_points = list(points)
            while next_offset is not None:
                points, next_offset = self._client.scroll(
                    self._collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_page_id",
                                match=models.MatchValue(value=source_page_id),
                            )
                        ]
                    ),
                    limit=10_000,
                    offset=next_offset,
                    with_payload=True,
                )
                all_points.extend(points)
            return all_points
        except Exception as error:
            raise EvidenceQdrantError(
                f"Qdrant read failed for source page {source_page_id}."
            ) from error

    @staticmethod
    def _is_unchanged(revision: ConfluenceEvidenceRevision, existing: list[object]) -> bool:
        if not existing:
            return False
        expected_chunk_ids = (
            {chunk.chunk_id for chunk in revision.chunks}
            if revision.lifecycle_state is EvidenceLifecycleState.ACTIVE
            else {f"{revision.source_page_id}:tombstone"}
        )
        return (
            {str(point.payload.get("source_revision")) for point in existing}
            == {revision.source_revision}
            and {str(point.payload.get("lifecycle_state")) for point in existing}
            == {revision.lifecycle_state.value}
            and {str(point.payload.get("chunk_id")) for point in existing}
            == expected_chunk_ids
            and {str(point.payload.get("revision_fingerprint")) for point in existing}
            == {_revision_fingerprint(revision)}
        )

    def _nodes_for_revision(
        self,
        revision: ConfluenceEvidenceRevision,
        *,
        lifecycle_state: EvidenceLifecycleState,
    ):
        if revision.lifecycle_state is EvidenceLifecycleState.ACTIVE:
            chunks = [(chunk.chunk_id, chunk.chunk_index, chunk.text) for chunk in revision.chunks]
        else:
            chunks = [(f"{revision.source_page_id}:tombstone", 0, "")]
        revision_fingerprint = _revision_fingerprint(revision)
        nodes = []
        for chunk_id, chunk_index, text in chunks:
            document = EvidenceDocument(
                document_id=chunk_id,
                metric_name=revision.metric_name,
                title=revision.title,
                text=text,
                product=revision.product,
                region=revision.region,
                tenant_ids=revision.tenant_ids,
                tenant_scope=revision.tenant_scope,
                classification=revision.source_access.classification,
                identifier_entitlement=revision.source_access.identifier_entitlement,
                relevant_date=revision.relevant_date,
                freshness=revision.freshness,
                support_status=revision.support_status,
                support_explanation=revision.support_explanation,
                source_document_id=revision.source_page_id,
                source_url=revision.source_url,
                source_revision=revision.source_revision,
                source_page_id=revision.source_page_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                lifecycle_state=lifecycle_state,
                embedding_model=revision.embedding_model,
                embedding_version=revision.embedding_version,
                revision_fingerprint=revision_fingerprint,
                access_groups=revision.source_access.access_groups,
                direct_principal_grants=revision.source_access.direct_principal_grants,
                policy_expires_at=revision.source_access.policy_expires_at,
            )
            try:
                vector = self._embedding_provider.embed(
                    f"{revision.title} {text}" if text else revision.source_page_id
                )
            except Exception as error:
                raise EvidenceEmbeddingError(
                    f"Embedding failed for source page {revision.source_page_id}."
                ) from error
            nodes.append(
                _evidence_node(
                    document,
                    embedding=vector,
                    node_identity=f"{revision.source_page_id}:{revision_fingerprint}:{chunk_id}",
                )
            )
        return nodes

    def _rollback_new_points(self, point_ids: list[str]) -> bool:
        """Best-effort cleanup that keeps the previous revision retrievable on failure."""
        if not point_ids:
            return True
        try:
            if self._client.collection_exists(self._collection_name):
                self._client.delete(
                    self._collection_name,
                    models.PointIdsList(points=point_ids),
                )
            return True
        except Exception:
            return False

    def _restore_points(self, point_ids: list[str]) -> None:
        """Restore the previous revision only when its replacement was not uploaded."""
        try:
            self._client.set_payload(
                self._collection_name,
                payload={"lifecycle_state": EvidenceLifecycleState.ACTIVE.value},
                points=point_ids,
            )
        except Exception:
            # Leave the old revision quarantined rather than risk claiming it is
            # retrievable after an incomplete replacement.
            return

    def _quarantine_points(self, point_ids: list[str]) -> None:
        """Keep partially promoted points out of retrieval after a failed promotion."""
        self._client.set_payload(
            self._collection_name,
            payload={"lifecycle_state": EvidenceLifecycleState.INACCESSIBLE.value},
            points=point_ids,
        )

    def _new_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            dense_config=models.VectorParams(size=32, distance=models.Distance.COSINE),
        )

    def readiness(self) -> dict[str, object]:
        """Report external Qdrant and embedding readiness without credentials."""
        embedding = self._embedding_provider.readiness()
        try:
            self._client.get_collections()
        except Exception:
            qdrant = {
                "status": "unavailable",
                "collection": self._collection_name,
                "external": True,
            }
        else:
            qdrant = {
                "status": "ready",
                "collection": self._collection_name,
                "external": True,
            }
        return {
            "status": (
                "ready"
                if qdrant["status"] == "ready" and embedding["status"] == "ready"
                else "unavailable"
            ),
            "qdrant": qdrant,
            "embedding": embedding,
        }


def _revision_fingerprint(revision: ConfluenceEvidenceRevision) -> str:
    """Hash non-content metadata so policy changes are never skipped."""
    canonical_revision = revision.model_dump(mode="json", exclude={"chunks"})
    return sha256(
        json.dumps(canonical_revision, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

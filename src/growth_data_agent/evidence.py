"""Entitlement-filtered evidence retrieval over a Qdrant vector store."""

from __future__ import annotations

import re
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from math import sqrt
from typing import Protocol

from llama_index.core.schema import TextNode
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from .contracts import (
    EvidenceAnswer,
    EvidenceCitation,
    EvidenceScope,
    EvidenceSupportStatus,
)

_VECTOR_SIZE = 32
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_IDENTIFIER_PATTERN = re.compile(r"\b(?:tenant|person|product-user)-\d+\b", re.IGNORECASE)


class EvidencePrincipalGrant(BaseModel):
    """An opaque, expiring direct-principal grant held in source policy metadata."""

    principal_id: str
    expires_at: datetime


class EvidenceDocument(BaseModel):
    """A document and the metadata needed for pre-retrieval authorization."""

    document_id: str
    metric_name: str | None = None
    title: str
    text: str
    product: str
    region: str
    tenant_ids: list[str]
    tenant_scope: str
    classification: str
    identifier_entitlement: str
    relevant_date: date
    freshness: datetime
    support_status: EvidenceSupportStatus
    support_explanation: str
    sensitive_identifiers: list[str] = Field(default_factory=list)
    accountable_team: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_revision: str = "synthetic-v1"
    access_groups: list[str] = Field(default_factory=lambda: ["evidence-general"])
    direct_principal_grants: list[EvidencePrincipalGrant] = Field(default_factory=list)
    policy_expires_at: datetime = Field(
        default_factory=lambda: datetime(2099, 12, 31, tzinfo=UTC)
    )


@dataclass(frozen=True)
class EvidenceAccessFilter:
    """All document constraints derived from an Access Profile and known driver."""

    products: tuple[str, ...]
    regions: tuple[str, ...]
    tenant_ids: tuple[str, ...]
    classifications: tuple[str, ...]
    identifier_entitlements: tuple[str, ...]
    excluded_tenant_ids: tuple[str, ...] = ()
    seat_tiers: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    agent_user_id: str | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))

    def allows(self, document: EvidenceDocument) -> bool:
        """Apply the same policy at the context boundary as a defensive second layer."""
        within_structured_scope = (
            document.product in self.products
            and document.region in self.regions
            and bool(document.tenant_ids)
            and set(document.tenant_ids).issubset(self.tenant_ids)
            and document.classification in self.classifications
            and document.identifier_entitlement in self.identifier_entitlements
            and (not self.metric_names or document.metric_name in self.metric_names)
        )
        group_permitted = not self.groups or not document.access_groups or bool(
            set(document.access_groups).intersection(self.groups)
        )
        direct_grant_permitted = not document.direct_principal_grants or any(
            grant.principal_id == self.agent_user_id and grant.expires_at > self.as_of
            for grant in document.direct_principal_grants
        )
        return (
            within_structured_scope
            and group_permitted
            and direct_grant_permitted
            and document.policy_expires_at > self.as_of
        )

    def as_qdrant_filter(self) -> models.Filter:
        must_not = []
        if self.excluded_tenant_ids:
            must_not.append(
                models.FieldCondition(
                    key="tenant_ids",
                    match=models.MatchAny(any=list(self.excluded_tenant_ids)),
                )
            )
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="product",
                    match=models.MatchAny(any=list(self.products)),
                ),
                models.FieldCondition(
                    key="policy_expires_at",
                    range=models.DatetimeRange(gt=self.as_of),
                ),
                models.FieldCondition(
                    key="region",
                    match=models.MatchAny(any=list(self.regions)),
                ),
                models.FieldCondition(
                    key="tenant_ids",
                    match=models.MatchAny(any=list(self.tenant_ids)),
                ),
                models.FieldCondition(
                    key="classification",
                    match=models.MatchAny(any=list(self.classifications)),
                ),
                models.FieldCondition(
                    key="identifier_entitlement",
                    match=models.MatchAny(any=list(self.identifier_entitlements)),
                ),
                *(
                    [
                        models.FieldCondition(
                            key="metric_name",
                            match=models.MatchAny(any=list(self.metric_names)),
                        )
                    ]
                    if self.metric_names
                    else []
                ),
            ],
            must_not=must_not,
            should=[
                *(
                    [
                        models.FieldCondition(
                            key="access_groups",
                            match=models.MatchAny(any=list(self.groups)),
                        )
                    ]
                    if self.groups
                    else []
                ),
                *(
                    [
                        models.FieldCondition(
                            key="direct_principal_ids",
                            match=models.MatchValue(value=self.agent_user_id),
                        )
                    ]
                    if self.agent_user_id
                    else []
                ),
            ],
        )


class VectorEvidenceStore(Protocol):
    def retrieve(
        self,
        query: str,
        access_filter: EvidenceAccessFilter,
        *,
        limit: int,
    ) -> list[EvidenceDocument]: ...


class QdrantEvidenceStore:
    """Retrieve LlamaIndex evidence nodes from Qdrant after entitlement filtering."""

    def __init__(
        self,
        documents: Iterable[EvidenceDocument],
        *,
        client: QdrantClient | None = None,
        collection_name: str = "growth_evidence",
    ):
        self._documents = tuple(documents)
        self._documents_by_id = {document.document_id: document for document in self._documents}
        self._nodes = tuple(_evidence_node(document) for document in self._documents)
        self._client = client or QdrantClient(location=":memory:")
        self._collection_name = collection_name
        self.last_filter: EvidenceAccessFilter | None = None
        self._last_scores: ContextVar[tuple[float, ...]] = ContextVar(
            "growth_data_agent_last_retrieval_scores", default=()
        )
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=_VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    vector=_vectorize(f"{node.metadata['title']} {node.text}"),
                    payload=node.metadata,
                )
                for index, node in enumerate(self._nodes, start=1)
            ],
        )

    def retrieve(
        self,
        query: str,
        access_filter: EvidenceAccessFilter,
        *,
        limit: int,
    ) -> list[EvidenceDocument]:
        """Return only filtered documents; ranking happens after Qdrant filtering."""
        self.last_filter = access_filter
        self._last_scores.set(())
        if not self._documents:
            return []
        points = self._client.query_points(
            collection_name=self._collection_name,
            query=_vectorize(query),
            query_filter=access_filter.as_qdrant_filter(),
            limit=len(self._documents),
            with_payload=True,
        ).points
        candidates = [
            (
                self._documents_by_id[str(point.payload["document_id"])],
                float(point.score),
            )
            for point in points
            if point.payload is not None
        ]
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -_lexical_score(query, candidate[0]),
                candidate[0].document_id,
            ),
        )[:limit]
        self._last_scores.set(tuple(score for _, score in ranked))
        return [document for document, _ in ranked]

    @property
    def nodes(self) -> tuple[TextNode, ...]:
        """Stable LlamaIndex nodes retained for ingestion and deterministic test inspection."""
        return self._nodes

    @property
    def last_scores(self) -> tuple[float, ...]:
        """Return scores for this execution context's final document ranking."""
        return self._last_scores.get()


def build_evidence_answer(
    documents: Iterable[EvidenceDocument],
) -> EvidenceAnswer:
    """Classify retrieved evidence without turning it into a causal conclusion."""
    documents = list(documents)
    citations = [_citation(document) for document in documents]
    statuses = {citation.support_status for citation in citations}
    if not citations:
        return EvidenceAnswer(
            citations=[],
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "Insufficient in-scope evidence was retrieved to support a Hypothesis."
            ),
        )
    if EvidenceSupportStatus.CONTRADICTS in statuses:
        return EvidenceAnswer(
            citations=citations,
            support_status=EvidenceSupportStatus.INCONCLUSIVE,
            support_explanation=(
                "The retrieved evidence is contradictory, so no supported explanation is "
                "returned."
            ),
        )
    if EvidenceSupportStatus.SUPPORTS in statuses:
        supporting_document = next(
            document
            for document in documents
            if document.support_status == EvidenceSupportStatus.SUPPORTS
        )
        return EvidenceAnswer(
            citations=citations,
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                f"{_redact_identifiers(supporting_document.support_explanation)} It supports a "
                "possible Hypothesis but does not establish causation."
            ),
        )
    return EvidenceAnswer(
        citations=citations,
        support_status=EvidenceSupportStatus.INCONCLUSIVE,
        support_explanation=(
            "The retrieved in-scope material does not provide enough support for a "
            "Hypothesis."
        ),
    )


def _citation(document: EvidenceDocument) -> EvidenceCitation:
    return EvidenceCitation(
        document_id=_redact_identifiers(document.document_id),
        title=_redact_identifiers(document.title),
        affected_scope=EvidenceScope(
            product=document.product,
            region=document.region,
            tenant_scope=_redact_identifiers(document.tenant_scope),
        ),
        relevant_date=document.relevant_date,
        freshness=document.freshness.astimezone(UTC),
        support_status=document.support_status,
        support_explanation=_redact_identifiers(document.support_explanation),
        source_document_id=_redact_identifiers(
            document.source_document_id or document.document_id
        ),
        source_url=document.source_url or _synthetic_source_url(document),
        source_revision=document.source_revision,
        chunk_id=f"{document.document_id}:chunk:0",
    )


def _evidence_node(document: EvidenceDocument) -> TextNode:
    """Create one stable LlamaIndex chunk while preserving source and policy provenance."""
    source_document_id = document.source_document_id or document.document_id
    chunk_id = f"{document.document_id}:chunk:0"
    metadata = document.model_dump(mode="json")
    metadata.update(
        {
            "document_id": document.document_id,
            "source_document_id": source_document_id,
            "source_url": document.source_url or _synthetic_source_url(document),
            "source_revision": document.source_revision,
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "direct_principal_ids": [
                grant.principal_id for grant in document.direct_principal_grants
            ],
        }
    )
    return TextNode(id_=chunk_id, text=document.text, metadata=metadata)


def _synthetic_source_url(document: EvidenceDocument) -> str:
    return f"https://evidence.local/synthetic/{document.document_id}"


def _redact_identifiers(value: str) -> str:
    """Prevent raw identifier text from reaching a citation or generated response."""
    return _IDENTIFIER_PATTERN.sub("[redacted identifier]", value)


def _vectorize(value: str) -> list[float]:
    vector = [0.0] * _VECTOR_SIZE
    for token in _TOKEN_PATTERN.findall(value.casefold()):
        index = int.from_bytes(sha256(token.encode()).digest()[:2], "big") % _VECTOR_SIZE
        vector[index] += 1.0
    magnitude = sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]


def _lexical_score(query: str, document: EvidenceDocument) -> int:
    query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
    document_tokens = set(
        _TOKEN_PATTERN.findall(f"{document.title} {document.text}".casefold())
    )
    return len(query_tokens & document_tokens)

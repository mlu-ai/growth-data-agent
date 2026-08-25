"""Entitlement-filtered evidence retrieval over a Qdrant vector store."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from math import sqrt
from typing import Protocol

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


class EvidenceDocument(BaseModel):
    """A document and the metadata needed for pre-retrieval authorization."""

    document_id: str
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


@dataclass(frozen=True)
class EvidenceAccessFilter:
    """All document constraints derived from an Access Profile and known driver."""

    products: tuple[str, ...]
    regions: tuple[str, ...]
    tenant_ids: tuple[str, ...]
    classifications: tuple[str, ...]
    identifier_entitlements: tuple[str, ...]
    excluded_tenant_ids: tuple[str, ...] = ()

    def allows(self, document: EvidenceDocument) -> bool:
        """Apply the same policy at the context boundary as a defensive second layer."""
        return (
            document.product in self.products
            and document.region in self.regions
            and set(document.tenant_ids).issubset(self.tenant_ids)
            and document.classification in self.classifications
            and document.identifier_entitlement in self.identifier_entitlements
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
            ],
            must_not=must_not,
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
    """Retrieve documents from Qdrant after applying the supplied payload filter."""

    def __init__(
        self,
        documents: Iterable[EvidenceDocument],
        *,
        client: QdrantClient | None = None,
        collection_name: str = "growth_evidence",
    ):
        self._documents = tuple(documents)
        self._documents_by_id = {document.document_id: document for document in self._documents}
        self._client = client or QdrantClient(location=":memory:")
        self._collection_name = collection_name
        self.last_filter: EvidenceAccessFilter | None = None
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
                    vector=_vectorize(f"{document.title} {document.text}"),
                    payload=document.model_dump(mode="json"),
                )
                for index, document in enumerate(self._documents, start=1)
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
            self._documents_by_id[str(point.payload["document_id"])]
            for point in points
            if point.payload is not None
        ]
        return sorted(
            candidates,
            key=lambda document: (-_lexical_score(query, document), document.document_id),
        )[:limit]


def build_evidence_answer(
    documents: Iterable[EvidenceDocument],
) -> EvidenceAnswer:
    """Classify retrieved evidence without turning it into a causal conclusion."""
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
        return EvidenceAnswer(
            citations=citations,
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation=(
                "The permitted incident overlaps the affected scope and decline period; it "
                "supports a possible Hypothesis but does not establish causation."
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
    )


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

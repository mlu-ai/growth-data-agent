"""Registered, bounded tools for a governed evidence investigation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from .evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    VectorEvidenceStore,
    _evidence_revision_key,
)
from .graph import GraphAccessFilter, GraphPath
from .lightrag import (
    AuthorizedEvidenceRevisionSet,
    LightRAGAuthorizationError,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceChain,
    require_governed_lightrag_adapter,
    validate_authorized_lightrag_chain,
)
from .observability import trace_span
from .reranking import (
    EvidenceReranker,
    EvidenceRerankerUnavailableError,
    EvidenceRerankingError,
)

_MAX_EVIDENCE_TOOL_RESULTS = 3

GraphTraversalTool = Callable[[str, GraphAccessFilter, str, int], list[GraphPath]]
RevisionReader = Callable[[EvidenceAccessFilter], Iterable[EvidenceDocument]]


@dataclass(frozen=True)
class EvidenceInvestigation:
    """Only the permitted output of the registered evidence tools."""

    documents: list[EvidenceDocument]
    graph_paths: list[GraphPath]
    lightrag_chain: LightRAGEvidenceChain


class BoundedEvidenceInvestigationTools:
    """Invoke the two approved evidence tools with a non-negotiable result budget."""

    def __init__(
        self,
        evidence_store: VectorEvidenceStore,
        graph_traversal_tool: GraphTraversalTool,
        evidence_reranker: EvidenceReranker | None,
        lightrag_adapter: LightRAGEvidenceAdapter | None = None,
    ) -> None:
        self._evidence_store = evidence_store
        self._graph_traversal_tool = graph_traversal_tool
        self._evidence_reranker = evidence_reranker
        self._lightrag_adapter = lightrag_adapter

    def investigate(
        self,
        *,
        query: str,
        evidence_filter: EvidenceAccessFilter,
        graph_filter: GraphAccessFilter,
        metric_name: str,
    ) -> EvidenceInvestigation:
        """Pass policy to each tool before it retrieves candidates or graph paths."""
        if self._lightrag_adapter is None:
            raise LightRAGAuthorizationError(
                "Governed LightRAG evidence retrieval is unavailable."
            )
        lightrag_adapter = require_governed_lightrag_adapter(self._lightrag_adapter)

        authorized_document_ids: set[str] | None = None
        authorized_revision_keys: set[tuple[str, str, str]] | None = None
        source_documents = cast(
            Iterable[EvidenceDocument] | None,
            getattr(self._evidence_store, "documents", None),
        )
        revision_reader = cast(
            RevisionReader | None,
            getattr(self._evidence_store, "authorized_revisions", None),
        )
        if not source_documents and callable(revision_reader):
            source_documents = revision_reader(evidence_filter)
        if source_documents is None:
            raise LightRAGAuthorizationError(
                "LightRAG requires an authoritative evidence revision source."
            )
        authorized_documents = [
            document for document in source_documents if evidence_filter.allows(document)
        ]
        if not authorized_documents:
            return EvidenceInvestigation(
                documents=[], graph_paths=[], lightrag_chain=_empty_lightrag_chain()
            )
        authorized_scope = AuthorizedEvidenceRevisionSet.from_documents(
            authorized_documents,
            evidence_filter,
            revision_source=revision_reader,
        )
        lightrag_chain = validate_authorized_lightrag_chain(
            lightrag_adapter.retrieve_chain(
                query,
                authorized_scope,
                evidence_filter,
                limit=_MAX_EVIDENCE_TOOL_RESULTS,
            ),
            authorized_scope,
            evidence_filter,
        )
        # Keep the downstream candidate set anchored to LightRAG's top-ranked
        # reference; the remaining chain records are explanatory context, not
        # permission to widen the vector candidate set.
        references = lightrag_chain.references[:1]
        authorized_document_ids = {
            reference.source_document_id for reference in references
        }
        authorized_revision_keys = {
            (
                reference.source_document_id,
                reference.source_revision,
                reference.chunk_id,
            )
            for reference in references
        }
        if not authorized_document_ids:
            return EvidenceInvestigation(
                documents=[], graph_paths=[], lightrag_chain=lightrag_chain
            )
        graph_filter = GraphAccessFilter(
            products=graph_filter.products,
            regions=graph_filter.regions,
            tenant_ids=graph_filter.tenant_ids,
            classifications=graph_filter.classifications,
            identifier_entitlements=graph_filter.identifier_entitlements,
            seat_tiers=graph_filter.seat_tiers,
            groups=evidence_filter.groups,
            agent_user_id=evidence_filter.agent_user_id,
            as_of=evidence_filter.as_of,
            authorized_document_ids=tuple(sorted(authorized_document_ids)),
            authorized_revision_keys=tuple(sorted(authorized_revision_keys or ())),
        )

        graph_paths = self._graph_traversal_tool(
            query,
            graph_filter,
            metric_name,
            _MAX_EVIDENCE_TOOL_RESULTS,
        )
        with trace_span(
            "evidence_retrieval",
            kind="tool",
            attributes={"result_limit": _MAX_EVIDENCE_TOOL_RESULTS},
        ):
            scoped_retriever = getattr(self._evidence_store, "retrieve_scoped", None)
            if not callable(scoped_retriever) or not authorized_document_ids:
                raise LightRAGAuthorizationError(
                    "LightRAG requires a backend-enforced scoped evidence retriever."
                )
            retrieved_documents = cast(Any, scoped_retriever)(
                query,
                evidence_filter,
                authorized_document_ids,
                limit=_MAX_EVIDENCE_TOOL_RESULTS,
                authorized_revision_keys=authorized_revision_keys,
            )
        authorized_revision_set = authorized_revision_keys or set()
        documents = []
        for document in retrieved_documents:
            if not isinstance(document, EvidenceDocument):
                raise LightRAGAuthorizationError(
                    "LightRAG scoped retrieval returned an invalid evidence document."
                )
            if _evidence_revision_key(document) not in authorized_revision_set:
                raise LightRAGAuthorizationError(
                    "LightRAG scoped retrieval returned an unauthorized evidence revision."
                )
            if not evidence_filter.allows(document):
                raise LightRAGAuthorizationError(
                    "LightRAG scoped retrieval returned evidence outside the current policy."
                )
            documents.append(document)
        documents = documents[:_MAX_EVIDENCE_TOOL_RESULTS]
        documents = self._rerank_authorized_documents(
            query,
            documents,
            limit=_MAX_EVIDENCE_TOOL_RESULTS,
        )
        return EvidenceInvestigation(
            documents=documents,
            graph_paths=[
                path
                for path in graph_paths
                if graph_filter.allows(path)
            ][:_MAX_EVIDENCE_TOOL_RESULTS],
            lightrag_chain=lightrag_chain,
        )

    def _rerank_authorized_documents(
        self,
        query: str,
        documents: list[EvidenceDocument],
        *,
        limit: int,
    ) -> list[EvidenceDocument]:
        """Rank an already-filtered set and retain only original document objects."""
        if not documents:
            return []
        if self._evidence_reranker is None:
            raise EvidenceRerankerUnavailableError(
                "The required cross-encoder evidence reranker is not configured."
            )
        try:
            ranked = self._evidence_reranker.rerank(query, documents, limit=limit)
        except EvidenceRerankingError:
            raise
        except Exception as error:
            raise EvidenceRerankerUnavailableError(
                "The required cross-encoder evidence reranker is unavailable."
            ) from error
        by_id = {document.document_id: document for document in documents}
        seen: set[str] = set()
        ordered_documents = []
        for document in ranked:
            if not isinstance(document, EvidenceDocument):
                raise EvidenceRerankingError(
                    "The cross-encoder returned a non-evidence candidate."
                )
            document_id = document.document_id
            if document_id not in by_id or document_id in seen:
                raise EvidenceRerankingError(
                    "The cross-encoder changed the authorized evidence candidate set."
                )
            seen.add(document_id)
            ordered_documents.append(by_id[document_id])
        if not ordered_documents:
            raise EvidenceRerankingError("The cross-encoder returned no evidence ranking.")
        return ordered_documents[:limit]


def _empty_lightrag_chain() -> LightRAGEvidenceChain:
    return LightRAGEvidenceChain(
        supporting_chunks=[], entities=[], relations=[], references=[]
    )

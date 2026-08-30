"""Registered, bounded tools for a governed evidence investigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .evidence import EvidenceAccessFilter, EvidenceDocument, VectorEvidenceStore
from .graph import GraphAccessFilter, GraphPath
from .observability import trace_span
from .reranking import (
    EvidenceReranker,
    EvidenceRerankerUnavailableError,
    EvidenceRerankingError,
)

_MAX_EVIDENCE_TOOL_RESULTS = 3

GraphTraversalTool = Callable[[str, GraphAccessFilter, str, int], list[GraphPath]]


@dataclass(frozen=True)
class EvidenceInvestigation:
    """Only the permitted output of the registered evidence tools."""

    documents: list[EvidenceDocument]
    graph_paths: list[GraphPath]


class BoundedEvidenceInvestigationTools:
    """Invoke the two approved evidence tools with a non-negotiable result budget."""

    def __init__(
        self,
        evidence_store: VectorEvidenceStore,
        graph_traversal_tool: GraphTraversalTool,
        evidence_reranker: EvidenceReranker | None,
    ) -> None:
        self._evidence_store = evidence_store
        self._graph_traversal_tool = graph_traversal_tool
        self._evidence_reranker = evidence_reranker

    def investigate(
        self,
        *,
        query: str,
        evidence_filter: EvidenceAccessFilter,
        graph_filter: GraphAccessFilter,
        metric_name: str,
    ) -> EvidenceInvestigation:
        """Pass policy to each tool before it retrieves candidates or graph paths."""
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
            retrieved_documents = self._evidence_store.retrieve(
                query,
                evidence_filter,
                limit=_MAX_EVIDENCE_TOOL_RESULTS,
            )
        documents = [
            document
            for document in retrieved_documents
            if evidence_filter.allows(document)
        ][:_MAX_EVIDENCE_TOOL_RESULTS]
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

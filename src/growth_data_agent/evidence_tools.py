"""Registered, bounded tools for a governed evidence investigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .evidence import EvidenceAccessFilter, EvidenceDocument, VectorEvidenceStore
from .graph import GraphAccessFilter, GraphPath

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
    ) -> None:
        self._evidence_store = evidence_store
        self._graph_traversal_tool = graph_traversal_tool

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
        documents = self._evidence_store.retrieve(
            query,
            evidence_filter,
            limit=_MAX_EVIDENCE_TOOL_RESULTS,
        )
        return EvidenceInvestigation(
            documents=[
                document
                for document in documents
                if evidence_filter.allows(document)
            ][:_MAX_EVIDENCE_TOOL_RESULTS],
            graph_paths=[
                path
                for path in graph_paths
                if graph_filter.allows(path)
            ][:_MAX_EVIDENCE_TOOL_RESULTS],
        )

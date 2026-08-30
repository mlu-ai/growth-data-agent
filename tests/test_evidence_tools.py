from __future__ import annotations

import pytest

from growth_data_agent.evidence import EvidenceAccessFilter, EvidenceDocument
from growth_data_agent.evidence_tools import BoundedEvidenceInvestigationTools
from growth_data_agent.graph import GraphAccessFilter
from growth_data_agent.lightrag import (
    InMemoryLightRAGStore,
    LightRAGAuthorizationError,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
)
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.synthetic import evidence_corpus, graph_corpus


class RecordingEvidenceStore:
    def __init__(self) -> None:
        self.access_filter: EvidenceAccessFilter | None = None
        self.limit: int | None = None

    def retrieve(self, query, access_filter, *, limit):
        self.access_filter = access_filter
        self.limit = limit
        return list(evidence_corpus())


class MaliciousScopedEvidenceStore:
    def __init__(self, authorized: EvidenceDocument, injected: EvidenceDocument) -> None:
        self.documents = [authorized]
        self.injected = injected

    def retrieve(self, query, access_filter, *, limit):
        del query, access_filter, limit
        return []

    def retrieve_scoped(
        self,
        query,
        access_filter,
        authorized_document_ids,
        *,
        limit,
        authorized_revision_keys,
    ):
        del query, access_filter, authorized_document_ids, limit, authorized_revision_keys
        return [self.injected]


def test_investigation_passes_policy_before_each_registered_tool_and_bounds_results() -> None:
    document = evidence_corpus()[0]
    path = graph_corpus()[0]
    evidence_store = RecordingEvidenceStore()
    received_graph_filters: list[GraphAccessFilter] = []
    received_graph_limits: list[int] = []
    evidence_filter = EvidenceAccessFilter(
        products=(document.product,),
        regions=(document.region,),
        tenant_ids=tuple(document.tenant_ids),
        classifications=(document.classification,),
        identifier_entitlements=(document.identifier_entitlement,),
        groups=("evidence-general",),
    )
    graph_filter = GraphAccessFilter(
        products=(path.nodes[0].product,),
        regions=(path.nodes[0].region,),
        tenant_ids=tuple(path.nodes[0].tenant_ids),
        classifications=("internal",),
        identifier_entitlements=("none",),
    )

    def traverse(query, access_filter, metric_name, limit):
        received_graph_filters.append(access_filter)
        received_graph_limits.append(limit)
        return [path] * 4

    tools = BoundedEvidenceInvestigationTools(
        evidence_store,
        traverse,
        DeterministicCrossEncoderReranker(),
    )

    investigation = tools.investigate(
        query="Jira APAC evidence",
        evidence_filter=evidence_filter,
        graph_filter=graph_filter,
        metric_name="jira_new_peu",
    )

    assert evidence_store.access_filter is evidence_filter
    assert evidence_store.limit == 3
    assert received_graph_filters == [graph_filter]
    assert received_graph_limits == [3]
    assert len(investigation.documents) <= 3
    assert len(investigation.graph_paths) == 3


def test_investigation_rejects_a_scoped_store_revision_mismatch() -> None:
    authorized = evidence_corpus()[0]
    injected = authorized.model_copy(
        update={
            "source_revision": "injected-revision",
            "text": "Injected evidence from another active revision.",
        }
    )
    evidence_filter = EvidenceAccessFilter(
        products=(authorized.product,),
        regions=(authorized.region,),
        tenant_ids=tuple(authorized.tenant_ids),
        classifications=(authorized.classification,),
        identifier_entitlements=(authorized.identifier_entitlement,),
        groups=tuple(authorized.access_groups),
    )
    graph_filter = GraphAccessFilter(
        products=(authorized.product,),
        regions=(authorized.region,),
        tenant_ids=tuple(authorized.tenant_ids),
        classifications=(authorized.classification,),
        identifier_entitlements=(authorized.identifier_entitlement,),
    )
    light_rag_store = InMemoryLightRAGStore(
        chunks=[
            LightRAGChunkRecord(
                reference=LightRAGEvidenceReference.from_document(authorized),
                text=authorized.text,
            )
        ]
    )
    tools = BoundedEvidenceInvestigationTools(
        MaliciousScopedEvidenceStore(authorized, injected),
        lambda query, access_filter, metric_name, limit: [],
        DeterministicCrossEncoderReranker(),
        LightRAGEvidenceAdapter(LightRAGBackend(light_rag_store)),
    )

    with pytest.raises(LightRAGAuthorizationError, match="revision"):
        tools.investigate(
            query="Jira APAC evidence",
            evidence_filter=evidence_filter,
            graph_filter=graph_filter,
            metric_name=authorized.metric_name or "jira_new_peu",
        )

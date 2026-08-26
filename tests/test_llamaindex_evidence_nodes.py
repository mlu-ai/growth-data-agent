from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from llama_index.core.vector_stores import VectorStoreQuery

from growth_data_agent.evidence import (
    EvidenceAccessFilter,
    EvidencePrincipalGrant,
    QdrantEvidenceStore,
    _vectorize,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.synthetic import evidence_corpus


def test_synthetic_evidence_is_ingested_as_stable_llamaindex_nodes() -> None:
    corpus = evidence_corpus()
    store = QdrantEvidenceStore(corpus)

    assert len(store.nodes) == len(corpus)
    node = store.nodes[0]
    assert node.node_id == str(
        uuid5(NAMESPACE_URL, "jira-apac-paid-provisioning-incident:chunk:0")
    )
    assert node.metadata["source_document_id"] == "jira-apac-paid-provisioning-incident"
    assert node.metadata["chunk_id"] == "jira-apac-paid-provisioning-incident:chunk:0"
    assert node.metadata["source_url"]
    assert node.metadata["source_revision"] == "synthetic-v1"
    assert node.metadata["freshness"] == "2026-06-13T00:00:00Z"


def test_direct_principal_grants_are_filtered_before_reranking() -> None:
    restricted = next(
        document
        for document in evidence_corpus()
        if document.document_id == "jira-apac-paid-provisioning-incident-restricted"
    )
    store = QdrantEvidenceStore([restricted])
    profile = resolve_access_profile("customer_success_manager")
    access_filter = profile.evidence_filter(
        "Jira",
        "APAC",
        metric_name="jira_new_peu",
        agent_user_id="customer_success_manager",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    documents = store.retrieve("Jira APAC provisioning incident", access_filter, limit=3)

    assert [document.document_id for document in documents] == [restricted.document_id]
    assert store.last_filter == access_filter


def test_expired_or_stale_policy_content_is_never_returned() -> None:
    restricted = next(
        document
        for document in evidence_corpus()
        if document.document_id == "jira-apac-paid-provisioning-incident-restricted"
    )
    expired = restricted.model_copy(
        update={
            "direct_principal_grants": [
                EvidencePrincipalGrant(
                    principal_id="customer_success_manager",
                    expires_at=datetime(2026, 8, 24, tzinfo=UTC),
                )
            ],
        }
    )
    store = QdrantEvidenceStore([expired])
    profile = resolve_access_profile("customer_success_manager")
    access_filter = profile.evidence_filter(
        "Jira",
        "APAC",
        metric_name="jira_new_peu",
        agent_user_id="customer_success_manager",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert store.retrieve("Jira APAC provisioning incident", access_filter, limit=3) == []
    assert store.last_scores == ()
    raw_result = store._vector_store.query(
        VectorStoreQuery(
            query_embedding=_vectorize("Jira APAC provisioning incident"),
            similarity_top_k=3,
        ),
        qdrant_filters=access_filter.as_qdrant_filter(),
    )
    assert raw_result.nodes == []


def test_mixed_tenant_and_group_direct_policy_content_never_reaches_reranking() -> None:
    document = evidence_corpus()[0].model_copy(
        update={
            "tenant_ids": ["tenant-0011", "tenant-0001"],
            "access_groups": ["evidence-general"],
            "direct_principal_grants": [
                EvidencePrincipalGrant(
                    principal_id="another_principal",
                    expires_at=datetime(2099, 12, 31, tzinfo=UTC),
                )
            ],
        }
    )
    store = QdrantEvidenceStore([document])
    profile = resolve_access_profile("customer_success_manager")
    access_filter = profile.evidence_filter(
        "Jira",
        "APAC",
        metric_name="jira_new_peu",
        agent_user_id="customer_success_manager",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert store.retrieve("Jira APAC provisioning incident", access_filter, limit=3) == []
    assert store.last_scores == ()


def test_group_entitlement_is_required_without_per_user_vector_copies() -> None:
    document = evidence_corpus()[0]
    store = QdrantEvidenceStore([document])
    denied_filter = EvidenceAccessFilter(
        products=("Jira",),
        regions=("APAC",),
        tenant_ids=tuple(document.tenant_ids),
        classifications=("internal",),
        identifier_entitlements=("none",),
        groups=("unrelated-group",),
        agent_user_id="data_analyst",
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert len(store.nodes) == 1
    assert store.retrieve("Jira APAC provisioning incident", denied_filter, limit=3) == []

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pytest

from growth_data_agent.evidence import (
    EvidenceAccessFilter,
    EvidenceDocument,
    EvidenceLifecycleState,
    EvidencePrincipalGrant,
    QdrantEvidenceStore,
)
from growth_data_agent.graph import (
    ApacheAgeEvidenceGraphStore,
    GraphPath,
    InMemoryEvidenceGraphStore,
)
from growth_data_agent.lightrag import (
    AuthorizedEvidenceRevisionSet,
    AuthorizedLightRAGIndex,
    InMemoryLightRAGStore,
    LightRAGAuthorizationError,
    LightRAGAuthorizedView,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEntityRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
    LightRAGRelationRecord,
    LightRAGRetrievalStore,
    QdrantAGELightRAGStore,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.synthetic import evidence_corpus, graph_corpus


class UnsupportedLightRAGBackend:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self,
        query: str,
        *,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        limit: int,
    ) -> list[LightRAGEvidenceReference]:
        del query, authorized_scope, access_filter, limit
        self.calls += 1
        return []


class UnsupportedLightRAGStore:
    def retrieve_chunk_vectors(
        self,
        query: str,
        *,
        authorized_scope: AuthorizedEvidenceRevisionSet,
        access_filter: EvidenceAccessFilter,
        limit: int,
    ) -> list[LightRAGChunkRecord]:
        del query, authorized_scope, access_filter, limit
        return []


class UnsafeAuthorizedViewStore(LightRAGRetrievalStore):
    def __init__(self) -> None:
        self.authorize_calls = 0

    def authorized_view(self, capability: object) -> LightRAGAuthorizedView:
        del capability
        self.authorize_calls += 1
        return cast(LightRAGAuthorizedView, object())


def _access_filter(
    *,
    agent_user_id: str = "apac_regional_manager",
    as_of: datetime = datetime(2026, 8, 25, tzinfo=UTC),
):
    return resolve_access_profile(agent_user_id).evidence_filter(
        "Jira",
        "APAC",
        metric_name="jira_new_peu",
        agent_user_id=agent_user_id,
        as_of=as_of,
    )


def _authorized_scope(
    document: EvidenceDocument,
    access_filter=None,
) -> AuthorizedEvidenceRevisionSet:
    return AuthorizedEvidenceRevisionSet.from_documents(
        [document],
        access_filter or _access_filter(),
    )


def _reference(
    document: EvidenceDocument,
    *,
    reference_kind: Literal["chunk", "entity", "relation"] = "chunk",
    **updates: object,
) -> LightRAGEvidenceReference:
    return LightRAGEvidenceReference.from_document(
        document,
        reference_kind=reference_kind,
        reference_id=(
            f"chunk:{document.chunk_id or f'{document.document_id}:chunk:0'}"
            if reference_kind == "chunk"
            else f"{reference_kind}:{document.document_id}"
        ),
    ).model_copy(update=updates)


def _store(*documents: EvidenceDocument) -> InMemoryLightRAGStore:
    chunks = [
        LightRAGChunkRecord(
            reference=_reference(document),
            text=f"{document.title} {document.text}",
        )
        for document in documents
    ]
    entities = [
        LightRAGEntityRecord(
            reference=_reference(document, reference_kind="entity"),
            name=f"{document.product} {document.region} provisioning",
            description=document.text,
        )
        for document in documents
    ]
    entities_by_document_id = {
        document.document_id: entity.reference
        for document, entity in zip(documents, entities, strict=True)
    }
    relations = [
        LightRAGRelationRecord(
            reference=_reference(
                document,
                reference_kind="relation",
                related_entity_references=[
                    entities_by_document_id[document.document_id],
                    entities_by_document_id[document.document_id],
                ],
            ),
            source_entity=entities_by_document_id[document.document_id],
            target_entity=entities_by_document_id[document.document_id],
            description=document.text,
        )
        for document in documents
    ]
    return InMemoryLightRAGStore(
        chunks=chunks,
        entities=entities,
        relations=relations,
    )


def test_adapter_passes_authorized_active_scope_before_real_retrieval() -> None:
    document = evidence_corpus()[0]
    store = _store(document)
    backend = LightRAGBackend(store)
    adapter = LightRAGEvidenceAdapter(backend)
    scope = _authorized_scope(document)
    access_filter = _access_filter()

    references = adapter.retrieve("APAC provisioning", scope, access_filter)

    assert {reference.reference_kind for reference in references} == {
        "chunk",
        "entity",
        "relation",
    }
    assert backend.last_scope is scope
    assert backend.last_query == "APAC provisioning"
    assert [call.kind for call in store.calls] == ["chunk_vector", "entity_graph", "relation_graph"]
    assert all(call.authorized_reference_ids for call in store.calls)
    assert [call.authorized_reference_ids for call in store.calls] == [
        frozenset({"chunk:jira-apac-paid-provisioning-incident:chunk:0"}),
        frozenset({"entity:jira-apac-paid-provisioning-incident"}),
        frozenset({"relation:jira-apac-paid-provisioning-incident"}),
    ]


def test_query_drives_chunk_entity_and_relation_retrieval() -> None:
    allowed, unrelated = evidence_corpus()[0], evidence_corpus()[3]
    store = _store(allowed, unrelated)
    access_filter = _access_filter()

    references = LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed, access_filter),
        access_filter,
    )

    assert {reference.source_document_id for reference in references} == {allowed.document_id}
    assert len(store.calls) == 3
    assert all(call.query == "APAC provisioning" for call in store.calls)


def test_reference_preserves_revision_and_source_access_metadata() -> None:
    document = evidence_corpus()[0].model_copy(
        update={
            "access_groups": ["evidence-general", "regional-managers"],
            "policy_expires_at": datetime(2027, 1, 1, tzinfo=UTC),
            "revision_fingerprint": "fingerprint-42",
        }
    )

    reference = LightRAGEvidenceReference.from_document(document)

    assert reference.source_document_id == document.document_id
    assert reference.source_revision == document.source_revision
    assert reference.chunk_id == f"{document.document_id}:chunk:0"
    assert reference.revision_fingerprint == "fingerprint-42"
    assert reference.access_groups == ["evidence-general", "regional-managers"]
    assert reference.policy_expires_at == datetime(2027, 1, 1, tzinfo=UTC)


def test_authorized_scope_cannot_be_constructed_without_the_authorization_factory() -> None:
    with pytest.raises(TypeError, match="factory"):
        AuthorizedEvidenceRevisionSet(revisions=(evidence_corpus()[0],))


def test_backend_cannot_override_the_scope_enforcing_retrieval_entrypoint() -> None:
    with pytest.raises(TypeError, match="retrieve"):
        type(
            "UnsafeLightRAGBackend",
            (LightRAGBackend,),
            {"retrieve": lambda self, query, *, authorized_scope, access_filter, limit: []},
        )


def test_backend_rejects_a_runtime_scope_impostor() -> None:
    backend = LightRAGBackend(_store(evidence_corpus()[0]))

    with pytest.raises(LightRAGAuthorizationError, match="authorized Evidence Revision"):
        backend.retrieve(
            "APAC provisioning",
            authorized_scope=cast(Any, object()),
            access_filter=_access_filter(),
            limit=1,
        )


def test_backend_rejects_a_store_without_proven_pre_retrieval_filtering() -> None:
    with pytest.raises(LightRAGAuthorizationError, match="graph/vector"):
        LightRAGBackend(cast(Any, UnsupportedLightRAGStore()))


def test_backend_rejects_an_authorized_view_without_typed_retrieval_operations() -> None:
    document = evidence_corpus()[0]
    store = UnsafeAuthorizedViewStore()
    access_filter = _access_filter()

    with pytest.raises(LightRAGAuthorizationError, match="graph/vector|authorized view"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            _authorized_scope(document, access_filter),
            access_filter,
        )

    assert store.authorize_calls == 0


def test_backend_accepts_the_concrete_qdrant_age_store() -> None:
    document = evidence_corpus()[0]
    access_filter = _access_filter()
    qdrant_store = QdrantEvidenceStore([document])
    store = QdrantAGELightRAGStore(
        qdrant_store,
        InMemoryEvidenceGraphStore(graph_corpus()),
    )

    references = LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(document, access_filter),
        access_filter,
    )

    assert {reference.reference_kind for reference in references} == {
        "chunk",
        "entity",
        "relation",
    }


def test_concrete_qdrant_and_age_queries_receive_the_authorized_revision_scope() -> None:
    document = evidence_corpus()[0]
    access_filter = _access_filter()
    scope = _authorized_scope(document, access_filter)

    class RecordingQdrantClient:
        def __init__(self) -> None:
            from qdrant_client import QdrantClient

            self.inner = QdrantClient(location=":memory:")
            self.query_calls: list[dict[str, object]] = []

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

        def query_points(self, **kwargs: object):
            self.query_calls.append(kwargs)
            return cast(Any, self.inner).query_points(**kwargs)

    class RecordingAgeExecutor:
        def __init__(self, paths: list[GraphPath]) -> None:
            self.paths = paths
            self.query_calls: list[tuple[str, dict[str, object]]] = []

        def query(self, cypher: str, parameters: dict[str, object]) -> list[GraphPath]:
            self.query_calls.append((cypher, parameters))
            return self.paths

    qdrant_client = RecordingQdrantClient()
    age_executor = RecordingAgeExecutor(
        [next(path for path in graph_corpus() if path.path_id.startswith(document.document_id))]
    )
    qdrant_store = QdrantEvidenceStore([document], client=cast(Any, qdrant_client))
    age_store = ApacheAgeEvidenceGraphStore(age_executor)
    store = QdrantAGELightRAGStore(qdrant_store, age_store)

    references = LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        scope,
        access_filter,
    )

    assert {reference.reference_kind for reference in references} == {
        "chunk",
        "entity",
        "relation",
    }
    assert len(qdrant_client.query_calls) == 1
    query_filter = cast(Any, qdrant_client.query_calls[0]["query_filter"])
    assert "source_document_id" in query_filter.model_dump_json()
    assert qdrant_store.last_authorized_revision_keys == (
        (document.document_id, document.source_revision, f"{document.document_id}:chunk:0"),
    )
    age_parameters = cast(Any, age_executor.query_calls[0][1])
    assert document.document_id in age_parameters["authorized_document_ids"]
    assert "$authorized_source_document_id_0" in age_executor.query_calls[0][0]


def test_qdrant_keeps_same_chunk_id_revisions_independently_scoped() -> None:
    base_document = evidence_corpus()[0]
    revision_one = base_document.model_copy(
        update={
            "document_id": "jira-apac-paid-provisioning-incident",
            "source_document_id": "jira-apac-paid-provisioning-incident",
            "source_page_id": "jira-apac-paid-provisioning-incident",
            "source_url": "https://evidence.local/jira-apac-paid-provisioning-incident",
            "source_revision": "revision-1",
            "chunk_id": "jira-apac-paid-provisioning-incident:chunk:0",
            "revision_fingerprint": "fingerprint-1",
            "text": "Revision one APAC provisioning evidence.",
        }
    )
    revision_two = revision_one.model_copy(
        update={
            "source_revision": "revision-2",
            "revision_fingerprint": "fingerprint-2",
            "text": "Revision two APAC provisioning evidence.",
        }
    )
    store = QdrantEvidenceStore([revision_one, revision_two])
    access_filter = _access_filter()

    assert len(store.nodes) == 2
    assert len({node.node_id for node in store.nodes}) == 2

    first = store.retrieve_scoped(
        "APAC provisioning",
        access_filter,
        {"jira-apac-paid-provisioning-incident"},
        limit=3,
        authorized_revision_keys={
            (
                "jira-apac-paid-provisioning-incident",
                "revision-1",
                "jira-apac-paid-provisioning-incident:chunk:0",
            )
        },
    )
    second = store.retrieve_scoped(
        "APAC provisioning",
        access_filter,
        {"jira-apac-paid-provisioning-incident"},
        limit=3,
        authorized_revision_keys={
            (
                "jira-apac-paid-provisioning-incident",
                "revision-2",
                "jira-apac-paid-provisioning-incident:chunk:0",
            )
        },
    )

    assert [document.source_revision for document in first] == ["revision-1"]
    assert [document.source_revision for document in second] == ["revision-2"]


def test_backend_retrieval_entrypoint_cannot_be_replaced_on_an_instance() -> None:
    backend = LightRAGBackend(_store(evidence_corpus()[0]))

    with pytest.raises(AttributeError, match="retrieve"):
        setattr(backend, "retrieve", lambda: [])


def test_scoped_index_enforces_the_global_result_bound() -> None:
    document = evidence_corpus()[0]
    store = _store(document)
    scope = _authorized_scope(document)
    index = AuthorizedLightRAGIndex(store, scope, _access_filter())

    assert len(index.retrieve_chunks("APAC provisioning", limit=10)) == 1


@pytest.mark.parametrize("reference_kind", ["chunk", "entity", "relation"])
def test_adversarial_out_of_scope_reference_never_reaches_model_context(
    reference_kind: str,
) -> None:
    allowed = evidence_corpus()[0]
    denied = next(
        document
        for document in evidence_corpus()
        if document.document_id == "jira-apac-paid-provisioning-incident-restricted"
    )
    store = _store(denied)
    access_filter = _access_filter()
    backend = LightRAGBackend(store)

    references = LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed, access_filter),
        access_filter,
    )

    assert references == []
    assert all(not call.returned_reference_ids for call in store.calls)


def test_tenant_scoped_reference_fails_closed() -> None:
    allowed = evidence_corpus()[0]
    denied = allowed.model_copy(update={"tenant_ids": ["tenant-0011"]})
    store = _store(denied)
    access_filter = _access_filter()

    assert LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed, access_filter),
        access_filter,
    ) == []


def test_reference_kind_must_match_its_reference_identity() -> None:
    document = evidence_corpus()[0]
    forged_reference = _reference(document).model_copy(update={"reference_kind": "relation"})
    store = InMemoryLightRAGStore(
        chunks=[LightRAGChunkRecord(reference=forged_reference, text=document.text)]
    )
    access_filter = _access_filter()

    assert LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(document, access_filter),
        access_filter,
    ) == []


def test_restricted_classification_and_identifier_reference_fails_closed() -> None:
    allowed = evidence_corpus()[0]
    restricted = allowed.model_copy(
        update={"classification": "restricted", "identifier_entitlement": "direct"}
    )
    store = _store(restricted)
    access_filter = _access_filter()

    assert LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed, access_filter),
        access_filter,
    ) == []


def test_relation_with_an_unauthorized_graph_endpoint_is_filtered_before_lookup() -> None:
    allowed, denied = evidence_corpus()[0], evidence_corpus()[1]
    source_entity = _reference(allowed, reference_kind="entity")
    target_entity = _reference(denied, reference_kind="entity")
    relation = LightRAGRelationRecord(
        reference=_reference(
            allowed,
            reference_kind="relation",
            related_entity_references=[source_entity, target_entity],
        ),
        source_entity=source_entity,
        target_entity=target_entity,
        description="APAC provisioning relation",
    )
    store = InMemoryLightRAGStore(relations=[relation])
    access_filter = _access_filter()

    assert LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed, access_filter),
        access_filter,
    ) == []
    assert not store.calls[-1].returned_reference_ids


def test_unsupported_pre_retrieval_filtering_fails_before_backend_call() -> None:
    backend = UnsupportedLightRAGBackend()
    document = evidence_corpus()[0]
    access_filter = _access_filter()

    with pytest.raises(LightRAGAuthorizationError, match="pre-retrieval"):
        LightRAGEvidenceAdapter(backend).retrieve(
            "APAC provisioning",
            _authorized_scope(document, access_filter),
            access_filter,
        )

    assert backend.calls == 0


def test_expired_policy_revalidates_before_any_retrieval() -> None:
    document = evidence_corpus()[0].model_copy(
        update={"policy_expires_at": datetime(2026, 8, 26, tzinfo=UTC)}
    )
    authorized_filter = _access_filter(as_of=datetime(2026, 8, 25, tzinfo=UTC))
    expired_filter = _access_filter(as_of=datetime(2026, 8, 27, tzinfo=UTC))
    store = _store(document)

    with pytest.raises(LightRAGAuthorizationError, match="active|policy"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            _authorized_scope(document, authorized_filter),
            expired_filter,
        )

    assert store.calls == []


def test_current_revision_manifest_revocation_invalidates_a_previously_issued_scope() -> None:
    document = evidence_corpus()[0]
    access_filter = _access_filter()
    revoked_document = document.model_copy(
        update={"lifecycle_state": EvidenceLifecycleState.DELETED}
    )
    scope = AuthorizedEvidenceRevisionSet.from_documents(
        [document],
        access_filter,
        revision_source=lambda _: [revoked_document],
    )
    store = _store(document)

    with pytest.raises(LightRAGAuthorizationError, match="stale|revoked"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            scope,
            access_filter,
        )

    assert store.calls == []


def test_revoked_group_revalidates_before_any_retrieval() -> None:
    document = evidence_corpus()[0]
    authorized_filter = _access_filter()
    revoked_filter = replace(authorized_filter, groups=("revoked-group",))
    store = _store(document)

    with pytest.raises(LightRAGAuthorizationError, match="authorized"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            _authorized_scope(document, authorized_filter),
            revoked_filter,
        )

    assert store.calls == []


def test_revoked_direct_grant_revalidates_before_any_retrieval() -> None:
    document = evidence_corpus()[0].model_copy(
        update={
            "direct_principal_grants": [
                EvidencePrincipalGrant(
                    principal_id="apac_regional_manager",
                    expires_at=datetime(2027, 1, 1, tzinfo=UTC),
                )
            ]
        }
    )
    authorized_filter = _access_filter()
    revoked_filter = replace(authorized_filter, agent_user_id="another-principal")
    store = _store(document)

    with pytest.raises(LightRAGAuthorizationError, match="authorized"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            _authorized_scope(document, authorized_filter),
            revoked_filter,
        )

    assert store.calls == []


def test_adapter_bounds_references_and_never_returns_a_generated_answer() -> None:
    document = evidence_corpus()[0]
    store = _store(document)
    access_filter = _access_filter()

    references = LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
        "APAC provisioning",
        _authorized_scope(document, access_filter),
        access_filter,
        limit=10,
    )

    assert len(references) == 3
    assert all(isinstance(reference, LightRAGEvidenceReference) for reference in references)
    assert not hasattr(references[0], "answer")

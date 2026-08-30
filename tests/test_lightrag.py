from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pytest

from growth_data_agent.evidence import EvidenceDocument, EvidencePrincipalGrant
from growth_data_agent.lightrag import (
    AuthorizedEvidenceRevisionSet,
    AuthorizedLightRAGIndex,
    InMemoryLightRAGStore,
    LightRAGAuthorizationError,
    LightRAGBackend,
    LightRAGChunkRecord,
    LightRAGEntityRecord,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
    LightRAGRelationRecord,
    LightRAGRetrievalStore,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.synthetic import evidence_corpus


class UnsupportedLightRAGBackend:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, *, authorized_scope, access_filter, limit: int):
        del query, authorized_scope, access_filter, limit
        self.calls += 1
        return []


class UnsupportedLightRAGStore:
    def retrieve_chunk_vectors(self, query, *, authorized_scope, access_filter, limit):
        del query, authorized_scope, access_filter, limit
        return []


class UnsafeAuthorizedViewStore(LightRAGRetrievalStore):
    def __init__(self) -> None:
        self.authorize_calls = 0

    def authorized_view(self, authorized_scope, access_filter):
        del authorized_scope, access_filter
        self.authorize_calls += 1
        return cast(Any, object())


class CompatibleLightRAGStore(LightRAGRetrievalStore):
    """A Qdrant/AGE-style wrapper proving the store seam is structural."""

    def __init__(self, store: InMemoryLightRAGStore) -> None:
        self._store = store

    def authorized_view(self, authorized_scope, access_filter):
        return self._store.authorized_view(authorized_scope, access_filter)


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
        reference_id=f"{reference_kind}:{document.document_id}",
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
        frozenset({"chunk:jira-apac-paid-provisioning-incident"}),
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

    with pytest.raises(LightRAGAuthorizationError, match="authorized view"):
        LightRAGEvidenceAdapter(LightRAGBackend(store)).retrieve(
            "APAC provisioning",
            _authorized_scope(document, access_filter),
            access_filter,
        )

    assert store.authorize_calls == 1


def test_backend_accepts_a_store_adapter_with_an_explicit_authorized_view() -> None:
    document = evidence_corpus()[0]
    access_filter = _access_filter()
    store = CompatibleLightRAGStore(_store(document))

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

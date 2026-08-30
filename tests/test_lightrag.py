from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, cast

import pytest

from growth_data_agent.evidence import EvidenceDocument
from growth_data_agent.lightrag import (
    AuthorizedEvidenceRevisionSet,
    AuthorizedLightRAGIndex,
    LightRAGAuthorizationError,
    LightRAGBackend,
    LightRAGEvidenceAdapter,
    LightRAGEvidenceReference,
)
from growth_data_agent.policy import resolve_access_profile
from growth_data_agent.synthetic import evidence_corpus


class UnsupportedLightRAGBackend:
    supports_pre_retrieval_authorization = False

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, *, authorized_scope, limit: int):
        self.calls += 1
        return []


def _access_filter(*, agent_user_id: str = "apac_regional_manager"):
    return resolve_access_profile(agent_user_id).evidence_filter(
        "Jira",
        "APAC",
        metric_name="jira_new_peu",
        agent_user_id=agent_user_id,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
    )


def _authorized_scope(document: EvidenceDocument) -> AuthorizedEvidenceRevisionSet:
    return AuthorizedEvidenceRevisionSet.from_documents(
        [document],
        _access_filter(),
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


def test_adapter_passes_authorized_active_scope_before_retrieval() -> None:
    document = evidence_corpus()[0]
    backend = LightRAGBackend([_reference(document)])
    adapter = LightRAGEvidenceAdapter(backend)
    scope = _authorized_scope(document)

    references = adapter.retrieve("APAC provisioning", scope)

    assert references == [_reference(document)]
    assert backend.last_scope is scope
    assert backend.last_query == "APAC provisioning"
    assert backend.last_candidate_references == tuple(references)
    assert scope.revision_keys == frozenset(
        {(document.source_document_id or document.document_id, document.source_revision)}
    )


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
            {
                "retrieve": lambda self, query, *, authorized_scope, limit: [],
                "_retrieve_from_scoped_index": lambda self, query, *, authorized_index, limit: [],
            },
        )


def test_backend_rejects_a_runtime_scope_impostor() -> None:
    backend = LightRAGBackend([])

    with pytest.raises(LightRAGAuthorizationError, match="authorized Evidence Revision"):
        backend.retrieve(
            "APAC provisioning",
            authorized_scope=cast(Any, object()),
            limit=1,
        )


def test_backend_retrieval_entrypoint_cannot_be_replaced_on_an_instance() -> None:
    backend = LightRAGBackend([])

    with pytest.raises(AttributeError, match="retrieve"):
        setattr(backend, "retrieve", lambda: [])


def test_scoped_index_enforces_the_global_result_bound() -> None:
    document = evidence_corpus()[0]
    scope = _authorized_scope(document)
    index = AuthorizedLightRAGIndex([_reference(document)] * 10, scope)

    assert len(index.retrieve(limit=10)) == 3


def test_indexed_backend_scopes_graph_and_vector_references_before_lookup() -> None:
    allowed, denied = evidence_corpus()[0], evidence_corpus()[1]
    backend = LightRAGBackend(
        [_reference(allowed), _reference(allowed, reference_kind="entity"), _reference(denied)]
    )

    references = LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning",
        _authorized_scope(allowed),
    )

    assert {reference.source_document_id for reference in references} == {allowed.document_id}
    assert {reference.reference_kind for reference in references} == {"chunk", "entity"}
    assert all(
        reference.source_revision == allowed.source_revision
        for reference in backend.last_candidate_references
    )


@pytest.mark.parametrize("reference_kind", ["chunk", "entity", "relation"])
def test_adversarial_out_of_scope_reference_fails_closed_before_model_context(
    reference_kind: str,
) -> None:
    allowed = evidence_corpus()[0]
    denied = next(
        document
        for document in evidence_corpus()
        if document.document_id == "jira-apac-paid-provisioning-incident-restricted"
    )
    backend = LightRAGBackend(
        [
            _reference(
                denied,
                reference_kind=cast(Literal["chunk", "entity", "relation"], reference_kind),
            )
        ]
    )
    adapter = LightRAGEvidenceAdapter(backend)

    assert adapter.retrieve("APAC provisioning", _authorized_scope(allowed)) == []


def test_tenant_scoped_reference_fails_closed() -> None:
    allowed = evidence_corpus()[0]
    denied = allowed.model_copy(update={"tenant_ids": ["tenant-0011"]})
    backend = LightRAGBackend([_reference(denied)])

    assert LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning", _authorized_scope(allowed)
    ) == []


def test_reference_kind_must_match_its_reference_identity() -> None:
    document = evidence_corpus()[0]
    forged_reference = _reference(document).model_copy(update={"reference_kind": "relation"})
    backend = LightRAGBackend([forged_reference])

    assert LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning", _authorized_scope(document)
    ) == []


def test_restricted_classification_and_identifier_reference_fails_closed() -> None:
    allowed = evidence_corpus()[0]
    restricted = allowed.model_copy(
        update={"classification": "restricted", "identifier_entitlement": "direct"}
    )
    backend = LightRAGBackend([_reference(restricted)])

    assert LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning", _authorized_scope(allowed)
    ) == []


def test_unsupported_pre_retrieval_filtering_fails_before_backend_call() -> None:
    backend = UnsupportedLightRAGBackend()
    document = evidence_corpus()[0]

    with pytest.raises(LightRAGAuthorizationError, match="pre-retrieval"):
        LightRAGEvidenceAdapter(backend).retrieve(
            "APAC provisioning",
            _authorized_scope(document),
        )

    assert backend.calls == 0


def test_adapter_bounds_references_and_never_returns_a_generated_answer() -> None:
    document = evidence_corpus()[0]
    backend = LightRAGBackend([_reference(document)] * 10)

    references = LightRAGEvidenceAdapter(backend).retrieve(
        "APAC provisioning",
        _authorized_scope(document),
        limit=10,
    )

    assert len(references) == 3
    assert all(isinstance(reference, LightRAGEvidenceReference) for reference in references)
    assert not hasattr(references[0], "answer")

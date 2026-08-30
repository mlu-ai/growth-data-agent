from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import ValidationError
from qdrant_client import QdrantClient

from growth_data_agent.contracts import EvidenceSupportStatus
from growth_data_agent.evidence import EvidenceAccessFilter, _vectorize
from growth_data_agent.evidence_sync import (
    ConfluenceEvidenceChunk,
    ConfluenceEvidenceRevision,
    EvidenceLifecycleState,
    EvidenceQdrantError,
    EvidenceSyncResult,
    HashEmbeddingProvider,
    QdrantEvidenceSynchronizer,
    SourceAccessMetadata,
)


class StaticConfluenceSource:
    def __init__(self, revisions: list[ConfluenceEvidenceRevision]) -> None:
        self.revisions = revisions

    def iter_revisions(self):
        return iter(self.revisions)


def _revision(
    *,
    source_revision: str = "42",
    lifecycle_state: EvidenceLifecycleState = EvidenceLifecycleState.ACTIVE,
    chunks: list[ConfluenceEvidenceChunk] | None = None,
) -> ConfluenceEvidenceRevision:
    return ConfluenceEvidenceRevision(
        source_page_id="page-123",
        source_url="https://confluence.example/pages/page-123",
        source_revision=source_revision,
        lifecycle_state=lifecycle_state,
        metric_name="confluence_new_mau",
        title="Onboarding regression",
        product="Confluence",
        region="EMEA",
        tenant_ids=["tenant-0001"],
        tenant_scope="EMEA 51-200 Seat Tier Tenants",
        relevant_date=date(2026, 6, 20),
        freshness=datetime(2026, 6, 21, tzinfo=UTC),
        support_status=EvidenceSupportStatus.SUPPORTS,
        support_explanation="The source overlaps the affected period.",
        chunks=chunks
        if chunks is not None
        else [
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:0",
                chunk_index=0,
                text="The onboarding email regression overlapped the decline.",
            ),
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:1",
                chunk_index=1,
                text="The affected page belongs to the EMEA segment.",
            ),
        ],
        source_access=SourceAccessMetadata(
            classification="internal",
            identifier_entitlement="none",
            access_groups=["evidence-general"],
            policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
        ),
        embedding_model="deterministic-hash",
        embedding_version="1",
    )


def _access_filter() -> EvidenceAccessFilter:
    return EvidenceAccessFilter(
        products=("Confluence",),
        regions=("EMEA",),
        tenant_ids=("tenant-0001",),
        classifications=("internal",),
        identifier_entitlements=("none",),
        metric_names=("confluence_new_mau",),
    )


def _synchronizer(client: QdrantClient) -> QdrantEvidenceSynchronizer:
    return QdrantEvidenceSynchronizer(
        client=client,
        collection_name="sync-test",
        embedding_provider=HashEmbeddingProvider(),
    )


def test_backfill_persists_revision_chunk_policy_and_embedding_metadata() -> None:
    client = QdrantClient(location=":memory:")
    result = _synchronizer(client).sync(StaticConfluenceSource([_revision()]))

    assert result == EvidenceSyncResult(indexed_revision_count=1)
    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert len(points) == 2
    payload = points[0].payload
    assert payload["source_page_id"] == "page-123"
    assert payload["source_url"] == "https://confluence.example/pages/page-123"
    assert payload["source_revision"] == "42"
    assert payload["chunk_id"].startswith("page-123:chunk:")
    assert payload["lifecycle_state"] == "active"
    assert payload["classification"] == "internal"
    assert payload["identifier_entitlement"] == "none"
    assert payload["embedding_model"] == "deterministic-hash"
    assert payload["embedding_version"] == "1"


def test_unchanged_revision_is_idempotent() -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    source = StaticConfluenceSource([_revision()])

    synchronizer.sync(source)
    result = synchronizer.sync(source)

    assert result == EvidenceSyncResult(skipped_revision_count=1)
    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert len(points) == 2


def test_updated_revision_replaces_old_chunks() -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    updated = _revision(
        source_revision="43",
        chunks=[
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:0",
                chunk_index=0,
                text="The corrected onboarding email finding.",
            )
        ],
    )
    result = synchronizer.sync(StaticConfluenceSource([updated]))

    assert result == EvidenceSyncResult(indexed_revision_count=1, removed_revision_count=1)
    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert len(points) == 1
    assert points[0].payload["source_revision"] == "43"
    assert points[0].payload["text"] == "The corrected onboarding email finding."


def test_chunk_ids_are_scoped_to_the_source_page() -> None:
    client = QdrantClient(location=":memory:")
    first = _revision().model_copy(update={"source_page_id": "page-123"})
    second = _revision().model_copy(
        update={
            "source_page_id": "page-456",
            "source_url": "https://confluence.example/pages/page-456",
        }
    )

    _synchronizer(client).sync(StaticConfluenceSource([first, second]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)

    assert len(points) == 4
    assert {point.payload["source_page_id"] for point in points} == {"page-123", "page-456"}
    assert len({point.id for point in points}) == 4


def test_existing_page_points_are_fully_paginated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    first = SimpleNamespace(id="point-1", payload={})
    second = SimpleNamespace(id="point-2", payload={})
    offsets = []

    monkeypatch.setattr(client, "collection_exists", lambda collection_name: True)

    def scroll(*args: object, offset=None, **kwargs: object):
        offsets.append(offset)
        return ([first], "next") if offset is None else ([second], None)

    monkeypatch.setattr(client, "scroll", scroll)

    points = synchronizer._existing_page_points("page-123")

    assert [point.id for point in points] == ["point-1", "point-2"]
    assert offsets == [None, "next"]


def test_access_metadata_change_is_not_hidden_by_same_content_revision() -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    restricted = _revision(
        chunks=[
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:0",
                chunk_index=0,
                text="The onboarding email regression overlapped the decline.",
            ),
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:1",
                chunk_index=1,
                text="The affected page belongs to the EMEA segment.",
            ),
        ],
    ).model_copy(
        update={
            "source_access": SourceAccessMetadata(
                classification="restricted",
                identifier_entitlement="direct",
                access_groups=[],
                policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
            )
        }
    )

    result = synchronizer.sync(StaticConfluenceSource([restricted]))
    points, _ = client.scroll("sync-test", limit=10, with_payload=True)

    assert result == EvidenceSyncResult(indexed_revision_count=1, removed_revision_count=1)
    assert {point.payload["classification"] for point in points} == {"restricted"}


@pytest.mark.parametrize(
    "lifecycle_state",
    [EvidenceLifecycleState.DELETED, EvidenceLifecycleState.INACCESSIBLE],
)
def test_deleted_or_inaccessible_revision_is_not_retrievable(
    lifecycle_state: EvidenceLifecycleState,
) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    tombstone = _revision(lifecycle_state=lifecycle_state, source_revision="43", chunks=[])
    result = synchronizer.sync(StaticConfluenceSource([tombstone]))
    vector_store = QdrantVectorStore(client=client, collection_name="sync-test")
    query_result = vector_store.query(
        VectorStoreQuery(
            query_embedding=_vectorize("onboarding regression"),
            similarity_top_k=10,
        ),
        qdrant_filters=_access_filter().as_qdrant_filter(),
    )

    assert result == EvidenceSyncResult(indexed_revision_count=0, removed_revision_count=1)
    assert query_result.nodes == []
    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert len(points) == 1
    assert points[0].payload["lifecycle_state"] == lifecycle_state.value
    assert points[0].payload["text"] == ""


def test_qdrant_failure_is_reported_without_claiming_a_successful_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)

    def fail_upsert(*args: object, **kwargs: object) -> None:
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(client, "upload_points", fail_upsert)

    with pytest.raises(EvidenceQdrantError, match="Qdrant synchronization failed"):
        synchronizer.sync(StaticConfluenceSource([_revision()]))


def test_failed_revision_removal_does_not_write_a_mixed_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    original_delete = client.delete
    delete_calls = 0

    def fail_first_delete(*args: object, **kwargs: object) -> None:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise RuntimeError("qdrant delete unavailable")
        original_delete(*args, **kwargs)

    monkeypatch.setattr(client, "delete", fail_first_delete)

    with pytest.raises(EvidenceQdrantError, match="Qdrant synchronization failed"):
        synchronizer.sync(StaticConfluenceSource([_revision(source_revision="43")]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert all(
        point.payload["lifecycle_state"] == EvidenceLifecycleState.INACCESSIBLE.value
        for point in points
    )

    synchronizer.sync(StaticConfluenceSource([_revision(source_revision="43")]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert len(points) == 2
    assert {point.payload["source_revision"] for point in points} == {"43"}
    assert {point.payload["lifecycle_state"] for point in points} == {
        EvidenceLifecycleState.ACTIVE.value
    }


def test_failed_revision_upload_preserves_the_previous_revision(monkeypatch) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    def fail_upsert(*args: object, **kwargs: object) -> None:
        raise RuntimeError("qdrant upload unavailable")

    monkeypatch.setattr(client, "upload_points", fail_upsert)

    with pytest.raises(EvidenceQdrantError, match="Qdrant synchronization failed"):
        synchronizer.sync(StaticConfluenceSource([_revision(source_revision="43")]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert {point.payload["source_revision"] for point in points} == {"42"}


def test_partial_revision_upload_is_quarantined_when_cleanup_fails(monkeypatch) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    original_upload_points = client.upload_points

    def partial_upload(collection_name, points, **kwargs):
        original_upload_points(collection_name, points[:1], **kwargs)
        raise RuntimeError("qdrant upload partially failed")

    def fail_delete(*args: object, **kwargs: object) -> None:
        raise RuntimeError("qdrant cleanup unavailable")

    monkeypatch.setattr(client, "upload_points", partial_upload)
    monkeypatch.setattr(client, "delete", fail_delete)

    with pytest.raises(EvidenceQdrantError, match="Qdrant synchronization failed"):
        synchronizer.sync(StaticConfluenceSource([_revision(source_revision="43")]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert points
    assert all(
        point.payload["lifecycle_state"] == EvidenceLifecycleState.INACCESSIBLE.value
        for point in points
    )


def test_failed_revision_promotion_quarantines_partially_promoted_points(monkeypatch) -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    synchronizer.sync(StaticConfluenceSource([_revision()]))

    original_set_payload = client.set_payload

    def partial_promotion(collection_name, payload, points, **kwargs):
        if payload["lifecycle_state"] == EvidenceLifecycleState.ACTIVE.value:
            original_set_payload(
                collection_name,
                payload=payload,
                points=points[:1],
                **kwargs,
            )
            raise RuntimeError("qdrant promotion partially failed")
        original_set_payload(collection_name, payload=payload, points=points, **kwargs)

    monkeypatch.setattr(client, "set_payload", partial_promotion)

    with pytest.raises(EvidenceQdrantError, match="Qdrant synchronization failed"):
        synchronizer.sync(StaticConfluenceSource([_revision(source_revision="43")]))

    points, _ = client.scroll("sync-test", limit=10, with_payload=True)
    assert points
    assert all(
        point.payload["lifecycle_state"] == EvidenceLifecycleState.INACCESSIBLE.value
        for point in points
    )


def test_invalid_revision_aborts_the_batch_before_any_content_is_indexed() -> None:
    client = QdrantClient(location=":memory:")
    synchronizer = _synchronizer(client)
    invalid_revision = _revision().model_dump()
    invalid_revision["source_url"] = ""

    with pytest.raises(ValidationError):
        synchronizer.sync(StaticConfluenceSource([_revision(), invalid_revision]))

    assert not client.collection_exists("sync-test")

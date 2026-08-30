from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from growth_data_agent.contracts import EvidenceSupportStatus
from growth_data_agent.evidence_sync import (
    ConfluenceEvidenceChunk,
    ConfluenceEvidenceRevision,
    EvidenceLifecycleState,
    EvidenceRevisionValidationError,
    SourceAccessMetadata,
)


def _access_metadata() -> SourceAccessMetadata:
    return SourceAccessMetadata(
        classification="internal",
        identifier_entitlement="none",
        access_groups=["evidence-general"],
        policy_expires_at=datetime(2099, 12, 31, tzinfo=UTC),
    )


def _revision(**overrides: object) -> ConfluenceEvidenceRevision:
    values: dict[str, object] = {
        "source_page_id": "page-123",
        "source_url": "https://confluence.example/pages/page-123",
        "source_revision": "42",
        "lifecycle_state": EvidenceLifecycleState.ACTIVE,
        "metric_name": "confluence_new_mau",
        "title": "Onboarding regression",
        "product": "Confluence",
        "region": "EMEA",
        "tenant_ids": ["tenant-0001"],
        "tenant_scope": "EMEA 51-200 Seat Tier Tenants",
        "relevant_date": date(2026, 6, 20),
        "freshness": datetime(2026, 6, 21, tzinfo=UTC),
        "support_status": EvidenceSupportStatus.SUPPORTS,
        "support_explanation": "The source overlaps the affected period.",
        "chunks": [
            ConfluenceEvidenceChunk(
                chunk_id="page-123:chunk:0",
                chunk_index=0,
                text="The onboarding email regression overlapped the decline.",
            )
        ],
        "source_access": _access_metadata(),
        "embedding_model": "deterministic-hash",
        "embedding_version": "1",
    }
    values.update(overrides)
    return ConfluenceEvidenceRevision.model_validate(values)


def test_active_revision_requires_and_preserves_governance_metadata() -> None:
    revision = _revision()

    assert revision.source_page_id == "page-123"
    assert revision.source_url == "https://confluence.example/pages/page-123"
    assert revision.source_revision == "42"
    assert revision.chunks[0].chunk_id == "page-123:chunk:0"
    assert revision.source_access.classification == "internal"
    assert revision.embedding_model == "deterministic-hash"
    assert revision.embedding_version == "1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_page_id", ""),
        ("source_url", ""),
        ("source_revision", ""),
        ("chunks", []),
        ("source_access", None),
        ("embedding_model", ""),
        ("embedding_version", ""),
    ],
)
def test_active_revision_rejects_missing_required_sync_metadata(
    field: str, value: object
) -> None:
    with pytest.raises((ValidationError, EvidenceRevisionValidationError)):
        _revision(**{field: value})


def test_active_revision_rejects_missing_chunk_provenance() -> None:
    with pytest.raises((ValidationError, EvidenceRevisionValidationError)):
        _revision(
            chunks=[
                ConfluenceEvidenceChunk(
                    chunk_id="",
                    chunk_index=0,
                    text="content",
                )
            ]
        )


@pytest.mark.parametrize(
    "lifecycle_state",
    [EvidenceLifecycleState.DELETED, EvidenceLifecycleState.INACCESSIBLE],
)
def test_tombstone_revision_carries_governance_metadata_without_content(
    lifecycle_state: EvidenceLifecycleState,
) -> None:
    revision = _revision(
        lifecycle_state=lifecycle_state,
        chunks=[],
    )

    assert revision.lifecycle_state is lifecycle_state
    assert revision.chunks == []
    assert revision.source_access.policy_expires_at.year == 2099

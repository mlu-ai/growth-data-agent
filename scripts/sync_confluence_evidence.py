"""Backfill the synthetic Confluence evidence source into external Qdrant."""

from __future__ import annotations

import json

from growth_data_agent.evidence_sync import (
    QdrantConfigurationError,
    QdrantEvidenceSynchronizer,
    qdrant_client_from_environment,
    qdrant_collection_from_environment,
)
from growth_data_agent.synthetic import SyntheticConfluenceEvidenceSource


def sync_synthetic_confluence_evidence() -> dict[str, int]:
    """Synchronize the fixture source through the same contract as live sources."""
    result = QdrantEvidenceSynchronizer(
        client=qdrant_client_from_environment(),
        collection_name=qdrant_collection_from_environment(),
    ).sync(SyntheticConfluenceEvidenceSource())
    return result.model_dump()


if __name__ == "__main__":
    try:
        print(json.dumps(sync_synthetic_confluence_evidence(), sort_keys=True))
    except QdrantConfigurationError as error:
        raise SystemExit(str(error)) from error

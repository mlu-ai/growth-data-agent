"""Publish the last successfully validated dbt artifact to DataHub GMS."""

from __future__ import annotations

import os
from pathlib import Path

from growth_data_agent.datahub import DataHubHttpTransport, DataHubMetadataPublisher
from growth_data_agent.semantic import SemanticArtifactStore


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact_path = Path(
        os.environ.get(
            "SEMANTIC_ARTIFACT_PATH",
            repository_root / "dbt/artifacts/last_validated_semantic.json",
        )
    )
    gms_url = os.environ.get("DATAHUB_GMS_URL", "http://127.0.0.1:8080")
    artifact = SemanticArtifactStore(artifact_path).load()
    result = DataHubMetadataPublisher(
        DataHubHttpTransport(gms_url, token=os.environ.get("DATAHUB_TOKEN")),
        platform=os.environ.get("DATAHUB_TARGET_PLATFORM", "postgres"),
        dataset_prefix=os.environ.get("DATAHUB_DATASET_PREFIX", "growth_data.analytics"),
    ).publish(artifact)
    print(
        f"Published {result.published_entity_count} DataHub entities for "
        f"semantic version {result.semantic_version}."
    )


if __name__ == "__main__":
    main()

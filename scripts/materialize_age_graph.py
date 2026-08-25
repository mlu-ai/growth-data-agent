"""Refresh the Apache AGE evidence index from validated metadata and documents."""

from __future__ import annotations

import os
from pathlib import Path

from growth_data_agent.datahub import (
    DataHubCatalogUnavailableError,
    DataHubHttpCatalog,
    validated_datahub_metadata,
)
from growth_data_agent.graph import (
    ApacheAgeEvidenceGraphMaterializer,
    PsycopgAgeGraphMutationExecutor,
    apache_age_preloaded_from_environment,
)
from growth_data_agent.semantic import SemanticArtifactStore
from growth_data_agent.synthetic import evidence_corpus


def _approved_catalog_entities(artifact):
    """Read published DataHub metadata when GMS is configured, else stay local-first."""
    validated_entities = validated_datahub_metadata(artifact)
    gms_url = os.environ.get("DATAHUB_GMS_URL")
    if not gms_url:
        return validated_entities
    catalog = DataHubHttpCatalog(
        gms_url,
        token=os.environ.get("DATAHUB_TOKEN"),
        platform=os.environ.get("DATAHUB_TARGET_PLATFORM", "postgres"),
        dataset_prefix=os.environ.get("DATAHUB_DATASET_PREFIX", "growth_data.analytics"),
    )
    try:
        published_entities = tuple(
            catalog.get(entity.entity_name) for entity in validated_entities
        )
    except DataHubCatalogUnavailableError as error:
        raise RuntimeError(
            "Published DataHub metadata is unavailable for AGE materialization."
        ) from error
    if any(entity is None for entity in published_entities):
        raise RuntimeError("Published DataHub metadata is incomplete for AGE materialization.")
    return tuple(entity for entity in published_entities if entity is not None)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact_path = Path(
        os.environ.get(
            "SEMANTIC_ARTIFACT_PATH",
            repository_root / "dbt/artifacts/last_validated_semantic.json",
        )
    )
    database_url = os.environ.get("APACHE_AGE_DATABASE_URL")
    if not database_url:
        raise RuntimeError("APACHE_AGE_DATABASE_URL must be configured to refresh AGE.")

    artifact = SemanticArtifactStore(artifact_path).load()
    result = ApacheAgeEvidenceGraphMaterializer(
        PsycopgAgeGraphMutationExecutor(
            database_url,
            graph_name=os.environ.get("APACHE_AGE_GRAPH_NAME", "growth_evidence"),
            age_preloaded=apache_age_preloaded_from_environment(),
        )
    ).replace(_approved_catalog_entities(artifact), evidence_corpus())
    print(
        f"Materialized {result.path_count} evidence chains, {result.node_count} nodes, "
        f"and {result.edge_count} edges."
    )


if __name__ == "__main__":
    main()

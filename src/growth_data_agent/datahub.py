"""DataHub catalog publication and lookup boundaries.

DataHub enriches the validated dbt semantic authority with ownership,
classification, and discovery metadata. It is deliberately kept behind a
small protocol so catalog outages cannot affect canonical metric execution.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


class DataHubCatalogUnavailableError(RuntimeError):
    """Raised when DataHub cannot answer a catalog request."""


class DataHubEntityMetadata(BaseModel):
    """Published catalog metadata for one validated dbt model or dataset.

    Logical metric names are resolved to their physical ``fct_*`` dbt model
    at the catalog boundary. This keeps one authoritative DataHub Dataset
    identity per dbt model while still supporting metric-name ownership
    questions in the application interface.
    """

    entity_name: str
    entity_type: Literal["metric", "model", "dataset"]
    urn: str
    product: str
    owners: list[str] = Field(min_length=1)
    classification: str
    discovery_tags: list[str] = Field(min_length=1)
    description: str
    semantic_version: str
    source_artifact_sha256: str
    published_at: datetime


class DataHubPublicationResult(BaseModel):
    """Bounded result of publishing one validated semantic artifact."""

    published_entity_count: int = Field(ge=1)
    published_at: datetime
    semantic_version: str
    source_artifact_sha256: str


class DataHubTransport(Protocol):
    def ingest(self, entity: DataHubEntityMetadata) -> None: ...


class DataHubCatalogStore(Protocol):
    def get(self, entity_name: str) -> DataHubEntityMetadata | None: ...


class DataHubMetadataPublisher:
    """Publish validated dbt metadata onto existing DataHub Dataset entities."""

    def __init__(
        self,
        transport: DataHubTransport,
        *,
        owner: str = "growth-data",
        platform: str = "postgres",
        dataset_prefix: str = "growth_data.analytics",
    ):
        self.transport = transport
        self.owner = owner
        self.platform = platform
        self.dataset_prefix = dataset_prefix

    def publish(self, artifact) -> DataHubPublicationResult:
        entities = validated_datahub_metadata(
            artifact,
            owner=self.owner,
            platform=self.platform,
            dataset_prefix=self.dataset_prefix,
        )
        for entity in entities:
            self.transport.ingest(entity)
        published_at = artifact.validation.validated_at.astimezone(UTC)
        return DataHubPublicationResult(
            published_entity_count=len(entities),
            published_at=published_at,
            semantic_version=artifact.semantic_version,
            source_artifact_sha256=artifact.semantic_manifest_sha256,
        )


def validated_datahub_metadata(
    artifact,
    *,
    owner: str = "growth-data",
    platform: str = "postgres",
    dataset_prefix: str = "growth_data.analytics",
) -> tuple[DataHubEntityMetadata, ...]:
    """Map only a successful validated artifact into catalog metadata."""
    if artifact.validation.status != "success":
        raise ValueError("Only successfully validated dbt artifacts may be published.")
    return build_datahub_metadata(
        artifact,
        owner=owner,
        platform=platform,
        dataset_prefix=dataset_prefix,
    )


class InMemoryDataHubCatalog:
    """Deterministic catalog used by the local POC and its public seam tests."""

    def __init__(self, entities: Iterable[DataHubEntityMetadata], *, available: bool = True):
        self._entities = {entity.entity_name: entity for entity in entities}
        for entity in tuple(self._entities.values()):
            if entity.entity_type == "model":
                metric_name = _metric_name_for_model(entity.entity_name)
                if metric_name is not None:
                    self._entities[metric_name] = entity
        self.available = available

    @classmethod
    def from_artifact(
        cls,
        artifact,
        *,
        owner: str = "growth-data",
        platform: str = "postgres",
        dataset_prefix: str = "growth_data.analytics",
    ) -> InMemoryDataHubCatalog:
        return cls(
            validated_datahub_metadata(
                artifact,
                owner=owner,
                platform=platform,
                dataset_prefix=dataset_prefix,
            )
        )

    def get(self, entity_name: str) -> DataHubEntityMetadata | None:
        if not self.available:
            raise DataHubCatalogUnavailableError("DataHub catalog is unavailable.")
        return self._entities.get(entity_name)


class DataHubHttpCatalog:
    """Read ownership, tags, and dataset properties through DataHub GraphQL."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 5.0,
        platform: str = "postgres",
        dataset_prefix: str = "growth_data.analytics",
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.platform = platform
        self.dataset_prefix = dataset_prefix

    def get(self, entity_name: str) -> DataHubEntityMetadata | None:
        model_name = _model_name_for_entity(entity_name)
        entity_urn = _urn(
            "model",
            model_name,
            platform=self.platform,
            dataset_prefix=self.dataset_prefix,
        )
        query = """
        query CatalogEntity($urn: String!) {
          dataset(urn: $urn) {
            urn
            properties { description customProperties }
            ownership { owners { owner { urn } } }
            tags { tags { tag { name urn } } }
          }
        }
        """
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        try:
            response = _request_json(
                f"{self.base_url}/api/graphql",
                method="POST",
                payload={"query": query, "variables": {"urn": entity_urn}},
                headers=headers,
                timeout=self.timeout,
            )
            if response.get("errors"):
                raise ValueError("DataHub GraphQL returned errors.")
            dataset = response.get("data", {}).get("dataset")
            if not dataset:
                return None
            properties = dataset.get("properties") or {}
            custom_properties = properties.get("customProperties") or {}
            owner_urns = [
                item.get("owner", {}).get("urn")
                for item in (dataset.get("ownership", {}).get("owners") or [])
            ]
            owners = [owner.rsplit(":", 1)[-1] for owner in owner_urns if owner]
            tags = dataset.get("tags", {}).get("tags") or []
            discovery_tags = [
                tag.get("tag", {}).get("name")
                for tag in tags
                if tag.get("tag", {}).get("name")
            ]
            classification = custom_properties.get("classification")
            description = properties.get("description")
            semantic_version = custom_properties.get("semantic_version")
            source_artifact_sha256 = custom_properties.get("source_artifact_sha256")
            published_at_value = custom_properties.get("published_at")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (
                    classification,
                    description,
                    semantic_version,
                    source_artifact_sha256,
                    published_at_value,
                )
            ) or not owners or not discovery_tags:
                raise ValueError("DataHub metadata is incomplete.")
            return DataHubEntityMetadata(
                entity_name=model_name,
                entity_type="model",
                urn=dataset.get("urn", entity_urn),
                product=_product_for_name(model_name),
                owners=owners,
                classification=classification,
                discovery_tags=discovery_tags,
                description=description,
                semantic_version=semantic_version,
                source_artifact_sha256=source_artifact_sha256,
                published_at=_parse_datetime(published_at_value),
            )
        except (HTTPError, URLError, TimeoutError, ValueError, AttributeError, TypeError) as error:
            raise DataHubCatalogUnavailableError("DataHub GMS is unavailable.") from error


class DataHubHttpTransport:
    """Small standard-library transport for the DataHub GMS ingest endpoint."""

    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def ingest(self, entity: DataHubEntityMetadata) -> None:
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        try:
            properties = {
                "description": entity.description,
                "customProperties": {
                    "classification": entity.classification,
                    "semantic_version": entity.semantic_version,
                    "source_artifact_sha256": entity.source_artifact_sha256,
                    "published_at": entity.published_at.astimezone(UTC).isoformat(),
                },
            }
            aspects = (
                ("datasetProperties", properties),
                (
                    "ownership",
                    {
                        "owners": [
                            {"owner": _owner_urn(owner), "type": "TECHNICAL_OWNER"}
                            for owner in entity.owners
                        ]
                    },
                ),
                (
                    "globalTags",
                    {
                        "tags": [
                            {"tag": _tag_urn(tag)} for tag in entity.discovery_tags
                        ]
                    },
                ),
            )
            if entity.entity_type == "model":
                aspects += (("subTypes", {"typeNames": ["Model"]}),)
            for aspect_name, aspect in aspects:
                _request_json(
                    f"{self.base_url}/aspects?action=ingestProposal",
                    method="POST",
                    payload={
                        "proposal": {
                            "entityType": "dataset",
                            "entityUrn": entity.urn,
                            "aspectName": aspect_name,
                            "changeType": "UPSERT",
                            "aspect": {
                                "value": json.dumps(aspect),
                                "contentType": "application/json",
                            },
                        }
                    },
                    headers=headers,
                    timeout=self.timeout,
                )
        except (HTTPError, URLError, TimeoutError, ValueError, TypeError) as error:
            raise DataHubCatalogUnavailableError("DataHub GMS is unavailable.") from error


def build_datahub_metadata(
    artifact,
    *,
    owner: str = "growth-data",
    platform: str = "postgres",
    dataset_prefix: str = "growth_data.analytics",
) -> tuple[DataHubEntityMetadata, ...]:
    """Map only validated dbt artifact fields into catalog metadata."""
    published_at = artifact.validation.validated_at.astimezone(UTC)
    entities: list[DataHubEntityMetadata] = []
    for metric in artifact.metrics:
        product = _product_for_name(metric.name)
        entities.append(
            DataHubEntityMetadata(
                entity_name=metric.model_name,
                entity_type="model",
                urn=_urn(
                    "model",
                    metric.model_name,
                    platform=platform,
                    dataset_prefix=dataset_prefix,
                ),
                product=product,
                owners=[owner],
                classification="internal",
                discovery_tags=[
                    "dbt-model",
                    "canonical-metric",
                    f"product:{product.casefold()}",
                ],
                description=metric.definition,
                semantic_version=artifact.semantic_version,
                source_artifact_sha256=artifact.semantic_manifest_sha256,
                published_at=published_at,
            )
        )
    return tuple(entities)


def _model_name_for_entity(entity_name: str) -> str:
    """Resolve a logical metric or model request to its dbt model name."""
    return entity_name if entity_name.casefold().startswith("fct_") else f"fct_{entity_name}"


def _metric_name_for_model(entity_name: str) -> str | None:
    """Return the logical metric alias for a conventional ``fct_*`` model."""
    if not entity_name.casefold().startswith("fct_"):
        return None
    return entity_name[4:]


def _product_for_name(name: str) -> str:
    if name.casefold().startswith("jira") or "jira" in name.casefold():
        return "Jira"
    if name.casefold().startswith("confluence") or "confluence" in name.casefold():
        return "Confluence"
    return "Growth Data"


def _owner_urn(owner: str) -> str:
    return f"urn:li:corpuser:{owner}"


def _tag_urn(tag: str) -> str:
    return f"urn:li:tag:{tag.replace(':', '-')}"


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        raise ValueError("DataHub metadata is missing published_at.")
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as error:
        raise ValueError("DataHub metadata has an invalid published_at.") from error


def _request_json(
    url: str,
    *,
    method: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise DataHubCatalogUnavailableError(
                f"DataHub GMS rejected the request ({response.status})."
            )
        raw = response.read()
    if not raw:
        return {}
    return json.loads(raw)


def _urn(
    entity_type: str,
    entity_name: str,
    *,
    platform: str = "postgres",
    dataset_prefix: str = "growth_data.analytics",
) -> str:
    del entity_type
    dataset_name = _model_name_for_entity(entity_name)
    if dataset_prefix:
        dataset_name = f"{dataset_prefix}.{dataset_name}"
    return f"urn:li:dataset:(urn:li:dataPlatform:{quote(platform)},{quote(dataset_name)},PROD)"

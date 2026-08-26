from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from conftest import RecordingMetricFlowPlanner, RecordingPostgresExecutor, write_artifact
from fastapi.testclient import TestClient

from growth_data_agent.datahub import (
    DataHubCatalogUnavailableError,
    DataHubEntityMetadata,
    DataHubHttpCatalog,
    DataHubHttpTransport,
    DataHubMetadataPublisher,
    InMemoryDataHubCatalog,
)
from growth_data_agent.evidence import EvidenceAccessFilter
from growth_data_agent.graph import (
    ApacheAgeEvidenceGraphMaterializer,
    ApacheAgeEvidenceGraphStore,
    EvidenceGraphUnavailableError,
    GraphAccessFilter,
    GraphNode,
    GraphPath,
    InMemoryEvidenceGraphStore,
    PsycopgAgeGraphMutationExecutor,
    PsycopgAgeGraphQueryExecutor,
    apache_age_preloaded_from_environment,
)
from growth_data_agent.main import create_app
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService
from growth_data_agent.synthetic import evidence_corpus, graph_corpus


class RecordingDataHubTransport:
    def __init__(self) -> None:
        self.entities = []

    def ingest(self, entity):
        self.entities.append(entity)


class RecordingAgeQueryExecutor:
    def __init__(self, paths: list[GraphPath]) -> None:
        self.paths = paths
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, cypher: str, parameters: dict[str, object]) -> list[GraphPath]:
        self.calls.append((cypher, parameters))
        return self.paths


class RecordingAgeMutationExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, cypher: str, parameters: dict[str, object]) -> None:
        self.calls.append((cypher, parameters))


class FakeAgeCursor:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls: list[object] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, parameters=None, **kwargs):
        self.calls.append(query)
        self.query = query
        self.parameters = parameters
        self.prepare = kwargs.get("prepare", False)

    def fetchall(self):
        return [(self.value,)]

    def fetchone(self):
        return None


class FakeAgeConnection:
    def __init__(self, value: str) -> None:
        self.cursor_value = value
        self.execute_calls: list[str] = []
        self.cursors: list[FakeAgeCursor] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def transaction(self):
        return self

    def execute(self, query):
        self.execute_calls.append(query)
        self.last_execute = query

    def cursor(self):
        cursor = FakeAgeCursor(self.cursor_value)
        self.cursors.append(cursor)
        return cursor


class DeniedLoadAgeConnection(FakeAgeConnection):
    def execute(self, query):
        self.execute_calls.append(query)
        if query == "LOAD 'age'":
            raise psycopg.errors.InsufficientPrivilege("access to library 'age' is not allowed")
        self.last_execute = query


class SyntaxErrorAgeCursor(FakeAgeCursor):
    def execute(self, query, parameters=None, **kwargs):
        raise psycopg.errors.SyntaxError("syntax error at or near FOREACH")


class SyntaxErrorAgeConnection(FakeAgeConnection):
    def cursor(self):
        cursor = SyntaxErrorAgeCursor(self.cursor_value)
        self.cursors.append(cursor)
        return cursor


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class UnavailableDataHubCatalog:
    def get(self, entity_name: str):
        raise DataHubCatalogUnavailableError("DataHub GMS is unavailable.")


class RecordingDataHubCatalog:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, entity_name: str):
        self.calls += 1
        return None


class RestrictedDataHubCatalog:
    def get(self, entity_name: str):
        return DataHubEntityMetadata(
            entity_name=entity_name,
            entity_type="metric",
            urn="urn:li:dataset:(urn:li:dataPlatform:dbt,metric/restricted,PROD)",
            product="Jira",
            owners=["restricted-team"],
            classification="restricted",
            discovery_tags=["restricted"],
            description="Restricted catalog metadata.",
            semantic_version="1.0.0",
            source_artifact_sha256="restricted",
            published_at="2026-08-25T00:00:00Z",
        )


def _gateway(tmp_path: Path) -> ValidatedMetricFlowGateway:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    return ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=RecordingPostgresExecutor(),
    )


def test_validated_dbt_metadata_is_publishable_for_catalog_ownership_and_discovery(
    tmp_path: Path,
) -> None:
    artifact = _gateway(tmp_path).artifact_store.load()
    transport = RecordingDataHubTransport()

    result = DataHubMetadataPublisher(transport).publish(artifact)

    assert result.published_entity_count == 4
    assert {entity.entity_name for entity in transport.entities} == {
        "fct_jira_new_peu",
        "fct_jira_new_mau",
        "fct_confluence_new_peu",
        "fct_confluence_new_mau",
    }
    model = next(
        entity for entity in transport.entities if entity.entity_name == "fct_jira_new_peu"
    )
    assert model.entity_type == "model"
    assert model.owners == ["growth-data"]
    assert model.classification == "internal"
    assert "canonical-metric" in model.discovery_tags
    assert model.source_artifact_sha256 == artifact.semantic_manifest_sha256
    assert len({entity.urn for entity in transport.entities}) == 4
    assert "dataPlatform:postgres" in model.urn


def test_datahub_catalog_resolves_metric_names_to_their_physical_dbt_models(
    tmp_path: Path,
) -> None:
    artifact = _gateway(tmp_path).artifact_store.load()
    catalog = InMemoryDataHubCatalog.from_artifact(artifact)

    metadata = catalog.get("jira_new_peu")

    assert metadata is not None
    assert metadata.entity_name == "fct_jira_new_peu"
    assert metadata.entity_type == "model"
    assert metadata.urn.endswith("growth_data.analytics.fct_jira_new_peu,PROD)")


def test_datahub_http_transport_emits_dataset_governance_aspects(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return FakeHttpResponse({})

    monkeypatch.setattr("growth_data_agent.datahub.urlopen", fake_urlopen)
    DataHubHttpTransport("http://datahub", token="secret").ingest(
        DataHubEntityMetadata(
            entity_name="fct_jira_new_peu",
            entity_type="model",
            urn="urn:li:dataset:(urn:li:dataPlatform:dbt,model/fct_jira_new_peu,PROD)",
            product="Jira",
            owners=["growth-data"],
            classification="internal",
            discovery_tags=["dbt-model"],
            description="Jira model.",
            semantic_version="1.0.0",
            source_artifact_sha256="artifact",
            published_at="2026-08-25T00:00:00Z",
        )
    )

    payloads = [json.loads(request.data)["proposal"] for request in requests]
    assert len(payloads) == 4
    assert {payload["entityType"] for payload in payloads} == {"dataset"}
    assert {payload["aspectName"] for payload in payloads} == {
        "datasetProperties",
        "ownership",
        "globalTags",
        "subTypes",
    }
    assert all(payload["aspect"]["contentType"] == "application/json" for payload in payloads)
    assert all(request.headers["Authorization"] == "Bearer secret" for request in requests)


def test_datahub_graphql_errors_are_reported_as_catalog_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "growth_data_agent.datahub.urlopen",
        lambda request, timeout: FakeHttpResponse({"errors": [{"message": "down"}]}),
    )

    with pytest.raises(DataHubCatalogUnavailableError):
        DataHubHttpCatalog("http://datahub").get("jira_new_peu")


def test_datahub_http_catalog_reads_complete_governance_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "growth_data_agent.datahub.urlopen",
        lambda request, timeout: FakeHttpResponse(
            {
                "data": {
                    "dataset": {
                        "urn": (
                            "urn:li:dataset:(urn:li:dataPlatform:postgres,"
                            "growth_data.analytics.fct_jira_new_peu,PROD)"
                        ),
                        "properties": {
                            "description": "Jira New PEU.",
                            "customProperties": {
                                "classification": "internal",
                                "semantic_version": "1.0.0",
                                "source_artifact_sha256": "artifact",
                                "published_at": "2026-08-25T00:00:00Z",
                            },
                        },
                        "ownership": {
                            "owners": [{"owner": {"urn": "urn:li:corpuser:growth-data"}}]
                        },
                        "tags": {"tags": [{"tag": {"name": "canonical-metric"}}]},
                    }
                }
            }
        ),
    )

    metadata = DataHubHttpCatalog("http://datahub").get("jira_new_peu")

    assert metadata is not None
    assert metadata.entity_name == "fct_jira_new_peu"
    assert metadata.entity_type == "model"
    assert metadata.owners == ["growth-data"]
    assert metadata.discovery_tags == ["canonical-metric"]
    assert metadata.classification == "internal"
    assert metadata.source_artifact_sha256 == "artifact"


def test_incomplete_datahub_metadata_is_treated_as_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "growth_data_agent.datahub.urlopen",
        lambda request, timeout: FakeHttpResponse(
            {
                "data": {
                    "dataset": {
                        "properties": {"description": "Jira New PEU.", "customProperties": {}},
                        "ownership": {"owners": []},
                        "tags": {"tags": []},
                    }
                }
            }
        ),
    )

    with pytest.raises(DataHubCatalogUnavailableError):
        DataHubHttpCatalog("http://datahub").get("jira_new_peu")


def test_datahub_publisher_rejects_an_unvalidated_artifact(tmp_path: Path) -> None:
    artifact = SemanticArtifactStore(
        write_artifact(tmp_path / "failed.json", status="failed")
    ).load()

    with pytest.raises(ValueError, match="successfully validated"):
        DataHubMetadataPublisher(RecordingDataHubTransport()).publish(artifact)


def test_datahub_ownership_answers_use_published_catalog_metadata(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    catalog = InMemoryDataHubCatalog.from_artifact(gateway.artifact_store.load())
    client = TestClient(
        create_app(AnswerQuestionService(gateway, catalog_store=catalog))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns the Jira New PEU metric?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "catalog_ownership"
    assert body["catalog_metadata"]["entity_name"] == "jira_new_peu"
    assert body["catalog_metadata"]["owners"] == ["growth-data"]
    assert body["catalog_metadata"]["classification"] == "internal"
    assert "canonical-metric" in body["catalog_metadata"]["discovery_tags"]
    assert body["catalog_freshness"]["available"] is True
    assert body["catalog_freshness"]["degraded"] is False


def test_datahub_unavailability_degrades_catalog_answers_but_not_canonical_metrics(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    client = TestClient(
        create_app(AnswerQuestionService(gateway, catalog_store=UnavailableDataHubCatalog()))
    )

    ownership = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns the Jira New PEU metric?",
        },
    )
    canonical = client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    assert ownership.status_code == 200
    assert ownership.json()["result_classification"] == "limitation"
    assert ownership.json()["catalog_freshness"] == {
        "available": False,
        "degraded": True,
        "detail": "DataHub GMS is unavailable.",
    }
    assert "degraded" in ownership.json()["answer"].casefold()
    assert canonical.status_code == 200
    assert canonical.json()["result_classification"] == "canonical_definition"
    assert canonical.json()["canonical_definition"]["name"] == "jira_new_peu"
    assert "DataHub catalog availability does not affect canonical metric logic" in (
        " ".join(canonical.json()["caveats"])
    )


def test_unconfigured_datahub_is_disclosed_as_degraded(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    client = TestClient(create_app(AnswerQuestionService(gateway)))

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns the Jira New PEU metric?",
        },
    )

    assert response.status_code == 200
    assert response.json()["catalog_freshness"]["available"] is False
    assert response.json()["catalog_freshness"]["degraded"] is True


def test_unknown_catalog_scope_is_refused_before_catalog_lookup(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    catalog = RecordingDataHubCatalog()
    client = TestClient(
        create_app(AnswerQuestionService(gateway, catalog_store=catalog))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns this dataset?",
            "requested_metric_name": "unscoped_dataset",
        },
    )

    assert response.status_code == 403
    assert catalog.calls == 0


def test_unpublished_product_scoped_entity_is_refused_before_catalog_lookup(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    catalog = RecordingDataHubCatalog()
    client = TestClient(create_app(AnswerQuestionService(gateway, catalog_store=catalog)))

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns this dataset?",
            "requested_metric_name": "jira_unpublished_dataset",
        },
    )

    assert response.status_code == 403
    assert catalog.calls == 0


def test_inaccessible_catalog_classification_is_not_disclosed(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    client = TestClient(
        create_app(AnswerQuestionService(gateway, catalog_store=RestrictedDataHubCatalog()))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "Who owns the Jira New PEU metric?",
        },
    )

    assert response.status_code == 403
    assert "restricted" not in response.text.casefold()


def test_apache_age_query_is_parameterized_with_the_pre_authorized_scope() -> None:
    allowed_path = GraphPath(
        path_id="jira-apac-chain",
        nodes=[
            GraphNode(
                node_id="jira_new_peu",
                node_type="metric",
                label="Jira New PEU",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
                graph_namespace="growth-data-agent",
            ),
            GraphNode(
                node_id="segment-jira-apac",
                node_type="segment",
                label="APAC 51-200 Seat Tier Tenants",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
                graph_namespace="growth-data-agent",
            ),
            GraphNode(
                node_id="tenant-jira-apac",
                node_type="tenant",
                label="APAC Tenant cohort",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
                graph_namespace="growth-data-agent",
            ),
            GraphNode(
                node_id="incident-jira-apac",
                node_type="incident",
                label="Jira APAC incident",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
                graph_namespace="growth-data-agent",
            ),
        ],
    )
    executor = RecordingAgeQueryExecutor([allowed_path])
    store = ApacheAgeEvidenceGraphStore(executor)
    access_filter = GraphAccessFilter(
        products=("Jira",),
        regions=("APAC",),
        tenant_ids=("tenant-0002",),
        classifications=("internal",),
        identifier_entitlements=("none",),
    )

    paths = store.traverse(
        "Jira APAC incident",
        access_filter,
        limit=3,
        metric_name="jira_new_peu",
    )

    assert paths == [allowed_path]
    assert len(executor.calls) == 1
    cypher, parameters = executor.calls[0]
    assert "[:EVIDENCE_CHAIN*3..4]" in cypher
    assert "$tenant_ids" in cypher
    assert "$classifications" in cypher
    assert "tenant-0002" not in cypher
    assert parameters["tenant_ids"] == ["tenant-0002"]
    assert parameters["products"] == ["Jira"]
    assert parameters["query"] == "Jira APAC incident"
    assert parameters["metric_name"] == "jira_new_peu"
    assert parameters["graph_namespace"] == "growth-data-agent"
    assert "graph_namespace" in cypher


def test_age_post_filter_rejects_a_chain_for_the_wrong_metric() -> None:
    wrong_metric = GraphPath(
        path_id="wrong-metric-chain",
        nodes=[
            GraphNode(
                node_id="confluence_new_peu",
                node_type="metric",
                label="Confluence New PEU",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
            ),
            GraphNode(
                node_id="segment",
                node_type="segment",
                label="APAC 51-200 Seat Tier Tenants",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
            ),
            GraphNode(
                node_id="tenant",
                node_type="tenant",
                label="APAC Tenant cohort",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
            ),
            GraphNode(
                node_id="incident",
                node_type="incident",
                label="Jira APAC incident",
                product="Jira",
                region="APAC",
                tenant_ids=["tenant-0002"],
                classification="internal",
                identifier_entitlement="none",
            ),
        ],
    )
    executor = RecordingAgeQueryExecutor([wrong_metric])
    store = ApacheAgeEvidenceGraphStore(executor)
    access_filter = GraphAccessFilter(
        products=("Jira",),
        regions=("APAC",),
        tenant_ids=("tenant-0002",),
        classifications=("internal",),
        identifier_entitlements=("none",),
    )

    assert (
        store.traverse(
            "Jira APAC incident",
            access_filter,
            limit=3,
            metric_name="jira_new_peu",
        )
        == []
    )


def test_age_traversal_fails_closed_for_empty_tenant_scope() -> None:
    executor = RecordingAgeQueryExecutor([])
    store = ApacheAgeEvidenceGraphStore(executor)
    access_filter = GraphAccessFilter(
        products=("Jira",),
        regions=("APAC",),
        tenant_ids=(),
        classifications=("internal",),
        identifier_entitlements=("none",),
    )

    assert (
        store.traverse(
            "Jira APAC incident",
            access_filter,
            limit=3,
            metric_name="jira_new_peu",
        )
        == []
    )
    assert executor.calls == []


def test_psycopg_age_executor_decodes_vertex_edge_agtype(monkeypatch) -> None:
    common_properties = {
        "product": "Jira",
        "region": "APAC",
        "tenant_ids": ["tenant-0002"],
        "classification": "internal",
        "identifier_entitlement": "none",
        "seat_tiers": ["51-200"],
    }
    path_elements: list[str] = []
    for node_id, node_type in (("metric", "metric"), ("segment", "segment"),
                               ("tenant", "tenant"), ("incident", "incident")):
        path_elements.append(
            json.dumps(
                {
                    "id": len(path_elements) * 2 + 1,
                    "label": node_type,
                    "properties": {
                        **common_properties,
                        "node_id": node_id,
                        "node_type": node_type,
                    },
                }
            )
            + "::vertex"
        )
        if node_type != "incident":
            start_id = len(path_elements) * 2 - 1
            path_elements.append(
                json.dumps(
                    {
                        "id": start_id + 1,
                        "label": "edge",
                        "startid": start_id,
                        "endid": start_id + 2,
                        "properties": {},
                    }
                )
                + "::edge"
            )
    age_path = "[" + ",".join(path_elements) + "]::path"
    connections: list[FakeAgeConnection] = []

    def connect(_database_url: str) -> FakeAgeConnection:
        connection = FakeAgeConnection(age_path)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "growth_data_agent.graph.psycopg.connect",
        connect,
    )

    paths = PsycopgAgeGraphQueryExecutor(
        "postgresql://example", age_preloaded=True
    ).query(
        "MATCH path RETURN path", {"tenant_ids": ["tenant-0002"]}
    )

    assert [node.node_type for node in paths[0].nodes] == [
        "metric",
        "segment",
        "tenant",
        "incident",
    ]
    assert connections[0].execute_calls == [
        "SET TRANSACTION READ ONLY",
        'SET search_path = ag_catalog, "$user", public',
    ]
    cypher_statement = connections[0].cursors[0].query.as_string(None)
    assert "$gda$MATCH path RETURN path$gda$" in cypher_statement
    assert "cypher(%s" not in cypher_statement
    assert connections[0].cursors[0].parameters == ('{"tenant_ids": ["tenant-0002"]}',)
    assert connections[0].cursors[0].prepare


def test_psycopg_age_executor_reports_preload_configuration_when_load_is_denied(
    monkeypatch,
) -> None:
    connection = DeniedLoadAgeConnection("not-used")
    monkeypatch.setattr(
        "growth_data_agent.graph.psycopg.connect",
        lambda database_url: connection,
    )

    with pytest.raises(EvidenceGraphUnavailableError, match="session_preload_libraries"):
        PsycopgAgeGraphQueryExecutor("postgresql://example").query(
            "MATCH path RETURN path", {}
        )

    assert connection.execute_calls == ["SET TRANSACTION READ ONLY", "LOAD 'age'"]


def test_psycopg_age_executor_does_not_label_cypher_syntax_as_preload_failure(
    monkeypatch,
) -> None:
    connection = SyntaxErrorAgeConnection("not-used")
    monkeypatch.setattr(
        "growth_data_agent.graph.psycopg.connect",
        lambda database_url: connection,
    )

    with pytest.raises(EvidenceGraphUnavailableError, match="Cypher statement") as error:
        PsycopgAgeGraphQueryExecutor(
            "postgresql://example", age_preloaded=True
        ).query("FOREACH", {})

    assert "session_preload_libraries" not in str(error.value)


def test_age_preloaded_environment_flag_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("APACHE_AGE_PRELOADED", raising=False)
    assert not apache_age_preloaded_from_environment()

    monkeypatch.setenv("APACHE_AGE_PRELOADED", "true")
    assert apache_age_preloaded_from_environment()


def test_psycopg_age_mutation_executor_skips_load_for_preloaded_non_superuser(
    monkeypatch,
) -> None:
    connection = FakeAgeConnection("ignored")
    monkeypatch.setattr(
        "growth_data_agent.graph.psycopg.connect",
        lambda database_url: connection,
    )

    PsycopgAgeGraphMutationExecutor(
        "postgresql://example", age_preloaded=True
    ).execute("MATCH (n) RETURN n", {})

    assert connection.execute_calls == [
        'SET search_path = ag_catalog, "$user", public',
    ]
    statements = [
        query.as_string(None)
        for cursor in connection.cursors
        for query in cursor.calls
    ]
    assert any("create_graph" in statement for statement in statements)
    assert all("ag_graph" not in statement for statement in statements)


def test_age_graph_name_is_validated_before_sql_composition() -> None:
    with pytest.raises(ValueError, match="APACHE_AGE_GRAPH_NAME"):
        PsycopgAgeGraphQueryExecutor(
            "postgresql://example", graph_name="growth_evidence;DROP TABLE users"
        )


def test_psycopg_age_executor_fails_closed_on_malformed_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "growth_data_agent.graph.psycopg.connect",
        lambda database_url: FakeAgeConnection("not-a-path"),
    )

    with pytest.raises(EvidenceGraphUnavailableError):
        PsycopgAgeGraphQueryExecutor("postgresql://example").query(
            "MATCH path RETURN path", {}
        )


def test_derived_graph_corpus_contains_metric_segment_tenant_incident_or_team_chains() -> None:
    public_paths = [
        path
        for path in graph_corpus()
        if all(node.classification == "internal" for node in path.nodes)
    ]

    assert any(
        [node.node_type for node in path.nodes][:3] == ["metric", "segment", "tenant"]
        and path.nodes[-1].node_type in {"incident", "team"}
        for path in public_paths
    )
    assert {path.nodes[0].node_id for path in public_paths} == {
        "jira_new_peu",
        "confluence_new_peu",
        "confluence_new_mau",
    }


def test_evidence_filters_are_scoped_to_the_requested_metric() -> None:
    document = evidence_corpus()[0]
    access_filter = EvidenceAccessFilter(
        products=(document.product,),
        regions=(document.region,),
        tenant_ids=tuple(document.tenant_ids),
        classifications=(document.classification,),
        identifier_entitlements=(document.identifier_entitlement,),
        metric_names=(document.metric_name,),
    )

    assert access_filter.allows(document)
    assert not access_filter.allows(
        document.model_copy(update={"metric_name": "other_metric"})
    )
    assert not access_filter.allows(document.model_copy(update={"tenant_ids": []}))
    assert any(
        condition.key == "metric_name" for condition in access_filter.as_qdrant_filter().must
    )


def test_in_memory_graph_fallback_is_scoped_to_the_requested_metric() -> None:
    paths = graph_corpus()
    access_filter = GraphAccessFilter(
        products=("Jira", "Confluence"),
        regions=("APAC", "Americas", "EMEA"),
        tenant_ids=tuple(sorted({
            tenant_id
            for path in paths
            for node in path.nodes
            for tenant_id in node.tenant_ids
        })),
        classifications=("internal", "restricted"),
        identifier_entitlements=("none", "direct"),
    )

    result = InMemoryEvidenceGraphStore(paths).traverse(
        "evidence",
        access_filter,
        limit=20,
        metric_name="jira_new_peu",
    )

    assert result
    assert {path.nodes[0].node_id for path in result} == {"jira_new_peu"}


def test_in_memory_graph_fallback_rejects_a_foreign_graph_namespace() -> None:
    foreign_path = graph_corpus()[0].model_copy(deep=True)
    for node in foreign_path.nodes:
        node.graph_namespace = "foreign-graph"
    access_filter = GraphAccessFilter(
        products=(foreign_path.nodes[0].product,),
        regions=(foreign_path.nodes[0].region,),
        tenant_ids=tuple(foreign_path.nodes[0].tenant_ids),
        classifications=("internal", "restricted"),
        identifier_entitlements=("none", "direct"),
    )

    result = InMemoryEvidenceGraphStore([foreign_path]).traverse(
        "evidence",
        access_filter,
        limit=1,
        metric_name=foreign_path.nodes[0].node_id,
    )

    assert result == []


def test_age_materializer_replaces_graph_from_approved_metadata() -> None:
    mutation_executor = RecordingAgeMutationExecutor()
    materializer = ApacheAgeEvidenceGraphMaterializer(mutation_executor)
    catalog_entity = DataHubEntityMetadata(
        entity_name="jira_new_peu",
        entity_type="metric",
        urn="urn:li:dataset:(urn:li:dataPlatform:dbt,metric/jira_new_peu,PROD)",
        product="Jira",
        owners=["growth-data"],
        classification="internal",
        discovery_tags=["canonical-metric"],
        description="Jira New PEU.",
        semantic_version="1.0.0",
        source_artifact_sha256="artifact",
        published_at="2026-08-25T00:00:00Z",
    )

    result = materializer.replace(
        [catalog_entity],
        evidence_corpus()[:1],
    )

    assert result.path_count == 1
    assert result.node_count == 4
    assert result.edge_count == 3
    assert len(mutation_executor.calls) == 3
    clear_cypher, clear_parameters = mutation_executor.calls[0]
    nodes_cypher, nodes_parameters = mutation_executor.calls[1]
    edges_cypher, edges_parameters = mutation_executor.calls[2]
    assert "DETACH DELETE" in clear_cypher
    assert clear_parameters == {"graph_namespace": "growth-data-agent"}
    assert "FOREACH" not in nodes_cypher
    assert len(nodes_parameters["nodes"]) == 4
    assert "UNWIND $edges" in edges_cypher
    assert len(edges_parameters["edges"]) == 3

    empty_executor = RecordingAgeMutationExecutor()
    empty_result = ApacheAgeEvidenceGraphMaterializer(empty_executor).replace([], [])
    assert empty_result.path_count == 0
    assert empty_executor.calls[0][1] == {"graph_namespace": "growth-data-agent"}


def test_graph_filter_is_derived_before_traversal_and_restricted_paths_are_not_returned(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    restricted_path = next(
        path for path in graph_corpus() if "identifier" in path.path_id
    )
    executor = RecordingAgeQueryExecutor([restricted_path])
    graph_store = ApacheAgeEvidenceGraphStore(executor)
    client = TestClient(
        create_app(AnswerQuestionService(gateway, graph_store=graph_store))
    )

    response = client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 200
    assert response.json()["graph_paths"] == []
    assert "tenant-0011" not in response.text
    assert executor.calls[0][1]["regions"] == ["APAC"]
    assert executor.calls[0][1]["identifier_entitlements"] == ["none"]

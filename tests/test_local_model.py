from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from growth_data_agent.contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    EffectiveAccessScope,
    EvidenceAnswer,
    EvidenceCitation,
    EvidenceScope,
    EvidenceSupportStatus,
    GovernedAnalyticalResponse,
    ResultClassification,
    SourceFreshness,
)
from growth_data_agent.local_model import (
    OLLAMA_INTENT_MODEL_NAME,
    LocalModelEvidenceDraftingAdapter,
    LocalModelIntentInterpreter,
    LocalModelOutputInvalid,
    OllamaBaselineModel,
    OllamaIntentModel,
    OllamaLocalModel,
    build_local_model_baseline_context,
)
from growth_data_agent.main import create_app
from growth_data_agent.service import AnswerQuestionService


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({"response": "A governed answer."}).encode()


def test_ollama_local_model_records_non_streaming_generation_request(monkeypatch) -> None:
    requests = []

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    model = OllamaBaselineModel(model_name="qwen3:8b", base_url="http://127.0.0.1:11434")

    output = model.generate({"answer": "Define Jira New PEU"})

    assert output == "A governed answer."
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert json.loads(request.data) == {
        "model": "qwen3:8b",
        "prompt": (
            "Produce a concise answer using only this governed response. "
            "Do not add facts or identifiers that are absent from it.\n"
            '{"answer": "Define Jira New PEU"}'
        ),
        "stream": False,
        "options": {"temperature": 0},
    }
    assert timeout == 60.0


def test_ollama_local_model_rejects_untyped_raw_context_requests() -> None:
    model = OllamaLocalModel(model_name=OLLAMA_INTENT_MODEL_NAME)

    with pytest.raises(LocalModelOutputInvalid):
        model.generate({"governed_context": '{"answer":"Do not bypass the adapter"}'})
    with pytest.raises(LocalModelOutputInvalid):
        model.generate(
            {
                "task": "baseline_evaluation",
                "input": {"governed_context": "Do not bypass the adapter"},
            }
        )


def test_ollama_local_model_rejects_unbounded_structured_input() -> None:
    model = OllamaLocalModel(model_name=OLLAMA_INTENT_MODEL_NAME)

    with pytest.raises(LocalModelOutputInvalid):
        model.generate(
            {
                "task": "intent_proposal",
                "input": {
                    "question": "What is Jira New PEU?",
                    "requested_metric_name": None,
                    "tools": ["sql"],
                },
            }
        )


def test_baseline_context_allowlists_redacted_evidence_fields(monkeypatch) -> None:
    context = build_local_model_baseline_context(
        {
            "answer": "The answer mentions tenant-0001.",
            "result_classification": "hypothesis",
            "source_document_id": "raw-source-id",
            "semantic_query_evidence": {"raw": "retrieval payload"},
            "evidence": {
                "support_status": "supports",
                "support_explanation": "tenant-0001 is mentioned.",
                "citations": [
                    {
                        "document_id": "doc-1",
                        "title": "Safe title",
                        "affected_scope": {
                            "product": "Jira",
                            "region": "APAC",
                            "tenant_scope": "tenant-0001",
                        },
                        "relevant_date": "2026-06-12",
                        "support_status": "supports",
                        "support_explanation": "tenant-0001 is mentioned.",
                        "source_document_id": "raw-source-id",
                        "source_url": "https://private.example/raw",
                        "source_revision": "v1",
                        "chunk_id": "private-chunk",
                        "raw_text": "restricted retrieval text",
                    }
                ],
            },
        }
    )

    encoded_context = json.dumps(context)
    assert set(context) == {"answer", "result_classification", "evidence"}
    assert "source_document_id" not in encoded_context
    assert "source_url" not in encoded_context
    assert "chunk_id" not in encoded_context
    assert "raw_text" not in encoded_context
    assert "tenant-0001" not in encoded_context
    assert "[redacted identifier]" in encoded_context

    requests = []

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    OllamaBaselineModel(model_name="qwen3:8b").generate(context)

    prompt = json.loads(requests[0][0].data)["prompt"]
    assert "source_document_id" not in prompt
    assert "source_url" not in prompt
    assert "chunk_id" not in prompt
    assert "raw_text" not in prompt
    assert "tenant-0001" not in prompt


class RecordingModel:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests = []

    def generate(self, request):
        self.requests.append(
            request.model_dump(mode="json") if isinstance(request, BaseModel) else request
        )
        return self.output


class SequencedModel(RecordingModel):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs[0])
        self.outputs = iter(outputs)

    def generate(self, request):
        self.requests.append(
            request.model_dump(mode="json") if isinstance(request, BaseModel) else request
        )
        return next(self.outputs)


def test_local_model_intent_is_reduced_to_a_deterministic_analytical_intent() -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}')
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu",),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    intent = interpreter.interpret(
        AnswerQuestionRequest(agent_user_id="data_analyst", question="What is Jira New PEU?")
    )

    assert intent == AnalyticalIntent(
        route=AnalyticalRoute.CANONICAL_DEFINITION,
        metric_name="jira_new_peu",
    )
    assert model.requests == [
        {
            "task": "intent_proposal",
            "input": {
                "task": "intent_proposal",
                "question": "What is Jira New PEU?",
                "requested_metric_name": None,
                "available_metric_names": ["jira_new_peu"],
            },
        }
    ]


def test_local_model_intent_accepts_a_paraphrase_from_semantic_candidates() -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}')
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu", "jira_new_mau"),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    intent = interpreter.interpret(
        AnswerQuestionRequest(
            agent_user_id="data_analyst",
            question="How do we define first-time paid Jira access?",
        )
    )

    assert intent == AnalyticalIntent(
        route=AnalyticalRoute.CANONICAL_DEFINITION,
        metric_name="jira_new_peu",
    )
    assert model.requests[0]["input"]["available_metric_names"] == [
        "jira_new_peu",
        "jira_new_mau",
    ]


def test_local_model_intent_rejects_an_ambiguous_candidate_proposal() -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"ambiguous"}')
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu", "jira_new_mau"),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    with pytest.raises(LocalModelOutputInvalid):
        interpreter.interpret(
            AnswerQuestionRequest(
                agent_user_id="data_analyst",
                question="How did paid access change?",
            )
        )


def test_local_model_intent_rejects_a_metric_not_in_the_validated_artifact() -> None:
    model = RecordingModel('{"metric_name":"made_up_metric"}')
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu",),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    with pytest.raises(LocalModelOutputInvalid):
        interpreter.interpret(
            AnswerQuestionRequest(
                agent_user_id="data_analyst",
                question="What is the made-up metric?",
            )
        )


def test_local_model_intent_redacts_identifier_shaped_request_text() -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}')
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu",),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    interpreter.interpret(
        AnswerQuestionRequest(
            agent_user_id="data_analyst",
            question="What is Jira New PEU for tenant-0001?",
        )
    )

    assert "tenant-0001" not in json.dumps(model.requests[0])
    assert "[redacted identifier]" in json.dumps(model.requests[0])


@pytest.mark.parametrize(
    "output",
    [
        "{}",
        '{"metric_name":"jira_new_peu","route":"direct_identifier"}',
        '{"metric_name":"confluence_new_peu"}',
        '{"metric_name":"jira_new_peu","regions":["EMEA"],"tools":["sql"]}',
    ],
)
def test_local_model_intent_cannot_change_route_or_metric_scope(output: str) -> None:
    model = RecordingModel(output)
    interpreter = LocalModelIntentInterpreter(
        model,
        metric_names_provider=lambda request: ("jira_new_peu",),
        route_resolver=lambda request, metric_name: AnalyticalRoute.CANONICAL_DEFINITION,
    )

    with pytest.raises(LocalModelOutputInvalid):
        interpreter.interpret(
            AnswerQuestionRequest(agent_user_id="data_analyst", question="What is Jira New PEU?")
        )


def _evidence_response(*, tenant_scope: str = "APAC 51-200 Seat Tier Tenants"):
    citation = EvidenceCitation(
        document_id="jira-apac-paid-provisioning-incident",
        title="Jira APAC paid provisioning incident",
        affected_scope=EvidenceScope(
            product="Jira",
            region="APAC",
            tenant_scope=tenant_scope,
        ),
        relevant_date="2026-06-12",
        freshness="2026-06-13T00:00:00Z",
        support_status=EvidenceSupportStatus.SUPPORTS,
        support_explanation="The incident overlaps the APAC 51-200 Seat Tier Tenant scope.",
        source_document_id="jira-apac-paid-provisioning-incident",
        source_url="https://evidence.local/synthetic/jira-apac-paid-provisioning-incident",
        source_revision="synthetic-v1",
        chunk_id="jira-apac-paid-provisioning-incident:chunk:0",
    )
    return GovernedAnalyticalResponse(
        answer="The evidence supports a hypothesis.",
        result_classification=ResultClassification.HYPOTHESIS,
        evidence=EvidenceAnswer(
            citations=[citation],
            support_status=EvidenceSupportStatus.SUPPORTS,
            support_explanation="The evidence supports the hypothesis.",
        ),
        source_freshness=SourceFreshness(
            validated_at="2026-08-25T00:00:00Z",
            maximum_age_seconds=86_400,
            is_current=True,
        ),
        effective_access_scope=EffectiveAccessScope(
            products=["Jira", "Confluence"],
            regions=["APAC"],
            tenant_scope="APAC Tenants only",
            permitted_columns=["metric_name"],
        ),
        caveats=[],
        trace_id="trace-id",
    )


def test_local_model_drafting_receives_only_redacted_citations_and_preserves_their_ids() -> None:
    model = RecordingModel(
        '{"answer":"The evidence supports the hypothesis.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}'
    )
    adapter = LocalModelEvidenceDraftingAdapter(model)

    answer = adapter.draft(_evidence_response())

    assert answer.answer == "The evidence supports the hypothesis."
    request = model.requests[0]
    encoded_request = json.dumps(request)
    assert "tenant-" not in encoded_request
    assert "chunk_id" not in encoded_request
    assert "source_url" not in encoded_request
    assert "Jira" in encoded_request
    assert "APAC" in encoded_request
    assert request["input"]["citations"][0]["document_id"] == (
        "jira-apac-paid-provisioning-incident"
    )


def test_local_model_drafting_redacts_identifier_shaped_citation_scope() -> None:
    model = RecordingModel(
        '{"answer":"The evidence supports the hypothesis.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}'
    )

    LocalModelEvidenceDraftingAdapter(model).draft(_evidence_response(tenant_scope="tenant-0001"))

    encoded_request = json.dumps(model.requests[0])
    assert "tenant-0001" not in encoded_request
    assert "[redacted identifier]" in encoded_request


@pytest.mark.parametrize(
    "output",
    [
        '{"answer":"The EMEA incident caused the decline.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"All Tenants were affected.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"The APAC 1-10 Seat Tier movement may explain the decline.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"tenant-0001 was affected.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"The answer is safe.","citation_document_ids":["unknown"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"The incident does not support the hypothesis.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
        '{"answer":"A database migration correlates with revenue growth.",'
        '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
        '"support_status":"supports",'
        '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant scope."]}',
    ],
)
def test_local_model_drafting_rejects_scope_expansion_and_untrusted_citations(output: str) -> None:
    adapter = LocalModelEvidenceDraftingAdapter(RecordingModel(output))

    with pytest.raises(LocalModelOutputInvalid):
        adapter.draft(_evidence_response())


@pytest.mark.parametrize(
    "model_output",
    [
        '{"metric_name":"jira_new_peu","route":"direct_identifier"}',
        '{"metric_name":"jira_new_peu","tools":["direct_identifier"]}',
        '{"metric_name":"jira_new_peu","products":["Confluence"]}',
        '{"metric_name":"jira_new_peu","regions":["EMEA"]}',
    ],
)
def test_configured_local_model_cannot_invoke_an_unallowlisted_route(
    client: TestClient, model_output: str
) -> None:
    model = RecordingModel(model_output)
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What is Jira New PEU?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"
    assert len(model.requests) == 1
    assert base_service.semantic_gateway.postgres_executor.plans == []


def test_configured_local_model_preserves_the_deterministic_authorized_scope(
    client: TestClient,
) -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}')
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What is Jira New PEU?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["effective_access_scope"]["regions"] == ["APAC"]
    assert body["effective_access_scope"]["tenant_scope"] == "APAC Tenants only"
    assert body["semantic_query_evidence"]["constrained_products"] == ["Jira"]
    assert body["semantic_query_evidence"]["constrained_regions"] == ["APAC"]
    assert "agent_user_id" not in json.dumps(model.requests[0])


def test_configured_local_model_routes_a_paraphrased_definition_to_canonical_handler(
    client: TestClient,
) -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}')
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "How is first-time paid access to Jira counted?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "canonical_definition"
    assert body["canonical_definition"]["citation"]["authority"] == "dbt/MetricFlow"
    assert body["canonical_definition"]["name"] == "jira_new_peu"


def test_configured_local_model_clarifies_an_ambiguous_definition_question(
    client: TestClient,
) -> None:
    model = RecordingModel('{"metric_name":"jira_new_peu","ambiguity":"ambiguous"}')
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "How did paid access change?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"
    assert base_service.semantic_gateway.postgres_executor.plans == []


def test_configured_local_model_receives_only_entitled_metric_candidates(
    client: TestClient,
) -> None:
    model = RecordingModel(
        '{"metric_name":"confluence_new_peu","ambiguity":"unambiguous"}'
    )
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "confluence_product_manager",
            "question": "How is first-time paid access to Confluence counted?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "canonical_definition"
    assert model.requests[0]["input"]["available_metric_names"] == [
        "confluence_new_peu",
        "confluence_new_mau",
    ]


def test_configured_local_model_drafts_only_the_authorized_evidence_response(
    client: TestClient,
) -> None:
    model = SequencedModel(
        [
            '{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}',
            '{"answer":"The incident overlaps the APAC 51-200 Seat Tier Tenant scope and '
            'the June 2026 decline period. It supports a possible Hypothesis but does not '
            'establish causation.",'
            '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
            '"support_status":"supports",'
            '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant '
            'scope and the June 2026 decline period. It supports a possible Hypothesis '
            'but does not establish causation."]}',
        ]
    )
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "hypothesis"
    assert body["answer"] == (
        "The incident overlaps the APAC 51-200 Seat Tier Tenant scope and the June 2026 "
        "decline period. It supports a possible Hypothesis but does not establish causation."
    )
    assert body["evidence"]["citations"][0]["document_id"] == (
        "jira-apac-paid-provisioning-incident"
    )
    assert "chunk_id" not in json.dumps(model.requests[1])
    assert "source_url" not in json.dumps(model.requests[1])


@pytest.mark.parametrize(
    "malicious_answer",
    [
        "The Confluence incident affected the APAC decline.",
        "The EMEA incident affected the APAC decline.",
        "All Tenants were affected.",
    ],
)
def test_configured_local_model_rejects_scope_expansion_at_service_boundary(
    client: TestClient, malicious_answer: str
) -> None:
    model = SequencedModel(
        [
            '{"metric_name":"jira_new_peu","ambiguity":"unambiguous"}',
            f'{{"answer":"{malicious_answer}",'
            '"citation_document_ids":["jira-apac-paid-provisioning-incident"],'
            '"support_status":"supports",'
            '"cited_claims":["The incident overlaps the APAC 51-200 Seat Tier Tenant '
            'scope."]}',
        ]
    )
    base_service = client.app.state.answer_service
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=model)
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What evidence may explain the APAC 51–200-seat Tenant decline?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_classification"] == "limitation"
    assert body["evidence"] is None
    assert body["effective_access_scope"]["regions"] == ["APAC"]
    assert body["effective_access_scope"]["tenant_scope"] == "APAC Tenants only"
    assert len(base_service.semantic_gateway.postgres_executor.plans) == 1


def test_unavailable_configured_local_model_fails_closed_before_semantic_query(
    client: TestClient,
) -> None:
    class UnavailableModel:
        def generate(self, request):
            raise OSError("Ollama is not running")

    base_service = client.app.state.answer_service
    executor = base_service.semantic_gateway.postgres_executor
    service = AnswerQuestionService(base_service.semantic_gateway, local_model=UnavailableModel())
    configured_client = TestClient(create_app(service))

    response = configured_client.post(
        "/answer_question",
        json={
            "agent_user_id": "apac_regional_manager",
            "question": "What is Jira New PEU?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "limitation"
    assert executor.plans == []


def test_ollama_configuration_is_opt_in_with_the_agreed_intent_model(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_NAME", raising=False)
    assert OllamaIntentModel.from_environment() is None

    monkeypatch.setenv("OLLAMA_MODEL_NAME", "qwen3:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "12.5")
    model = OllamaIntentModel.from_environment()

    assert model is not None
    assert model.model_name == "qwen3:4b"
    assert model.base_url == "http://127.0.0.1:11434"
    assert model.timeout == 12.5

    monkeypatch.setenv("OLLAMA_MODEL_NAME", "")
    assert OllamaIntentModel.from_environment() is None


def test_ollama_intent_model_requires_the_agreed_model_without_restricting_generic_transport():
    generic_model = OllamaLocalModel(model_name="llama3.1:8b")
    assert generic_model.model_name == "llama3.1:8b"

    with pytest.raises(ValueError, match="qwen3:4b"):
        OllamaIntentModel(model_name="llama3.1:8b")


def test_ollama_intent_provider_does_not_enable_an_unapproved_model(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "llama3.1:8b")

    assert OllamaIntentModel.from_environment() is None


def test_readiness_route_reports_deterministic_mode_when_ollama_is_disabled(client) -> None:
    response = client.get("/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "local_model": {
            "provider": "none",
            "status": "disabled",
            "model": None,
        },
    }


def test_ollama_readiness_reports_unavailable_without_exposing_connection_details(
    monkeypatch,
) -> None:
    def urlopen(request, *, timeout):
        raise OSError("Ollama is not running")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    model = OllamaLocalModel(model_name="qwen3:4b")

    assert model.readiness() == {
        "provider": "ollama",
        "status": "unavailable",
        "model": "qwen3:4b",
    }


def test_readiness_route_reports_the_available_ollama_model(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "qwen3:4b")

    def urlopen(request, *, timeout):
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    response = TestClient(create_app()).get("/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "local_model": {
            "provider": "ollama",
            "status": "ready",
            "model": "qwen3:4b",
        },
    }

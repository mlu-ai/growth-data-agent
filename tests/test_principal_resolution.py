from __future__ import annotations

import secrets

from fastapi.testclient import TestClient
from starlette.testclient import TestClient as RawTestClient

from growth_data_agent.main import create_app
from growth_data_agent.principal import (
    DEVELOPMENT_PRINCIPAL_IDS,
    DevelopmentTokenPrincipalResolver,
    PrincipalAuthenticationError,
    VerifiedPrincipal,
    development_token_environment_variable,
)


def _development_resolver(principal_id: str) -> tuple[str, DevelopmentTokenPrincipalResolver]:
    token = secrets.token_urlsafe(32)
    principal = VerifiedPrincipal(
        principal_id=principal_id,
        issuer="development",
        subject=principal_id,
    )
    return token, DevelopmentTokenPrincipalResolver({token: principal})


def test_development_bearer_token_resolves_to_provider_neutral_principal() -> None:
    token, resolver = _development_resolver("data_analyst")

    principal = resolver.resolve(f"Bearer {token}")

    assert principal == VerifiedPrincipal(
        principal_id="data_analyst",
        issuer="development",
        subject="data_analyst",
    )


def test_environment_resolver_supports_each_development_principal(monkeypatch) -> None:
    tokens = {
        principal_id: secrets.token_urlsafe(32)
        for principal_id in DEVELOPMENT_PRINCIPAL_IDS
    }
    for principal_id, token in tokens.items():
        monkeypatch.setenv(development_token_environment_variable(principal_id), token)

    resolver = DevelopmentTokenPrincipalResolver.from_environment()

    for principal_id, token in tokens.items():
        assert resolver.resolve(f"Bearer {token}").principal_id == principal_id


def test_development_resolver_does_not_retain_raw_token() -> None:
    token, resolver = _development_resolver("data_analyst")

    assert token not in str(resolver.__dict__)


def test_environment_resolver_rejects_duplicate_tokens(monkeypatch) -> None:
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv(
        development_token_environment_variable("data_analyst"),
        token,
    )
    monkeypatch.setenv(
        development_token_environment_variable("apac_regional_manager"),
        token,
    )

    try:
        DevelopmentTokenPrincipalResolver.from_environment()
    except ValueError as error:
        assert str(error) == "Development bearer token configuration contains a duplicate."
    else:
        raise AssertionError("duplicate development tokens must fail closed")


def test_missing_or_invalid_bearer_credentials_are_rejected_without_echoing_token() -> None:
    token, resolver = _development_resolver("data_analyst")

    try:
        resolver.resolve(None)
    except PrincipalAuthenticationError as error:
        assert str(error) == "Authentication credentials are required."
    else:
        raise AssertionError("missing credentials must be rejected")

    try:
        resolver.resolve(f"Bearer {token}-invalid")
    except PrincipalAuthenticationError as error:
        assert str(error) == "Invalid authentication credentials."
        assert token not in str(error)
    else:
        raise AssertionError("invalid credentials must be rejected")


def test_authenticated_answer_uses_verified_principal_not_body_identity(client) -> None:
    token, resolver = _development_resolver("data_analyst")
    authenticated_client = TestClient(
        create_app(client.app.state.answer_service, principal_resolver=resolver)
    )

    response = authenticated_client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "agent_user_id": "apac_regional_manager",
            "access_profile": "apac_regional_manager",
            "question": "What is Jira New PEU?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "canonical_definition"
    assert response.json()["effective_access_scope"]["regions"] == [
        "Americas",
        "APAC",
        "EMEA",
    ]


def test_authenticated_answer_can_use_a_future_oidc_shaped_resolver(client) -> None:
    authorization = f"Bearer {secrets.token_urlsafe(32)}"

    class OidcLikeResolver:
        def resolve(self, received_authorization: str | None) -> VerifiedPrincipal:
            assert received_authorization == authorization
            return VerifiedPrincipal(
                principal_id="data_analyst",
                issuer="https://issuer.example.test",
                subject="oidc-subject-123",
            )

    authenticated_client = TestClient(
        create_app(client.app.state.answer_service, principal_resolver=OidcLikeResolver())
    )

    response = authenticated_client.post(
        "/answer_question",
        headers={"Authorization": authorization},
        json={"question": "What is Jira New PEU?"},
    )

    assert response.status_code == 200
    assert response.json()["canonical_definition"]["name"] == "jira_new_peu"


def test_known_body_identity_without_bearer_header_is_rejected(client) -> None:
    token, resolver = _development_resolver("data_analyst")
    unauthenticated_client = RawTestClient(
        create_app(client.app.state.answer_service, principal_resolver=resolver)
    )

    response = unauthenticated_client.post(
        "/answer_question",
        json={"agent_user_id": "data_analyst", "question": "What is Jira New PEU?"},
    )

    assert response.status_code == 401
    assert token not in response.text


def test_bearer_token_is_absent_from_response_and_trace(client) -> None:
    token, resolver = _development_resolver("data_analyst")

    class RecordingTraceSink:
        def __init__(self) -> None:
            self.recorded = None

        def record(self, trace) -> None:
            self.recorded = trace

    trace_sink = RecordingTraceSink()
    base_service = client.app.state.answer_service
    authenticated_client = TestClient(
        create_app(
            base_service.__class__(base_service.semantic_gateway, trace_sink=trace_sink),
            principal_resolver=resolver,
        )
    )

    response = authenticated_client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is Jira New PEU?"},
    )

    assert response.status_code == 200
    assert token not in response.text
    assert trace_sink.recorded is not None
    assert token not in str(trace_sink.recorded)


def test_answer_question_rejects_missing_and_invalid_http_credentials(client) -> None:
    token, resolver = _development_resolver("data_analyst")
    authenticated_client = TestClient(
        create_app(client.app.state.answer_service, principal_resolver=resolver)
    )
    payload = {"question": "What is Jira New PEU?"}

    missing = authenticated_client.post("/answer_question", json=payload)
    invalid = authenticated_client.post(
        "/answer_question",
        headers={"Authorization": f"Bearer {token}-invalid"},
        json=payload,
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert token not in missing.text
    assert token not in invalid.text

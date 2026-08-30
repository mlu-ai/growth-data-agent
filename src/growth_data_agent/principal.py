"""Provider-neutral principal verification for authenticated answer requests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from typing import Protocol

DEVELOPMENT_PRINCIPAL_IDS = (
    "data_analyst",
    "apac_regional_manager",
    "jira_product_manager",
    "confluence_product_manager",
    "customer_success_manager",
)
_DEVELOPMENT_TOKEN_ENV_PREFIX = "GROWTH_DATA_AGENT_DEV_TOKEN_"


def development_token_environment_variable(principal_id: str) -> str:
    """Return the environment variable name for one development principal."""
    return f"{_DEVELOPMENT_TOKEN_ENV_PREFIX}{principal_id.upper()}"


@dataclass(frozen=True)
class VerifiedPrincipal:
    """The source-neutral identity contract consumed by authorization policy."""

    principal_id: str
    issuer: str
    subject: str

    def __post_init__(self) -> None:
        if not self.principal_id or not self.issuer or not self.subject:
            raise ValueError("A Verified Principal requires an id, issuer, and subject.")


class PrincipalAuthenticationError(ValueError):
    """Raised when an authentication header cannot be verified."""


class PrincipalResolver(Protocol):
    """Verify credentials and return the stable principal contract."""

    def resolve(self, authorization: str | None) -> VerifiedPrincipal: ...


class DevelopmentTokenPrincipalResolver:
    """Resolve opaque bearer tokens configured outside the repository."""

    def __init__(self, token_to_principal: Mapping[str, VerifiedPrincipal]):
        if any(not token for token in token_to_principal):
            raise ValueError("Development bearer tokens must be non-empty.")
        self._token_digests = tuple(
            (sha256(token.encode("utf-8")).digest(), principal)
            for token, principal in token_to_principal.items()
        )

    @classmethod
    def from_environment(cls) -> DevelopmentTokenPrincipalResolver:
        """Load five optional local principal tokens without committing their values."""
        configured_tokens: dict[str, VerifiedPrincipal] = {}
        for principal_id in DEVELOPMENT_PRINCIPAL_IDS:
            token = os.environ.get(development_token_environment_variable(principal_id))
            if not token:
                continue
            if token in configured_tokens:
                raise ValueError("Development bearer token configuration contains a duplicate.")
            configured_tokens[token] = VerifiedPrincipal(
                principal_id=principal_id,
                issuer="development",
                subject=principal_id,
            )
        return cls(configured_tokens)

    def resolve(self, authorization: str | None) -> VerifiedPrincipal:
        if authorization is None or not authorization.strip():
            raise PrincipalAuthenticationError("Authentication credentials are required.")

        scheme, separator, token = authorization.partition(" ")
        if (
            scheme.casefold() != "bearer"
            or not separator
            or not token
            or token != token.strip()
            or " " in token
        ):
            raise PrincipalAuthenticationError("Invalid authentication credentials.")

        token_digest = sha256(token.encode("utf-8")).digest()
        for configured_digest, principal in self._token_digests:
            if compare_digest(configured_digest, token_digest):
                return principal
        raise PrincipalAuthenticationError("Invalid authentication credentials.")

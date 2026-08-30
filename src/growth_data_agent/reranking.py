"""Provider-neutral cross-encoder ranking for already-authorized evidence."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from .evidence import EvidenceDocument
from .observability import redact_identifiers

RERANKER_MODEL_NAME = "dengcao/Qwen3-Reranker-0.6B"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EvidenceReranker(Protocol):
    """Rank only the candidate documents supplied by the governed retriever."""

    model_name: str
    model_version: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[EvidenceDocument],
        *,
        limit: int,
    ) -> list[EvidenceDocument]: ...

    def readiness(self) -> dict[str, object]: ...


class EvidenceRerankingError(RuntimeError):
    """Raised when a cross-encoder cannot produce a bounded ranking."""


class EvidenceRerankerUnavailableError(EvidenceRerankingError, OSError):
    """Raised when the required cross-encoder dependency is unavailable."""


class DeterministicCrossEncoderReranker:
    """Explicit deterministic test/local double for the configured cross-encoder."""

    model_name = "deterministic-cross-encoder"
    model_version = "1"

    def rerank(
        self,
        query: str,
        candidates: Sequence[EvidenceDocument],
        *,
        limit: int,
    ) -> list[EvidenceDocument]:
        query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
        scored = [
            (
                len(query_tokens & set(_TOKEN_PATTERN.findall(document.title.casefold()))) * 2
                + len(query_tokens & set(_TOKEN_PATTERN.findall(document.text.casefold()))),
                document,
            )
            for document in candidates
        ]
        return [
            document
            for _, document in sorted(scored, key=lambda item: (-item[0], item[1].document_id))
        ][:limit]

    def readiness(self) -> dict[str, object]:
        return {
            "provider": "deterministic",
            "status": "ready",
            "model": self.model_name,
            "version": self.model_version,
        }


class OllamaCrossEncoderReranker:
    """Call the required Ollama-hosted cross-encoder through a bounded JSON contract."""

    model_name = RERANKER_MODEL_NAME
    model_version = "1"

    def __init__(
        self,
        *,
        model_name: str = RERANKER_MODEL_NAME,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ) -> None:
        if model_name != RERANKER_MODEL_NAME:
            raise ValueError(f"The governed evidence reranker requires {RERANKER_MODEL_NAME}.")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number.")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> OllamaCrossEncoderReranker:
        model_name = os.environ.get("OLLAMA_RERANKER_MODEL_NAME", RERANKER_MODEL_NAME)
        timeout_text = os.environ.get("OLLAMA_RERANKER_TIMEOUT_SECONDS", "60")
        try:
            timeout = float(timeout_text)
        except ValueError as error:
            raise ValueError(
                "OLLAMA_RERANKER_TIMEOUT_SECONDS must be a positive number."
            ) from error
        return cls(
            model_name=model_name,
            base_url=os.environ.get(
                "OLLAMA_RERANKER_BASE_URL",
                os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ),
            timeout=timeout,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[EvidenceDocument],
        *,
        limit: int,
    ) -> list[EvidenceDocument]:
        if not candidates:
            return []
        candidate_keys = [f"candidate-{index}" for index in range(len(candidates))]
        candidate_payloads = [
            {
                "candidate_id": candidate_key,
                "title": _safe_reranker_text(document, document.title),
                "text": _safe_reranker_text(document, document.text),
            }
            for candidate_key, document in zip(candidate_keys, candidates, strict=True)
        ]
        raw_output = self._send(
            {
                "model": self.model_name,
                "prompt": (
                    "Return only JSON with scores for every supplied candidate. Do not add, "
                    "remove, or alter candidate IDs. Score relevance to the query from 0 to 1.\n"
                    f"Query: {json.dumps(query)}\n"
                    f"Candidates: {json.dumps(candidate_payloads, sort_keys=True)}"
                ),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
        )
        try:
            payload = json.loads(raw_output)
        except (TypeError, ValueError) as error:
            raise EvidenceRerankingError("Cross-encoder returned invalid JSON.") from error
        scores = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(scores, list):
            raise EvidenceRerankingError("Cross-encoder returned no candidate scores.")
        parsed: dict[str, float] = {}
        for item in scores:
            if not isinstance(item, dict):
                raise EvidenceRerankingError("Cross-encoder returned a malformed score.")
            candidate_id = item.get("candidate_id")
            score = item.get("score")
            if (
                not isinstance(candidate_id, str)
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
                or candidate_id in parsed
            ):
                raise EvidenceRerankingError("Cross-encoder returned invalid candidate scores.")
            parsed[candidate_id] = float(score)
        if set(parsed) != set(candidate_keys):
            raise EvidenceRerankingError("Cross-encoder changed the authorized candidate set.")
        by_key = dict(zip(candidate_keys, candidates, strict=True))
        return [
            by_key[candidate_id]
            for candidate_id in sorted(parsed, key=lambda item: (-parsed[item], item))[:limit]
        ]

    def readiness(self) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=json.dumps({"name": self.model_name}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            status = "unavailable"
        else:
            status = (
                "ready"
                if isinstance(payload, dict) and not payload.get("error")
                else "unavailable"
            )
        return {
            "provider": "ollama",
            "status": status,
            "model": self.model_name,
            "version": self.model_version,
        }

    def _send(self, request_data: dict[str, object]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(request_data).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            raise EvidenceRerankerUnavailableError(
                f"Required cross-encoder model {self.model_name} is unavailable."
            ) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise EvidenceRerankingError("Cross-encoder returned no text response.")
        return payload["response"]


def reranker_readiness(reranker: EvidenceReranker | None) -> dict[str, object]:
    """Return readiness without making an absent reranker look ready."""
    if reranker is None:
        return {
            "provider": "none",
            "status": "unconfigured",
            "model": None,
            "version": None,
        }
    try:
        status = reranker.readiness()
    except Exception:
        status = {}
    if not isinstance(status, dict):
        status = {}
    readiness_status = status.get("status")
    if readiness_status not in {"ready", "configured", "unavailable", "unconfigured"}:
        readiness_status = "unavailable"
    return {
        "provider": status.get("provider", "unknown"),
        "status": readiness_status,
        "model": status.get("model"),
        "version": status.get("version"),
    }


def _safe_reranker_text(document: EvidenceDocument, value: str) -> str:
    """Keep direct identifiers out of model input unless the revision is entitled."""
    if document.identifier_entitlement == "direct":
        return value
    return str(redact_identifiers(value))

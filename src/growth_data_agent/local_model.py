"""Small local Ollama boundary used only by the baseline evaluator."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping


class LocalModelUnavailable(RuntimeError):
    """Raised when the configured local model cannot produce a result."""


class OllamaLocalModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, fixture: Mapping[str, object]) -> str:
        request_data = {
            "model": self.model_name,
            "prompt": (
                "Produce a concise answer using only this governed response. "
                "Do not add facts or identifiers that are absent from it.\n"
                f"{fixture['governed_context']}"
            ),
            "stream": False,
            "options": {"temperature": 0},
        }
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
            raise LocalModelUnavailable(
                f"Local Ollama model {self.model_name} is unavailable."
            ) from error
        output = payload.get("response")
        if not isinstance(output, str):
            raise LocalModelUnavailable(
                f"Local Ollama model {self.model_name} returned no text response."
            )
        return output

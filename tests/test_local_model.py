from __future__ import annotations

import json

from growth_data_agent.local_model import OllamaLocalModel


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
    model = OllamaLocalModel(model_name="qwen3:8b", base_url="http://127.0.0.1:11434")

    output = model.generate(
        {"id": "definition", "governed_context": '{"answer":"Define Jira New PEU"}' }
    )

    assert output == "A governed answer."
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert json.loads(request.data) == {
        "model": "qwen3:8b",
        "prompt": (
            "Produce a concise answer using only this governed response. "
            "Do not add facts or identifiers that are absent from it.\n"
            '{"answer":"Define Jira New PEU"}'
        ),
        "stream": False,
        "options": {"temperature": 0},
    }
    assert timeout == 60.0

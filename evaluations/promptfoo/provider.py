"""Promptfoo provider for the authenticated governed evaluation endpoint.

The provider receives only the evaluator-safe projection produced by the
private endpoint. It never receives an answer, evidence body, identifier, or
bearer token, and Promptfoo must target a local or private deployment through
PROMPTFOO_TARGET_URL.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _token_environment_variable(principal_id: str) -> str:
    """Mirror the repository's development-token naming without importing app code."""
    normalized = "_".join(principal_id.split()).upper().replace("-", "_")
    return f"GROWTH_DATA_AGENT_DEV_TOKEN_{normalized}"


def _selected_tools(executed_tools: object) -> list[str]:
    if not isinstance(executed_tools, list):
        return []
    return [
        tool["name"]
        for tool in executed_tools
        if isinstance(tool, dict)
        and tool.get("status") == "success"
        and isinstance(tool.get("name"), str)
        and len(tool["name"]) <= 64
    ]


def call_api(prompt: str, options: dict, context: dict) -> dict:
    variables = context.get("vars", {})
    principal_id = variables.get("agent_user_id")
    if not isinstance(principal_id, str) or not principal_id:
        return {"output": json.dumps({"error": "missing agent_user_id"})}
    token = os.environ.get(_token_environment_variable(principal_id))
    if not token:
        return {"output": json.dumps({"error": "missing development token"})}
    evaluation_token = os.environ.get("GROWTH_DATA_AGENT_EVALUATION_TOKEN")
    if not evaluation_token:
        return {"output": json.dumps({"error": "missing evaluator capability"})}
    config = options.get("config", {})
    base_url = config.get("url") or os.environ.get("PROMPTFOO_TARGET_URL")
    if not isinstance(base_url, str) or not base_url:
        return {"output": json.dumps({"error": "missing Promptfoo target URL"})}
    endpoint = base_url.rstrip("/") + "/evaluation/answer_question"
    body = json.dumps({"question": prompt}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Evaluation-Token": evaluation_token,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            status_code = response.status
            response_body = json.loads(response.read())
    except HTTPError as error:
        status_code = error.code
        response_body = {}
    except (OSError, URLError) as error:
        return {"output": json.dumps({"error": type(error).__name__})}
    return {
        "output": json.dumps(
            {
                "status_code": status_code,
                "response": response_body.get("response", {})
                if isinstance(response_body, dict)
                else {},
                "selected_tools": _selected_tools(
                    response_body.get("executed_tools", [])
                    if isinstance(response_body, dict)
                    else []
                ),
            },
            sort_keys=True,
        )
    }

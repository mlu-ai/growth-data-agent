# Model-Backed Metric Definitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an authenticated Agent User ask a paraphrased Canonical Metric definition question through the bounded Ollama intent provider while keeping dbt/MetricFlow, authorization, and route selection deterministic.

**Architecture:** The local model will classify only against the metric names exposed by the current validated semantic artifact. It will never produce a definition, route, scope, tool choice, or SQL; the service will retrieve the canonical definition and run the existing MetricFlow query after deterministic policy routing. A readiness endpoint will expose whether the opt-in Ollama dependency is configured, while model failures remain fail-closed clarification outcomes.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, LangGraph, MetricFlow/dbt semantic artifacts, Ollama HTTP API, pytest, Ruff, uv.

**Spec:** GitHub issue #53, “Model-backed paraphrased metric definitions”.

## Global Constraints

- dbt and MetricFlow remain the only canonical metric-definition authority.
- LLM output is a schema-validated metric proposal only; it may not define metrics, permissions, routes, tools, SQL, or scope.
- Only metric names from a current, successfully validated semantic artifact may be offered as canonical candidates.
- Ambiguous, invalid, unavailable, or out-of-catalog model output must route to clarification or an explicit unavailable/limitation response.
- Do not implement issue #52 authentication changes or any later ticket.
- Preserve the existing deterministic interpreter when Ollama is not configured.

---

### Task 1: Route paraphrased metric questions through semantic candidates

**Files:**
- Modify: `src/growth_data_agent/semantic.py`
- Modify: `src/growth_data_agent/local_model.py`
- Modify: `src/growth_data_agent/service.py`
- Test: `tests/test_local_model.py`
- Test: `tests/test_execution_graph.py`

**Interfaces:**
- Consumes: `ValidatedMetricFlowGateway.artifact_store` and `LocalModelTransport`.
- Produces: `ValidatedMetricFlowGateway.available_metric_names() -> tuple[str, ...]` and a `LocalModelIntentInterpreter` that accepts a callable candidate provider and rejects proposals outside the current validated artifact.

- [ ] **Step 1: Write the failing paraphrase and candidate-boundary tests**

```python
def test_local_model_intent_accepts_a_paraphrase_from_semantic_candidates():
    model = RecordingModel('{"metric_name":"jira_new_peu"}')
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


def test_local_model_intent_rejects_a_metric_not_in_the_validated_artifact():
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
```

- [ ] **Step 2: Run the focused tests to verify they fail for the missing provider contract**

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest tests/test_local_model.py -k 'paraphrase or validated_artifact' -v`

Expected: FAIL because the interpreter does not yet accept `metric_names_provider` or include semantic candidates in its model request.

- [ ] **Step 3: Implement the smallest candidate-backed interpreter change**

Add a gateway method that returns metric names only when the loaded artifact is current and has successful validation. Change the intent request schema to carry `available_metric_names`, make the interpreter call the request-scoped provider, reject `None` and names outside that tuple as `LocalModelOutputInvalid`, and then call the deterministic route resolver with the accepted name. Do not put definitions or formulas in the model prompt. Wire both service interpreter paths through the artifact-backed route resolver, and filter model candidates by the resolved Access Profile's entitled products before constructing the prompt.

- [ ] **Step 4: Run the focused tests to verify the candidate-backed path passes**

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest tests/test_local_model.py tests/test_execution_graph.py tests/test_answer_question.py -v`

Expected: PASS, including existing fail-closed and deterministic-scope tests.

- [ ] **Step 5: Commit the candidate-backed boundary**

```bash
git add src/growth_data_agent/semantic.py src/growth_data_agent/local_model.py src/growth_data_agent/service.py tests/test_local_model.py tests/test_execution_graph.py
git commit -m "feat: route paraphrased definitions through semantic candidates"
```

### Task 2: Make Ollama dependency readiness observable

**Files:**
- Modify: `src/growth_data_agent/local_model.py`
- Modify: `src/growth_data_agent/service.py`
- Modify: `src/growth_data_agent/main.py`
- Test: `tests/test_local_model.py`
- Test: `tests/test_answer_question.py`

**Interfaces:**
- Consumes: the configured `OllamaLocalModel` and the service’s selected intent interpreter.
- Produces: `GET /readiness` with explicit local-model provider/model status, while `GET /health` remains a liveness check.

- [ ] **Step 1: Write the failing readiness tests**

```python
def test_readiness_reports_deterministic_mode_when_ollama_is_not_configured(client):
    response = client.get("/readiness")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["local_model"] == {
        "provider": "none",
        "status": "disabled",
        "model": None,
    }


def test_readiness_reports_the_configured_ollama_model(monkeypatch, client):
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "qwen3:4b")
    app = create_app()

    response = TestClient(app).get("/readiness")

    assert response.status_code == 200
    assert response.json()["local_model"] == {
        "provider": "ollama",
        "status": "ready",
        "model": "qwen3:4b",
    }
```

- [ ] **Step 2: Run the readiness tests to verify they fail**

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest tests/test_answer_question.py -k readiness -v`

Expected: FAIL with a 404 because `/readiness` does not exist.

- [ ] **Step 3: Implement explicit readiness metadata with a bounded model probe**

Expose a small service/app status projection based on whether `OllamaLocalModel.from_environment()` selected the provider. When Ollama is selected, probe only its `/api/show` endpoint for the configured model and return HTTP 503 if that dependency is unavailable; report the provider and model name without credentials or question data. Keep `/health` unchanged and keep model generation failures fail-closed through the existing clarification path.

- [ ] **Step 4: Run focused tests and lint**

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest tests/test_local_model.py tests/test_answer_question.py tests/test_execution_graph.py -v`

Expected: PASS.

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run ruff check src tests`

Expected: exit 0 with no diagnostics.

- [ ] **Step 5: Commit the readiness surface**

```bash
git add src/growth_data_agent/local_model.py src/growth_data_agent/service.py src/growth_data_agent/main.py tests/test_local_model.py tests/test_answer_question.py
git commit -m "feat: expose local model readiness"
```

### Task 3: Document and verify issue #53 end to end

**Files:**
- Modify: `README.md`
- Modify: `docs/evaluation.md`
- Test: `tests/test_local_model.py`

- [ ] **Step 1: Add a public boundary test for a paraphrased definition**

```python
def test_configured_local_model_routes_a_paraphrased_definition_to_canonical_handler(
    client,
):
    model = RecordingModel('{"metric_name":"jira_new_peu"}')
    base_service = client.app.state.answer_service
    configured = AnswerQuestionService(base_service.semantic_gateway, local_model=model)

    response = TestClient(create_app(configured)).post(
        "/answer_question",
        json={
            "agent_user_id": "data_analyst",
            "question": "How is first-time paid access to Jira counted?",
        },
    )

    assert response.status_code == 200
    assert response.json()["result_classification"] == "canonical_definition"
    assert response.json()["canonical_definition"]["citation"]["authority"] == "dbt/MetricFlow"
```

- [ ] **Step 2: Run the complete local verification suite**

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run pytest`

Expected: PASS with zero failures.

Run: `env UV_CACHE_DIR=/private/tmp/growth-data-agent-uv-cache uv run ruff check .`

Expected: exit 0 with no diagnostics.

- [ ] **Step 3: Review the final diff against `origin/main` using the repository code-review skill**

Run: `git rev-parse origin/main`, `git diff origin/main...HEAD`, and `git log origin/main..HEAD --oneline`; then run both the Standards and Spec axes. The Spec axis must use GitHub issue #53, and the review must confirm no #52 or later-ticket implementation appears in the diff.

- [ ] **Step 4: Update docs and commit the final issue-scoped changes**

Document that `OLLAMA_MODEL_NAME` opt-in activates the bounded intent provider, paraphrases are classified only to current semantic artifact names, canonical definitions still come from dbt/MetricFlow, and `/readiness` exposes provider/model status. Then run `git diff --check` and commit:

```bash
git add README.md docs/evaluation.md tests/test_local_model.py
git diff --check
git commit -m "docs: explain model-backed metric definition boundary"
```

- [ ] **Step 5: Push the branch, open a PR closing #53, wait for GitHub CI, and report results without merging**

Use `~/.local/bin/gh-axi` for the push/PR/check workflow. The PR body must state that dbt/MetricFlow remains authoritative, model output is schema-validated and candidate-bound, and #53 is closed by the PR. Do not merge.

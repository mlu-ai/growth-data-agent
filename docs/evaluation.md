# Local evaluation and observability

The first baseline is deterministic and local-first. The fixture catalog in
`evaluations/fixtures.json` uses the public `answer_question` response seam to
check the definition, Driver Decomposition, evidence-backed Hypothesis,
cross-source authorization, direct-identifier safety, stale semantic
validation, and unsupported requests.

Retrieval is scored independently from response wording. The retrieval fixture
uses a human-labelled relevant document and computes recall@k, precision@k, and
reciprocal rank before the governed-response fixtures are judged.
Generation fixtures can opt into the `answer_faithfulness` evaluation category;
their deterministic field, wording, and refusal assertions are linked to the
governed trace in MLflow alongside the retrieval-quality metrics.

After the local Postgres and dbt steps in `README.md` have completed, run:

```sh
make evaluate
```

The command calls the configured local baseline through Ollama. The checked-in
baseline was recorded with the locally available `llama3.1:8b`; the POC's
intended `qwen3:8b` can be selected with `LOCAL_MODEL_NAME=qwen3:8b` when it is
available. The command calls the model for
each labelled request and records a redacted output hash and length in
`evaluations/baseline.json`. Start Ollama and make the model available before
running the command. Set `LOCAL_MODEL_NAME` and `LOCAL_MODEL_PROVIDER` when
comparing another locally served model; candidate results are written under
`evaluations/comparisons/` and compared with the canonical baseline metrics.
An existing canonical baseline is never overwritten by a candidate run; set
`BASELINE_EVALUATION_PATH` explicitly when recording a new baseline.
The governed response path remains deterministic: semantic truth,
authorization, retrieval, reranking, and safety are evaluated independently of
model wording. The evaluation runner uses an explicit deterministic reranker
double; the production app requires the governed Ollama cross-encoder.

The request-time intent provider is enabled only when `OLLAMA_MODEL_NAME` is
`qwen3:4b`. It receives a paraphrased question plus only the current validated
semantic artifact's metric-name candidates and returns a schema-validated
proposal with an explicit ambiguity status. It does not generate canonical
definitions or choose policy, routes, tools, or SQL; the answer service loads
the canonical definition from dbt/MetricFlow after deterministic route
validation. `GET /readiness` checks the configured Ollama intent model with its
model endpoint and returns HTTP 503 when that dependency is unavailable. When
the variable is unset, empty, or names another model, the deterministic
interpreter remains available; other configured model names remain available
only to the existing evidence-drafting adapter.

The service writes one redacted MLflow run per governed response. Set the
private hosted `MLFLOW_TRACKING_URI` (for example, `http://go/mlflow`) outside
development; `file:./data/mlruns` is available only when
`GROWTH_DATA_AGENT_ENVIRONMENT` is `development` or `test` (the default is
`development`). Trace tags include the route, policy fingerprint, evaluation
outcome, trace-delivery attempt, and trace identifier. Parameters and metrics
include source/configuration versions, tool outcomes, retrieval scores, and
turn latency. The `governed_trace.json` artifact recursively redacts
direct-identifier-shaped values before logging.

`GET /readiness` exposes only safe trace-delivery health: provider, delivery
attempt and failure counts, and the last exception class. If MLflow delivery
fails, the governed response remains available but readiness becomes degraded
(HTTP 503) so operators can alert on it. MLflow never receives raw prompts,
answers, SQL, Evidence Revision bodies, or direct identifiers.

## Governed Evaluation Dataset

`evaluations/dataset/v1/cases.json` is the versioned Governed Evaluation
Dataset: about 60 Evaluation Cases stratified across every supported route,
a shared review rubric, an Error Taxonomy, development/validation/held-out
splits, and a two-reviewer overlap sample. See
`docs/adr/0012-governed-evaluation-dataset-is-versioned-and-never-runtime-evidence.md`
for the versioning and provenance policy. Regenerate it with:

```sh
make evaluation-dataset
```

This is offline review content, not the local baseline runner above — it is
never imported by request-serving code.

## Governed evaluation runner and scorecard

`scripts/run_governed_evaluations.py` replays the Governed Evaluation
Dataset through the real `POST /answer_question` seam and publishes a
separate Evaluation Scorecard to MLflow (`MlflowTraceSink.record_scorecard`).
Run it with:

```sh
make governed-evaluate
```

It needs the same live Postgres + `make dbt-build` prerequisites as `make
evaluate`. A Case is executed only when none of its turns declares a
`setup_note` (bespoke harness state — a custom evidence store, a
monkeypatched Access Profile, an injected local model); the rest are
reported as `not_yet_automated_cases` on the scorecard, never silently
skipped. The scorecard reports safety, semantic correctness, trace delivery,
latency, and token/cost separately — never one composite score. See
`docs/adr/0013-governed-evaluation-runner-reports-honest-coverage.md`.

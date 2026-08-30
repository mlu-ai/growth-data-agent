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
authorization, retrieval, and safety are evaluated independently of model
wording.

The request-time intent provider is the opt-in Ollama-hosted `qwen3:4b` model. It
receives a paraphrased question plus only the current validated semantic
artifact's metric-name candidates and returns a schema-validated proposal. It
does not generate canonical definitions or choose policy, routes, tools, or
SQL; the answer service loads the canonical definition from dbt/MetricFlow
after deterministic route validation. `GET /readiness` checks the configured
Ollama model with its model endpoint and returns HTTP 503 when that dependency
is unavailable. When `OLLAMA_MODEL_NAME` is unset or empty, the deterministic
interpreter remains available for test/fallback mode.

The service writes one redacted MLflow run per governed response. The default
tracking URI is `file:./data/mlruns`; set `MLFLOW_TRACKING_URI` for a local
MLflow server. Trace tags include the route, policy fingerprint, evaluation
outcome, and trace identifier. Parameters and metrics include source versions,
tool outcomes, and retrieval scores. The `governed_trace.json` artifact
recursively redacts direct-identifier-shaped values before logging.

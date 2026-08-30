# Growth Data Agent

The first governed response seam returns typed, cited canonical definitions for
Jira and Confluence New PEU and New MAU. FastAPI verifies the validated semantic
artifact, has MetricFlow compile a bounded entitlement-constrained aggregate,
and executes that generated SQL against Postgres in a read-only transaction
before describing the metric as canonical.

When a named metric is absent from the current semantic artifact, the response
is a `metric_definition_gap`, never a canonical result. A Provisional Metric is
returned only by a configured entitlement-aware calculator that declares its
formula, inputs, scope, freshness, unverified status, and material caveats;
otherwise the service safely refuses to calculate it. The response offers a
data-team verification request, but creates its local POC record only when the
Agent User sends `verification_request_confirmation.approved: true` together
with approval context. It never creates an external ticket.

Authorized direct-identifier releases and explicitly approved Metric Definition
Gap verification requests are persisted in a local SQLite decision-record
database. Set `GROWTH_DATA_AGENT_DECISION_RECORDS_PATH` to choose its location
and `GROWTH_DATA_AGENT_AUDIT_RETENTION_DAYS` to configure retention; the default
is `data/decision_records.sqlite3` with twelve months (365 days). Only decision
metadata is stored: identifier values, approval prose, and source-page bodies
are excluded; a SHA-256 digest is retained for approval-context correlation.

Answer requests also support private durable Conversations. The server creates
an opaque `conversation_id` on the first answer; send it on a later request to
continue that Conversation. It is bound to the verified Agent User identity, and
each Turn re-resolves authorization and the current semantic artifact before
using the bounded recent context and structured summary. Set
`CONVERSATION_DATABASE_URL` to choose the Postgres checkpoint database (it
defaults to `DATABASE_URL`) and
`GROWTH_DATA_AGENT_CONVERSATION_RETENTION_DAYS` to configure raw turn-metadata
retention; the default is thirty days. Evidence chunks and full model payloads
are not stored as conversation memory.

Evidence questions about the APAC Jira decline or the Confluence EMEA New MAU
decline first run the validated Driver Decomposition, then query the synthetic
Qdrant corpus with Access Profile-derived product, Region, Tenant,
classification, and identifier filters. The response returns a cited
`hypothesis` only when permitted evidence supports the explanation; insufficient
or contradictory evidence is returned as `inconclusive`, without a causal
claim.

Causal questions for Jira New MAU run through the deterministic experiment
registry. The registered onboarding treatment/control design returns a
`causal_estimate` only when its support checks, pre-approved
`difference_in_means` estimator, diagnostics, assumptions, and human review
pass. Failed support, unregistered or observational designs, and missing review
return a descriptive result or reviewable analysis plan. All-user pre/post
comparisons are always labelled descriptive.

Canonical New MAU counts a New PEU only when at least one Visit is recorded for
the same Product User, product, and calendar month as first paid enablement. A
Visit to another product never qualifies the Product User. The deterministic
fixtures include a Confluence EMEA 51–200 Seat Tier New MAU decline and a
labelled onboarding-email regression Hypothesis.

## Local run

Choose one Postgres setup before running the data and application commands.

Before starting either path, configure one locally generated opaque bearer
token for each of the five development principals. The following commands
generate values without printing them; keep them in the shell environment and
never commit or log them:

```sh
export GROWTH_DATA_AGENT_DEV_TOKEN_DATA_ANALYST="$(openssl rand -hex 32)"
export GROWTH_DATA_AGENT_DEV_TOKEN_APAC_REGIONAL_MANAGER="$(openssl rand -hex 32)"
export GROWTH_DATA_AGENT_DEV_TOKEN_JIRA_PRODUCT_MANAGER="$(openssl rand -hex 32)"
export GROWTH_DATA_AGENT_DEV_TOKEN_CONFLUENCE_PRODUCT_MANAGER="$(openssl rand -hex 32)"
export GROWTH_DATA_AGENT_DEV_TOKEN_CUSTOMER_SUCCESS_MANAGER="$(openssl rand -hex 32)"
```

### Option A: Docker Compose

This is the local-container quick start. It creates the repository's local
Apache AGE/Postgres container on host port 5432.

```sh
uv sync --all-groups
make generate-data
docker compose up -d postgres
make load-data
make dbt-build
make semantic-artifact
DATAHUB_GMS_URL=http://127.0.0.1:8080 make publish-datahub
APACHE_AGE_DATABASE_URL=postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data make materialize-age
make evaluate
make serve
```

The bounded local intent provider uses Ollama when explicitly enabled. Set
`OLLAMA_MODEL_NAME=qwen3:4b` for the agreed initial model (only this value
enables intent interpretation; optionally set `OLLAMA_BASE_URL` or
`OLLAMA_TIMEOUT_SECONDS`):

```sh
OLLAMA_MODEL_NAME=qwen3:4b make serve
```

The intent model receives only the question and metric names from the current,
successfully validated dbt/MetricFlow artifact. It emits a schema-validated
metric proposal with an explicit ambiguity status; it cannot define metrics,
choose permissions, routes, tools, or SQL. Canonical definitions and MetricFlow
queries remain deterministic and are loaded from the validated semantic
artifact after routing. Invalid, ambiguous, or unavailable model output fails
closed to clarification. Other `OLLAMA_MODEL_NAME` values leave the intent
provider disabled while remaining available to the existing evidence-drafting
adapter.

Check the model dependency before sending analytical requests:

```sh
curl http://127.0.0.1:8000/readiness
```

Readiness reports the selected provider and model without exposing credentials;
an unavailable configured Ollama model returns HTTP 503. When Ollama is not
configured, readiness reports the deterministic interpreter as disabled and
`GET /health` remains the liveness check.

If port 5432 is already occupied, use the external-Postgres path below or set
`POSTGRES_PORT` before starting Compose. Do not start both Postgres paths for
the same run.

### Option B: Existing or pre-provisioned Postgres

Use this path when Postgres is already running locally or is provided by your
development environment. It avoids the Compose host-port collision, but the
database must be a dedicated, disposable POC database. `make load-data` creates
the synthetic source tables and can replace their contents; never point it at a
shared, staging, or production database.

Copy the tracked example, then set the connection values for that dedicated
database:

```sh
cp .env.example .env
# Edit .env with your local host, port, database, role, and password.
set -a
source .env
set +a
```

`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` configure dbt. `DATABASE_URL` configures the synthetic
loader and the API. Keep both pointing to the same database and role. A
successful external-Postgres run is:

```sh
uv sync --all-groups
make generate-data
make load-data
make dbt-build
make semantic-artifact
make serve
```

Skip `docker compose up -d postgres` when using this path. The optional
`publish-datahub`, `evaluate`, and `materialize-age` commands can be run after
the same environment has been loaded. Apache AGE runtime requirements for a
least-privileged application role are tracked separately in [issue #27](https://github.com/mlu-ai/growth-data-agent/issues/27);
this setup does not elevate the role or bypass those requirements.

```sh
export GROWTH_DATA_AGENT_DEV_TOKEN_DATA_ANALYST='<locally-generated-opaque-token>'
curl -X POST http://127.0.0.1:8000/answer_question \
  -H 'content-type: application/json' \
  -H "Authorization: Bearer ${GROWTH_DATA_AGENT_DEV_TOKEN_DATA_ANALYST}" \
  -d '{"question":"What is Jira New PEU?"}'
```

The deterministic generator produces 1,000 Tenants, 10,000 Persons, 16,000
Product Users, immutable Paid Enablement events, and Visit events across
eighteen months. Rebuild the semantic artifact after `make dbt-build`.

### Governed Confluence evidence sync

The operator sync path normalizes the synthetic Confluence corpus through the
same source contract used by live-shaped adapters and writes it to external
Qdrant. Configure `QDRANT_URL` (and `QDRANT_API_KEY` when required) before
running the backfill or incremental source sync:

```sh
make sync-confluence-evidence
```

The synchronizer skips unchanged page revisions, replaces updated revisions,
and retains deleted or inaccessible revisions only as non-retrievable
tombstones. Missing provenance, access policy, or embedding version metadata
fails closed before the batch mutates Qdrant. The `/readiness` response reports
Qdrant and embedding status without returning credentials.

## Checks

Pull requests to `main` run three independently selectable GitHub Actions
checks: `Ruff`, `pytest`, and `dbt build`. After the workflow has succeeded at
least once, repository administrators can select these exact check names in
`main` branch protection.

Run their local equivalents with:

```sh
uv sync --all-groups --locked
uv run ruff check .
uv run pytest

docker compose up -d postgres
uv run python scripts/generate_synthetic_data.py
uv run --group warehouse python scripts/load_postgres.py
cd dbt && uv run --group warehouse dbt build --profiles-dir .
```

See [docs/evaluation.md](docs/evaluation.md) for the deterministic fixture
evaluation, local-model baseline, and redacted MLflow trace configuration.

`make publish-datahub` publishes ownership, classification, and discovery metadata for
the current successfully validated dbt artifact. DataHub is catalog context only; the
last validated dbt/MetricFlow artifact remains the semantic authority, so canonical
metric answers remain available when DataHub is unavailable and catalog-dependent
answers disclose degraded availability. It targets the dbt-created Postgres Dataset
entities by default; metric ownership requests resolve to the corresponding `fct_*` model,
which is represented as the DataHub `Model` subtype. This keeps one physical DataHub
identity per validated dbt model. Override
`DATAHUB_TARGET_PLATFORM` or `DATAHUB_DATASET_PREFIX` when
the deployed DataHub dbt recipe uses another identity.

`make materialize-age` replaces the namespaced Apache AGE evidence index with bounded
metric-to-segment-to-Tenant-to-incident-or-team chains derived from that validated
artifact and the approved document-ingestion corpus. Configure `APACHE_AGE_DATABASE_URL`
and optionally `APACHE_AGE_GRAPH_NAME` before running it. For the deliberately
non-superuser application role, the dedicated database administrator must set
`session_preload_libraries = 'age'` and grant `USAGE ON SCHEMA ag_catalog`, for
example:

```sql
ALTER DATABASE growth_data_agent_poc SET session_preload_libraries = 'age';
GRANT USAGE ON SCHEMA ag_catalog TO growth_data_agent_poc;
```

Set `APACHE_AGE_PRELOADED=true` for the application and materializer sessions. The
adapter never elevates the role and reports an actionable configuration error when
AGE is not preloaded or available.

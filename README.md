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

Evidence questions about the APAC Jira decline or the Confluence EMEA New MAU
decline first run the validated Driver Decomposition, then query the synthetic
Qdrant corpus with Access Profile-derived product, Region, Tenant,
classification, and identifier filters. The response returns a cited
`hypothesis` only when permitted evidence supports the explanation; insufficient
or contradictory evidence is returned as `inconclusive`, without a causal
claim.

Canonical New MAU counts a New PEU only when at least one Visit is recorded for
the same Product User, product, and calendar month as first paid enablement. A
Visit to another product never qualifies the Product User. The deterministic
fixtures include a Confluence EMEA 51–200 Seat Tier New MAU decline and a
labelled onboarding-email regression Hypothesis.

## Local run

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

```sh
curl -X POST http://127.0.0.1:8000/answer_question \
  -H 'content-type: application/json' \
  -d '{"agent_user_id":"data_analyst","question":"What is Jira New PEU?"}'
```

The deterministic generator produces 1,000 Tenants, 10,000 Persons, 16,000
Product Users, immutable Paid Enablement events, and Visit events across
eighteen months. Rebuild the semantic artifact after `make dbt-build`.

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
and optionally `APACHE_AGE_GRAPH_NAME` before running it.

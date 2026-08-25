# Growth Data Agent

The first governed response seam returns a typed, cited canonical definition for
Jira New PEU. FastAPI verifies the validated semantic artifact, has MetricFlow
compile a bounded entitlement-constrained aggregate, and executes that generated
SQL against Postgres in a read-only transaction before describing the metric as
canonical.

When a named metric is absent from the current semantic artifact, the response
is a `metric_definition_gap`, never a canonical result. A Provisional Metric is
returned only by a configured entitlement-aware calculator that declares its
formula, inputs, scope, freshness, unverified status, and material caveats;
otherwise the service safely refuses to calculate it. The response offers a
data-team verification request, but creates its local POC record only when the
Agent User sends `verification_request_confirmation.approved: true` together
with approval context. It never creates an external ticket.

Evidence questions about the APAC 51–200 Seat Tier Tenant decline first run the
validated Driver Decomposition, then query the synthetic Qdrant corpus with
Access Profile-derived product, Region, Tenant, classification, and identifier
filters. The response returns a cited `hypothesis` only when permitted evidence
supports the explanation; insufficient or contradictory evidence is returned as
`inconclusive`, without a causal claim.

## Local run

```sh
uv sync --all-groups
make generate-data
docker compose up -d postgres
make load-data
make dbt-build
make semantic-artifact
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

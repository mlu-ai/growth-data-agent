# Growth Data Agent

The first governed response seam returns a typed, cited canonical definition for
Jira New PEU. FastAPI verifies the validated semantic artifact, has MetricFlow
compile a bounded entitlement-constrained aggregate, and executes that generated
SQL against Postgres in a read-only transaction before describing the metric as
canonical.

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

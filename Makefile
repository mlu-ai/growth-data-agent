.PHONY: generate-data load-data dbt-build semantic-artifact publish-datahub materialize-age sync-confluence-evidence serve evaluate lint test

generate-data:
	uv run python scripts/generate_synthetic_data.py

load-data:
	uv run --group warehouse python scripts/load_postgres.py

dbt-build:
	cd dbt && uv run --group warehouse dbt build --profiles-dir .

semantic-artifact:
	uv run python scripts/build_semantic_artifact.py

publish-datahub:
	uv run python scripts/publish_datahub_metadata.py

materialize-age:
	uv run python scripts/materialize_age_graph.py

sync-confluence-evidence:
	uv run python scripts/sync_confluence_evidence.py

serve:
	uv run uvicorn --app-dir src growth_data_agent.main:app --reload

evaluate:
	uv run python scripts/run_evaluations.py

lint:
	uv run ruff check .

test:
	uv run pytest

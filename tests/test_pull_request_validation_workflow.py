from __future__ import annotations

from pathlib import Path


def test_pull_requests_to_main_run_the_required_validation_checks() -> None:
    repository = Path(__file__).resolve().parents[1]
    workflow = (repository / ".github/workflows/pull-request-validation.yml").read_text()

    assert "pull_request:\n    branches: [main]" in workflow
    assert "name: Ruff" in workflow
    assert "uv sync --all-groups --locked" in workflow
    assert "uv run ruff check ." in workflow
    assert "name: pytest" in workflow
    assert "uv run pytest" in workflow
    assert "name: dbt build" in workflow
    assert "image: postgres:16-alpine" in workflow
    assert "ports:\n          - 5432:5432" in workflow
    assert "POSTGRES_HOST: 127.0.0.1" in workflow
    assert "uv run python scripts/generate_synthetic_data.py" in workflow
    assert "uv run --group warehouse python scripts/load_postgres.py" in workflow
    assert "uv run --group warehouse dbt build --profiles-dir ." in workflow

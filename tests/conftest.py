from __future__ import annotations

import ast
import json
import os
import secrets
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi import testclient as fastapi_testclient
from fastapi.testclient import TestClient as _BaseTestClient

from growth_data_agent.main import create_app
from growth_data_agent.metricflow_query import PlannedMetricFlowQuery
from growth_data_agent.principal import (
    DEVELOPMENT_PRINCIPAL_IDS,
    development_token_environment_variable,
)
from growth_data_agent.reranking import DeterministicCrossEncoderReranker
from growth_data_agent.semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from growth_data_agent.service import AnswerQuestionService

_TEST_PRINCIPAL_IDS = DEVELOPMENT_PRINCIPAL_IDS
_TEST_TOKENS = {
    principal_id: secrets.token_urlsafe(32) for principal_id in _TEST_PRINCIPAL_IDS
}
def pytest_configure() -> None:
    for principal_id, token in _TEST_TOKENS.items():
        os.environ[development_token_environment_variable(principal_id)] = token


class AuthenticatedTestClient(_BaseTestClient):
    """Keep pre-authentication fixtures valid while sending real bearer headers."""

    def post(self, url, *args, **kwargs):
        if url == "/answer_question":
            headers = dict(kwargs.get("headers") or {})
            has_authorization = any(key.casefold() == "authorization" for key in headers)
            payload = kwargs.get("json")
            principal_id = payload.get("agent_user_id") if isinstance(payload, dict) else None
            token = _TEST_TOKENS.get(principal_id)
            if not has_authorization and token is not None:
                headers["Authorization"] = f"Bearer {token}"
                kwargs["headers"] = headers
        return super().post(url, *args, **kwargs)


fastapi_testclient.TestClient = AuthenticatedTestClient
TestClient = AuthenticatedTestClient


def write_artifact(path: Path, *, status: str = "success", hours_old: int = 0) -> Path:
    validated_at = datetime(2026, 8, 25, tzinfo=UTC)
    semantic_manifest_path = path.with_name("semantic_manifest.json")
    semantic_manifest_path.write_text('{"semantic_models": [], "metrics": []}')
    artifact = {
        "artifact_type": "dbt_metricflow_semantic_artifact",
        "semantic_version": "1.0.0",
        "semantic_manifest_sha256": sha256(semantic_manifest_path.read_bytes()).hexdigest(),
        "validation": {
            "status": status,
            "validated_at": validated_at.isoformat(),
            "maximum_age_seconds": 86400,
        },
        "metrics": [
            {
                "name": "jira_new_peu",
                "definition": "A Product User's first-ever Paid Enablement for Jira.",
                "formula": (
                    "count_distinct(product_user_id) where product = Jira "
                    "and paid_enablement_ordinal = 1"
                ),
                "grain": "Product User in a Tenant and Jira product",
                "time_rule": "Attribute to first-ever Jira Paid Enablement.",
                "model_name": "fct_jira_new_peu",
                "citation_path": "dbt/models/marts/jira_new_peu.yml#jira_new_peu",
            },
            {
                "name": "jira_new_mau",
                "definition": (
                    "Jira New MAU is a New PEU with at least one Jira Visit in the "
                    "same calendar month as first paid enablement."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Jira product",
                "time_rule": (
                    "Attribute to first paid enablement only when a same-product Visit "
                    "occurs in that calendar month."
                ),
                "model_name": "fct_jira_new_mau",
                "citation_path": "dbt/models/marts/jira_new_mau.yml#jira_new_mau",
            },
            {
                "name": "confluence_new_peu",
                "definition": (
                    "Confluence New PEU is a Product User's first-ever Paid Enablement "
                    "for Confluence."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Confluence product",
                "time_rule": (
                    "Attribute to the first-ever Confluence Paid Enablement; later "
                    "restorations do not qualify again."
                ),
                "model_name": "fct_confluence_new_peu",
                "citation_path": "dbt/models/marts/confluence_new_peu.yml#confluence_new_peu",
            },
            {
                "name": "confluence_new_mau",
                "definition": (
                    "Confluence New MAU is a New PEU with at least one Confluence Visit "
                    "in the same calendar month as first paid enablement."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Confluence product",
                "time_rule": (
                    "Attribute to first paid enablement only when a same-product Visit "
                    "occurs in that calendar month."
                ),
                "model_name": "fct_confluence_new_mau",
                "citation_path": "dbt/models/marts/confluence_new_mau.yml#confluence_new_mau",
            },
            {
                "name": "jira_new_peu_eligible_population",
                "definition": (
                    "Jira New PEU Eligible Population: entitled Product Users who have "
                    "not previously qualified through Paid Enablement for Jira."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Jira product",
                "time_rule": "Entitled but with no first-ever Jira Paid Enablement.",
                "model_name": "fct_jira_new_peu_eligible_population",
                "citation_path": (
                    "dbt/models/marts/jira_new_peu_eligible_population.yml"
                    "#jira_new_peu_eligible_population"
                ),
            },
            {
                "name": "confluence_new_peu_eligible_population",
                "definition": (
                    "Confluence New PEU Eligible Population: entitled Product Users who "
                    "have not previously qualified through Paid Enablement for "
                    "Confluence."
                ),
                "formula": "count_distinct(product_user_id)",
                "grain": "Product User in a Tenant and Confluence product",
                "time_rule": "Entitled but with no first-ever Confluence Paid Enablement.",
                "model_name": "fct_confluence_new_peu_eligible_population",
                "citation_path": (
                    "dbt/models/marts/confluence_new_peu_eligible_population.yml"
                    "#confluence_new_peu_eligible_population"
                ),
            },
        ],
    }
    path.write_text(json.dumps(artifact))
    return path


class RecordingMetricFlowPlanner:
    def __init__(self, semantic_manifest_path: Path):
        self.semantic_manifest_path = semantic_manifest_path
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return PlannedMetricFlowQuery(
            metric_name=request.metric_name,
            sql=f"select 1 as {request.metric_name}",
            parameters={},
            where_constraints=request.where_constraints,
            group_by_names=request.group_by_names,
        )


def _driver_row(month: str, region: str, seat_tier: str, value: int) -> dict[str, object]:
    return {
        "metric_time__month": f"{month}-01",
        "product_user__region": region,
        "product_user__seat_tier": seat_tier,
        "jira_new_mau_product_user__region": region,
        "jira_new_mau_product_user__seat_tier": seat_tier,
        "jira_new_peu": value,
        "jira_new_mau": value,
    }


def _confluence_driver_row(
    month: str, region: str, seat_tier: str, value: int
) -> dict[str, object]:
    return {
        "metric_time__month": f"{month}-01",
        "confluence_product_user__region": region,
        "confluence_product_user__seat_tier": seat_tier,
        "confluence_new_peu": value,
    }


def _confluence_new_mau_driver_row(
    month: str, region: str, seat_tier: str, value: int
) -> dict[str, object]:
    return {
        "metric_time__month": f"{month}-01",
        "confluence_new_mau_product_user__region": region,
        "confluence_new_mau_product_user__seat_tier": seat_tier,
        "confluence_new_mau": value,
    }


class RecordingPostgresExecutor:
    def __init__(self) -> None:
        self.plans = []

    _driver_rows = [
        _driver_row("2026-05", "APAC", "51-200", 800),
        _driver_row("2026-06", "APAC", "51-200", 380),
        _driver_row("2026-05", "Americas", "1-10", 1000),
        _driver_row("2026-06", "Americas", "1-10", 940),
        _driver_row("2026-05", "EMEA", "11-50", 700),
        _driver_row("2026-06", "EMEA", "11-50", 680),
        _driver_row("2026-05", "APAC", "1-10", 600),
        _driver_row("2026-06", "APAC", "1-10", 580),
        _driver_row("2026-05", "EMEA", "51-200", 500),
        _driver_row("2026-06", "EMEA", "51-200", 480),
        _driver_row("2026-05", "Americas", "51-200", 400),
        _driver_row("2026-06", "Americas", "51-200", 380),
    ]
    _confluence_driver_rows = [
        _confluence_driver_row("2026-05", "Americas", "11-50", 1200),
        _confluence_driver_row("2026-06", "Americas", "11-50", 1620),
        _confluence_driver_row("2026-05", "APAC", "1-10", 600),
        _confluence_driver_row("2026-06", "APAC", "1-10", 600),
        _confluence_driver_row("2026-05", "EMEA", "51-200", 600),
        _confluence_driver_row("2026-06", "EMEA", "51-200", 600),
    ]
    _confluence_new_mau_driver_rows = [
        _confluence_new_mau_driver_row("2026-05", "APAC", "1-10", 32),
        _confluence_new_mau_driver_row("2026-06", "APAC", "1-10", 33),
        _confluence_new_mau_driver_row("2026-05", "Americas", "11-50", 66),
        _confluence_new_mau_driver_row("2026-06", "Americas", "11-50", 89),
        _confluence_new_mau_driver_row("2026-05", "EMEA", "51-200", 600),
        _confluence_new_mau_driver_row("2026-06", "EMEA", "51-200", 300),
    ]

    def execute(self, plan: PlannedMetricFlowQuery) -> int:
        self.plans.append(plan)
        return 1

    def execute_rows(self, plan: PlannedMetricFlowQuery):
        self.plans.append(plan)
        if plan.metric_name == "jira_new_peu_eligible_population":
            return [{"jira_new_peu_eligible_population": 40}]
        if plan.metric_name == "confluence_new_peu_eligible_population":
            return [{"confluence_new_peu_eligible_population": 25}]
        if plan.metric_name == "confluence_new_mau":
            if any(
                "confluence_new_mau_product_user__region IN ('APAC')" in constraint
                for constraint in plan.where_constraints
            ):
                return [
                    row
                    for row in self._confluence_new_mau_driver_rows
                    if row["confluence_new_mau_product_user__region"] == "APAC"
                ]
            return self._confluence_new_mau_driver_rows
        if plan.metric_name == "confluence_new_peu":
            return self._confluence_driver_rows
        if any("__region IN ('APAC')" in constraint for constraint in plan.where_constraints):
            return [row for row in self._driver_rows if row["product_user__region"] == "APAC"]
        return self._driver_rows


_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/growth_data_agent"


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` — a block whose
    imports never execute at runtime, only for static type checkers."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _iter_runtime_nodes(node: ast.AST):
    """Walk the tree like `ast.walk`, but never descend into an
    `if TYPE_CHECKING:` block's body — those imports never execute."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking_guard(child.test):
            continue
        yield from _iter_runtime_nodes(child)


def _local_module_imports(path: Path) -> set[str]:
    """Names this file imports **at runtime** that could refer to another module
    in this package (bare module names for `import x` / relative `from .x import
    y`) — third-party absolute imports (`from fastapi import ...`) are filtered
    out by the caller, which only follows names that resolve to an actual file in
    _SOURCE_DIR. Imports inside an `if TYPE_CHECKING:` block are excluded: they
    never run, so they can't make anything reachable at runtime."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in _iter_runtime_nodes(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level > 0:
            names.add(node.module.split(".")[0])
    return names


def modules_reachable_from_main() -> set[str]:
    """All `src/growth_data_agent` module names transitively imported at
    runtime starting from `main.py` — the ASGI app and every request-serving
    path actually loads only these. Shared by tests proving some offline-only
    module (an Evaluation Dataset, a RAG dataset, ...) is never runtime
    evidence: a graph walk from the live entrypoint, not a hand-maintained
    file blocklist, so a new offline-only consumer is naturally exempt and a
    future accidental import from ANY request-serving module is still caught.
    """
    visited: set[str] = set()
    to_visit = ["main"]
    while to_visit:
        module_name = to_visit.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        module_path = _SOURCE_DIR / f"{module_name}.py"
        if not module_path.exists():
            continue
        for imported in _local_module_imports(module_path):
            if (_SOURCE_DIR / f"{imported}.py").exists() and imported not in visited:
                to_visit.append(imported)
    return visited


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    artifact_path = write_artifact(tmp_path / "semantic.json")
    planner = RecordingMetricFlowPlanner(tmp_path / "semantic_manifest.json")
    executor = RecordingPostgresExecutor()
    gateway = ValidatedMetricFlowGateway(
        SemanticArtifactStore(artifact_path),
        metricflow_planner=planner,
        postgres_executor=executor,
        now=lambda: datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    return TestClient(
        create_app(
            AnswerQuestionService(
                gateway,
                evidence_reranker=DeterministicCrossEncoderReranker(),
            )
        )
    )

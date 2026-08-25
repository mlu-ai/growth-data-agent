from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_artifact_builder_accepts_passing_dbt_tests(tmp_path, monkeypatch) -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts/build_semantic_artifact.py"
    specification = importlib.util.spec_from_file_location("artifact_builder", module_path)
    assert specification and specification.loader
    builder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(builder)

    run_results_path = tmp_path / "run_results.json"
    run_results_path.write_text(
        json.dumps({"results": [{"status": "success"}, {"status": "pass"}]})
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "semantic_models": {
                    "semantic_model.growth_data_agent.jira_new_peu": {
                        "original_file_path": "models/marts/jira_new_peu.yml",
                        "config": {
                            "meta": {
                                "semantic_version": "1.0.0",
                                "canonical_grain": "Product User in a Tenant and Jira product",
                                "canonical_time_rule": "Use the first-ever Paid Enablement.",
                            }
                        },
                    },
                    "semantic_model.growth_data_agent.confluence_new_peu": {
                        "original_file_path": "models/marts/confluence_new_peu.yml",
                        "config": {
                            "meta": {
                                "semantic_version": "1.0.0",
                                "canonical_grain": (
                                    "Product User in a Tenant and Confluence product"
                                ),
                                "canonical_time_rule": (
                                    "Use the first-ever Confluence Paid Enablement."
                                ),
                            }
                        },
                    }
                }
            }
        )
    )
    semantic_manifest_path = tmp_path / "semantic_manifest.json"
    semantic_manifest_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "name": "jira_new_peu",
                        "description": "First-ever Jira Paid Enablement.",
                        "type_params": {"measure": {"name": "jira_new_peu"}},
                    },
                    {
                        "name": "confluence_new_peu",
                        "description": "First-ever Confluence Paid Enablement.",
                        "type_params": {"measure": {"name": "confluence_new_peu"}},
                    },
                    ],
                "semantic_models": [
                    {
                        "name": "jira_new_peu",
                        "node_relation": {"alias": "fct_jira_new_peu"},
                        "measures": [
                            {
                                "name": "jira_new_peu",
                                "agg": "count_distinct",
                                "expr": "product_user_id",
                            }
                        ],
                    },
                    {
                        "name": "confluence_new_peu",
                        "node_relation": {"alias": "fct_confluence_new_peu"},
                        "measures": [
                            {
                                "name": "confluence_new_peu",
                                "agg": "count_distinct",
                                "expr": "product_user_id",
                            }
                        ],
                    }
                ],
            }
        )
    )
    artifact_path = tmp_path / "semantic.json"
    monkeypatch.setattr(builder, "_ARTIFACT", artifact_path)
    monkeypatch.setattr(builder, "_RUN_RESULTS", run_results_path)
    monkeypatch.setattr(builder, "_MANIFEST", manifest_path)
    monkeypatch.setattr(builder, "_SEMANTIC_MANIFEST", semantic_manifest_path)

    builder.main()

    artifact = json.loads(artifact_path.read_text())
    assert artifact["validation"]["status"] == "success"
    assert artifact["metrics"][0]["formula"] == "count_distinct(product_user_id)"
    assert [metric["name"] for metric in artifact["metrics"]] == [
        "jira_new_peu",
        "confluence_new_peu",
    ]
    assert (
        artifact["metrics"][0]["citation_path"]
        == "dbt/models/marts/jira_new_peu.yml#jira_new_peu"
    )

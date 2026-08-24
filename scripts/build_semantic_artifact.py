"""Mark the checked-in semantic metadata current after a successful dbt build."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

_ARTIFACT = Path("dbt/artifacts/last_validated_semantic.json")
_RUN_RESULTS = Path("dbt/target/run_results.json")
_MANIFEST = Path("dbt/target/manifest.json")
_SEMANTIC_MANIFEST = Path("dbt/target/semantic_manifest.json")
_METRIC_NAME = "jira_new_peu"


def main() -> None:
    required_artifacts = (_RUN_RESULTS, _MANIFEST, _SEMANTIC_MANIFEST)
    if missing := [str(path) for path in required_artifacts if not path.exists()]:
        raise SystemExit(
            "Missing dbt artifacts: "
            f"{', '.join(missing)}. Run make dbt-build before refreshing the artifact."
        )
    run_results = json.loads(_RUN_RESULTS.read_text())
    failed = [
        result
        for result in run_results.get("results", [])
        if result.get("status") not in {"success", "pass"}
    ]
    if failed:
        raise SystemExit("dbt validation did not succeed; semantic artifact was not refreshed.")

    semantic_manifest = json.loads(_SEMANTIC_MANIFEST.read_text())
    manifest = json.loads(_MANIFEST.read_text())
    metric = _find_by_name(semantic_manifest["metrics"], _METRIC_NAME)
    semantic_model = _find_semantic_model(semantic_manifest, metric)
    manifest_model = manifest["semantic_models"][
        f"semantic_model.growth_data_agent.{semantic_model['name']}"
    ]
    metadata = manifest_model["config"]["meta"]
    measure = _find_by_name(semantic_model["measures"], _METRIC_NAME)

    artifact = {
        "artifact_type": "dbt_metricflow_semantic_artifact",
        "semantic_version": metadata["semantic_version"],
        "semantic_manifest_sha256": sha256(_SEMANTIC_MANIFEST.read_bytes()).hexdigest(),
        "validation": {
            "status": "success",
            "validated_at": datetime.now(UTC).isoformat(),
            "maximum_age_seconds": 86_400,
        },
        "metrics": [
            {
                "name": metric["name"],
                "definition": metric["description"],
                "formula": f"{measure['agg']}({measure['expr']})",
                "grain": metadata["canonical_grain"],
                "time_rule": metadata["canonical_time_rule"],
                "model_name": semantic_model["node_relation"]["alias"],
                "citation_path": (
                    f"dbt/{manifest_model['original_file_path']}#{metric['name']}"
                ),
            }
        ],
    }
    _ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Built {_ARTIFACT} from validated dbt/MetricFlow artifacts.")


def _find_semantic_model(semantic_manifest: dict, metric: dict) -> dict:
    measure_name = metric["type_params"]["measure"]["name"]
    return next(
        model
        for model in semantic_manifest["semantic_models"]
        if any(measure["name"] == measure_name for measure in model["measures"])
    )


def _find_by_name(items: list[dict], name: str) -> dict:
    return next(item for item in items if item["name"] == name)


if __name__ == "__main__":
    main()

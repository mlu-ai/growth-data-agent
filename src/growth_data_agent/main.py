"""FastAPI entrypoint for the governed answer_question seam."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .contracts import AnswerQuestionRequest, GovernedAnalyticalResponse
from .datahub import DataHubHttpCatalog
from .graph import (
    ApacheAgeEvidenceGraphStore,
    EvidenceGraphUnavailableError,
    PsycopgAgeGraphQueryExecutor,
)
from .metricflow_query import (
    MetricFlowPlanner,
    PostgresMetricFlowExecutor,
    SemanticQueryExecutionError,
)
from .observability import MlflowTraceSink
from .policy import AccessDeniedError, UnknownAgentUserError
from .semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from .service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_DEFAULT_SEMANTIC_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"


def create_app(service: AnswerQuestionService | None = None) -> FastAPI:
    app = FastAPI(title="Growth Data Agent", version="0.1.0")
    datahub_gms_url = os.environ.get("DATAHUB_GMS_URL")
    age_database_url = os.environ.get("APACHE_AGE_DATABASE_URL")
    app.state.answer_service = service or AnswerQuestionService(
        ValidatedMetricFlowGateway(
            SemanticArtifactStore(
                Path(os.environ.get("SEMANTIC_ARTIFACT_PATH", _DEFAULT_ARTIFACT))
            ),
            metricflow_planner=MetricFlowPlanner(
                Path(os.environ.get("SEMANTIC_MANIFEST_PATH", _DEFAULT_SEMANTIC_MANIFEST))
            ),
            postgres_executor=PostgresMetricFlowExecutor(
                os.environ.get(
                    "DATABASE_URL",
                    "postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data",
                )
            ),
        ),
        catalog_store=(
            DataHubHttpCatalog(
                datahub_gms_url,
                token=os.environ.get("DATAHUB_TOKEN"),
                platform=os.environ.get("DATAHUB_TARGET_PLATFORM", "postgres"),
                dataset_prefix=os.environ.get(
                    "DATAHUB_DATASET_PREFIX", "growth_data.analytics"
                ),
            )
            if datahub_gms_url
            else None
        ),
        graph_store=(
            ApacheAgeEvidenceGraphStore(
                PsycopgAgeGraphQueryExecutor(
                    age_database_url,
                    graph_name=os.environ.get("APACHE_AGE_GRAPH_NAME", "growth_evidence"),
                )
            )
            if age_database_url
            else None
        ),
        trace_sink=MlflowTraceSink.from_environment(),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/answer_question", response_model=GovernedAnalyticalResponse)
    def answer_question(request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        try:
            return app.state.answer_service.answer_question(request)
        except UnknownAgentUserError as error:
            trace_id = error.trace_id or "unavailable"
            raise HTTPException(
                status_code=403,
                detail=f"{error} (trace_id={trace_id})",
            ) from error
        except AccessDeniedError as error:
            trace_id = error.trace_id or "unavailable"
            raise HTTPException(
                status_code=403,
                detail=f"{error} (trace_id={trace_id})",
            ) from error
        except EvidenceGraphUnavailableError as error:
            raise HTTPException(
                status_code=503, detail="Evidence graph is unavailable."
            ) from error
        except (OSError, ValidationError, SemanticQueryExecutionError) as error:
            raise HTTPException(
                status_code=503, detail="Semantic artifact is unavailable."
            ) from error

    return app


app = create_app()

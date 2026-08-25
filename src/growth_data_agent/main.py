"""FastAPI entrypoint for the governed answer_question seam."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .contracts import AnswerQuestionRequest, GovernedAnalyticalResponse
from .metricflow_query import (
    MetricFlowPlanner,
    PostgresMetricFlowExecutor,
    SemanticQueryExecutionError,
)
from .policy import AccessDeniedError, UnknownAgentUserError
from .semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from .service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_DEFAULT_SEMANTIC_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"


def create_app(service: AnswerQuestionService | None = None) -> FastAPI:
    app = FastAPI(title="Growth Data Agent", version="0.1.0")
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
        )
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/answer_question", response_model=GovernedAnalyticalResponse)
    def answer_question(request: AnswerQuestionRequest) -> GovernedAnalyticalResponse:
        try:
            return app.state.answer_service.answer_question(request)
        except UnknownAgentUserError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except AccessDeniedError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (OSError, ValidationError, SemanticQueryExecutionError) as error:
            raise HTTPException(
                status_code=503, detail="Semantic artifact is unavailable."
            ) from error

    return app


app = create_app()

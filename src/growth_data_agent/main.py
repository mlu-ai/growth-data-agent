"""FastAPI entrypoint for the governed answer_question seam."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .contracts import AnswerQuestionRequest, GovernedAnalyticalResponse
from .policy import UnknownAgentUserError
from .semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from .service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"


def create_app(service: AnswerQuestionService | None = None) -> FastAPI:
    app = FastAPI(title="Growth Data Agent", version="0.1.0")
    app.state.answer_service = service or AnswerQuestionService(
        ValidatedMetricFlowGateway(
            SemanticArtifactStore(Path(os.environ.get("SEMANTIC_ARTIFACT_PATH", _DEFAULT_ARTIFACT)))
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
        except (OSError, ValidationError) as error:
            raise HTTPException(
                status_code=503, detail="Semantic artifact is unavailable."
            ) from error

    return app


app = create_app()

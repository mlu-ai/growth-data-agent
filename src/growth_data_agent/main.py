"""FastAPI entrypoint for the governed answer_question seam."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .audit import SQLiteDirectIdentifierAuditRecorder
from .contracts import AnswerQuestionPayload, AnswerQuestionRequest, GovernedAnalyticalResponse
from .conversations import (
    PostgresConversationCheckpointStore,
    conversation_retention_from_environment,
)
from .datahub import DataHubHttpCatalog
from .graph import (
    ApacheAgeEvidenceGraphStore,
    EvidenceGraphUnavailableError,
    PsycopgAgeGraphQueryExecutor,
    apache_age_preloaded_from_environment,
)
from .local_model import OllamaIntentModel, OllamaLocalModel
from .metric_definition_gaps import SQLiteDataTeamVerificationRequestRecorder
from .metricflow_query import (
    MetricFlowPlanner,
    PostgresMetricFlowExecutor,
    SemanticQueryExecutionError,
)
from .observability import MlflowTraceSink
from .persistence import (
    decision_record_path_from_environment,
    decision_record_retention_from_environment,
)
from .policy import AccessDeniedError, UnknownAgentUserError
from .principal import (
    DevelopmentTokenPrincipalResolver,
    PrincipalAuthenticationError,
    PrincipalResolver,
)
from .semantic import SemanticArtifactStore, ValidatedMetricFlowGateway
from .service import AnswerQuestionService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACT = _REPOSITORY_ROOT / "dbt/artifacts/last_validated_semantic.json"
_DEFAULT_SEMANTIC_MANIFEST = _REPOSITORY_ROOT / "dbt/target/semantic_manifest.json"
_DEFAULT_DECISION_RECORDS = _REPOSITORY_ROOT / "data/decision_records.sqlite3"


def create_app(
    service: AnswerQuestionService | None = None,
    *,
    principal_resolver: PrincipalResolver | None = None,
) -> FastAPI:
    app = FastAPI(title="Growth Data Agent", version="0.1.0")
    app.state.principal_resolver = (
        principal_resolver or DevelopmentTokenPrincipalResolver.from_environment()
    )
    datahub_gms_url = os.environ.get("DATAHUB_GMS_URL")
    age_database_url = os.environ.get("APACHE_AGE_DATABASE_URL")
    intent_model = None if service is not None else OllamaIntentModel.from_environment()
    evidence_model = None if service is not None else OllamaLocalModel.from_environment()
    decision_records_path = decision_record_path_from_environment(_DEFAULT_DECISION_RECORDS)
    decision_record_retention = decision_record_retention_from_environment()
    conversation_database_url = os.environ.get(
        "CONVERSATION_DATABASE_URL",
        os.environ.get(
            "DATABASE_URL",
            "postgresql://growth_data:growth_data@127.0.0.1:5432/growth_data",
        ),
    )
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
                dataset_prefix=os.environ.get("DATAHUB_DATASET_PREFIX", "growth_data.analytics"),
            )
            if datahub_gms_url
            else None
        ),
        graph_store=(
            ApacheAgeEvidenceGraphStore(
                PsycopgAgeGraphQueryExecutor(
                    age_database_url,
                    graph_name=os.environ.get("APACHE_AGE_GRAPH_NAME", "growth_evidence"),
                    age_preloaded=apache_age_preloaded_from_environment(),
                )
            )
            if age_database_url
            else None
        ),
        verification_request_recorder=SQLiteDataTeamVerificationRequestRecorder(
            decision_records_path,
            retention=decision_record_retention,
        ),
        direct_identifier_audit_recorder=SQLiteDirectIdentifierAuditRecorder(
            decision_records_path,
            retention=decision_record_retention,
        ),
        trace_sink=MlflowTraceSink.from_environment(),
        local_model=intent_model,
        evidence_model=evidence_model,
        conversation_store=PostgresConversationCheckpointStore(
            conversation_database_url,
            retention=conversation_retention_from_environment(),
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readiness")
    def readiness() -> JSONResponse:
        status = app.state.answer_service.readiness()
        return JSONResponse(
            status_code=503 if status["status"] == "unavailable" else 200,
            content=status,
        )

    @app.post("/answer_question", response_model=GovernedAnalyticalResponse)
    def answer_question(
        payload: AnswerQuestionPayload,
        authorization: str | None = Header(default=None),
    ) -> GovernedAnalyticalResponse:
        try:
            principal = app.state.principal_resolver.resolve(authorization)
        except PrincipalAuthenticationError as error:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Authentication credentials are required."
                    if authorization is None or not authorization.strip()
                    else "Invalid authentication credentials."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

        request = AnswerQuestionRequest(
            agent_user_id=principal.principal_id,
            question=payload.question,
            requested_metric_name=payload.requested_metric_name,
            experiment_id=payload.experiment_id,
            conversation_id=payload.conversation_id,
            verification_request_confirmation=payload.verification_request_confirmation,
            verified_principal=principal,
        )
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
                status_code=503,
                detail=(
                    "Evidence graph is unavailable. "
                    f"(trace_id={getattr(error, 'trace_id', 'unavailable')})"
                ),
            ) from error
        except (OSError, ValidationError, SemanticQueryExecutionError) as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Semantic artifact is unavailable. "
                    f"(trace_id={getattr(error, 'trace_id', 'unavailable')})"
                ),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Governed dependency is unavailable. "
                    f"(trace_id={getattr(error, 'trace_id', 'unavailable')})"
                ),
            ) from error

    return app


app = create_app()

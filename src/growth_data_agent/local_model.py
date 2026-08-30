"""Bounded local-model adapters for intent proposals and evidence prose."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .contracts import (
    AnalyticalIntent,
    AnalyticalRoute,
    AnswerQuestionRequest,
    ConversationContext,
    EvidenceSupportStatus,
    GovernedAnalyticalResponse,
    ResultClassification,
)
from .observability import redact_identifiers

_MAX_INTENT_QUESTION_LENGTH = 2_000
_MAX_DRAFT_LENGTH = 2_000
OLLAMA_INTENT_MODEL_NAME = "qwen3:4b"
_ModelResponse = TypeVar("_ModelResponse", bound=BaseModel)


class LocalModelError(RuntimeError):
    """Base error for a local-model boundary failure."""


class LocalModelUnavailable(LocalModelError):
    """Raised when the configured local model cannot produce a result."""


class LocalModelOutputInvalid(LocalModelError):
    """Raised when a model response is not valid for its bounded contract."""


class LocalModelTransport(Protocol):
    """The only transport capability exposed to the bounded adapters."""

    def generate(self, request: LocalModelCall) -> str: ...


class EvidenceDraftingAdapter(Protocol):
    """Draft prose from a governed response without changing its typed data."""

    def draft(self, response: GovernedAnalyticalResponse) -> LocalModelDraftProposal: ...


class LocalModelIntentRequest(BaseModel):
    """The small, non-authoritative request sent to an intent model."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["intent_proposal"] = "intent_proposal"
    question: str = Field(min_length=1, max_length=_MAX_INTENT_QUESTION_LENGTH)
    requested_metric_name: str | None = Field(default=None, min_length=1, max_length=128)
    available_metric_names: list[
        Annotated[
            str,
            Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_]+$"),
        ]
    ] = Field(min_length=1, max_length=64)
    conversation_context: ConversationContext | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class LocalModelIntentProposal(BaseModel):
    """Only the deterministic metric proposal is accepted from the model."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str | None = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9_]+$",
    )
    ambiguity: Literal["unambiguous", "ambiguous"]


class CitedEvidenceCitation(BaseModel):
    """Redacted citation metadata safe to place in a drafting prompt."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=512)
    product: str = Field(min_length=1, max_length=64)
    region: str = Field(min_length=1, max_length=64)
    tenant_scope: str = Field(min_length=1, max_length=512)
    relevant_date: str = Field(min_length=1, max_length=64)
    support_status: EvidenceSupportStatus
    support_explanation: str = Field(min_length=1, max_length=2_000)
    source_revision: str = Field(min_length=1, max_length=128)


def build_local_model_citation(
    *,
    document_id: str,
    title: str,
    product: str,
    region: str,
    tenant_scope: str,
    relevant_date: str,
    support_status: EvidenceSupportStatus,
    support_explanation: str,
    source_revision: str,
) -> CitedEvidenceCitation:
    """Build one redacted citation projection shared by service and evaluator paths."""
    return CitedEvidenceCitation(
        document_id=str(redact_identifiers(document_id)),
        title=str(redact_identifiers(title)),
        product=str(redact_identifiers(product)),
        region=str(redact_identifiers(region)),
        tenant_scope=str(redact_identifiers(tenant_scope)),
        relevant_date=str(redact_identifiers(relevant_date)),
        support_status=support_status,
        support_explanation=str(redact_identifiers(support_explanation)),
        source_revision=str(redact_identifiers(source_revision)),
    )


def build_local_model_baseline_context(body: Mapping[str, object]) -> dict[str, object]:
    """Project a governed response to the evaluator's redacted, citation-safe context."""
    context: dict[str, object] = {
        "answer": body.get("answer", ""),
        "result_classification": body.get("result_classification"),
    }
    evidence = body.get("evidence")
    if not isinstance(evidence, Mapping):
        return dict(redact_identifiers(context))

    safe_citations: list[dict[str, object]] = []
    citations = evidence.get("citations", [])
    if isinstance(citations, list):
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            scope = citation.get("affected_scope", {})
            if not isinstance(scope, Mapping):
                continue
            safe_citations.append(
                build_local_model_citation(
                    document_id=str(citation.get("document_id", "")),
                    title=str(citation.get("title", "")),
                    product=str(scope.get("product", "")),
                    region=str(scope.get("region", "")),
                    tenant_scope=str(scope.get("tenant_scope", "")),
                    relevant_date=str(citation.get("relevant_date", "")),
                    support_status=citation.get("support_status", ""),
                    support_explanation=str(citation.get("support_explanation", "")),
                    source_revision=str(citation.get("source_revision", "")),
                ).model_dump(mode="json")
            )
    if safe_citations:
        context["evidence"] = {
            "support_status": evidence.get("support_status", ""),
            "support_explanation": evidence.get("support_explanation", ""),
            "citations": safe_citations,
        }
    return dict(redact_identifiers(context))


class LocalModelBaselineEvidence(BaseModel):
    """The allowlisted evidence subset used by the evaluator-only baseline."""

    model_config = ConfigDict(extra="forbid")

    support_status: EvidenceSupportStatus
    support_explanation: str = Field(min_length=1, max_length=2_000)
    citations: list[CitedEvidenceCitation] = Field(min_length=1, max_length=3)


class LocalModelBaselineInput(BaseModel):
    """The evaluator's allowlisted, non-service model input."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=_MAX_DRAFT_LENGTH)
    result_classification: str | None = Field(default=None, max_length=64)
    evidence: LocalModelBaselineEvidence | None = None


class LocalModelIntentCall(BaseModel):
    """Typed Ollama envelope for the non-authoritative intent proposal."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["intent_proposal"]
    input: LocalModelIntentRequest


class CitedEvidenceDraft(BaseModel):
    """The only evidence context a prose-drafting model is allowed to receive."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["evidence_draft"] = "evidence_draft"
    result_classification: Literal["hypothesis", "inconclusive"]
    support_status: EvidenceSupportStatus
    support_explanation: str = Field(min_length=1, max_length=2_000)
    citations: list[CitedEvidenceCitation] = Field(min_length=1, max_length=3)


class LocalModelEvidenceCall(BaseModel):
    """Typed Ollama envelope for the evidence-only prose draft."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["evidence_draft"]
    input: CitedEvidenceDraft


LocalModelCall = Annotated[
    LocalModelIntentCall | LocalModelEvidenceCall,
    Field(discriminator="task"),
]
_LOCAL_MODEL_CALL_ADAPTER = TypeAdapter(LocalModelCall)


class LocalModelDraftProposal(BaseModel):
    """A prose proposal anchored to the supplied evidence claims and document IDs."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=_MAX_DRAFT_LENGTH)
    citation_document_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=3
    )
    support_status: EvidenceSupportStatus
    cited_claims: list[Annotated[str, Field(min_length=1, max_length=2_000)]] = Field(
        min_length=1, max_length=3
    )


def _ollama_timeout_from_environment() -> float:
    timeout_text = os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60")
    try:
        timeout = float(timeout_text)
    except ValueError as error:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number.") from error
    if timeout <= 0:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number.")
    return timeout


class _OllamaHttpClient:
    """Shared local HTTP mechanics for the constrained and evaluator-only clients."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank.")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _send(self, request_data: Mapping[str, object]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(request_data).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            raise LocalModelUnavailable(
                f"Local Ollama model {self.model_name} is unavailable."
            ) from error
        if not isinstance(payload, dict):
            raise LocalModelUnavailable(
                f"Local Ollama model {self.model_name} returned an invalid payload."
            )
        output = payload.get("response")
        if not isinstance(output, str):
            raise LocalModelUnavailable(
                f"Local Ollama model {self.model_name} returned no text response."
            )
        return output


class OllamaLocalModel(_OllamaHttpClient):
    """Generic Ollama transport retained for evidence drafting compatibility."""

    @classmethod
    def from_environment(cls) -> OllamaLocalModel | None:
        """Build the generic local model adapter when explicitly configured."""
        model_name = os.environ.get("OLLAMA_MODEL_NAME")
        if not model_name:
            return None
        return cls(
            model_name=model_name,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout=_ollama_timeout_from_environment(),
        )

    def generate(self, request: LocalModelCall) -> str:
        return self._send(self._request_data(request))

    def _request_data(self, request: Mapping[str, object]) -> dict[str, object]:
        try:
            call = _LOCAL_MODEL_CALL_ADAPTER.validate_python(request)
        except ValidationError as error:
            raise LocalModelOutputInvalid(
                "Local-model requests must use a supported bounded task envelope."
            ) from error
        model_input = redact_identifiers(call.input.model_dump(mode="json", exclude={"task"}))
        return {
            "model": self.model_name,
            "prompt": (
                "Return only one JSON object matching the requested bounded schema. "
                "Never decide permissions, routes, tools, SQL, or add citations not in the input.\n"
                f"Task: {call.task}\n"
                "For intent_proposal return only metric_name and ambiguity. Set ambiguity to "
                "ambiguous and metric_name to null unless the question clearly selects exactly "
                "one listed metric. For evidence_draft return only "
                "answer, citation_document_ids, support_status, and cited_claims. The answer "
                "must copy the supplied support_explanation exactly, and every claim must be "
                "copied exactly from supplied support text.\n"
                f"Input: {json.dumps(model_input, sort_keys=True)}"
            ),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

    def readiness(self) -> dict[str, str]:
        """Check that Ollama responds for the configured model."""
        request = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=json.dumps({"name": self.model_name}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            status = "unavailable"
        else:
            status = "ready" if isinstance(payload, dict) else "unavailable"
        return {
            "provider": "ollama",
            "status": status,
            "model": self.model_name,
        }


class OllamaIntentModel(OllamaLocalModel):
    """Ollama transport restricted to the agreed request-time intent model."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ) -> None:
        if model_name != OLLAMA_INTENT_MODEL_NAME:
            raise ValueError(f"The governed intent provider requires {OLLAMA_INTENT_MODEL_NAME}.")
        super().__init__(model_name=model_name, base_url=base_url, timeout=timeout)

    @classmethod
    def from_environment(cls) -> OllamaIntentModel | None:
        """Enable the intent provider only for the agreed model configuration."""
        model_name = os.environ.get("OLLAMA_MODEL_NAME")
        if not model_name or model_name != OLLAMA_INTENT_MODEL_NAME:
            return None
        return cls(
            model_name=model_name,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout=_ollama_timeout_from_environment(),
        )


class OllamaBaselineModel(_OllamaHttpClient):
    """Evaluator-only client isolated from the constrained service transport."""

    def generate(self, context: Mapping[str, object]) -> str:
        """Generate a baseline answer from an allowlisted governed-response projection."""
        try:
            baseline_input = LocalModelBaselineInput.model_validate(context)
        except ValidationError as error:
            raise LocalModelOutputInvalid(
                "Baseline evaluation input did not match its bounded schema."
            ) from error
        model_input = redact_identifiers(baseline_input.model_dump(mode="json", exclude_none=True))
        return self._send(
            {
                "model": self.model_name,
                "prompt": (
                    "Produce a concise answer using only this governed response. "
                    "Do not add facts or identifiers that are absent from it.\n"
                    f"{json.dumps(model_input, sort_keys=True)}"
                ),
                "stream": False,
                "options": {"temperature": 0},
            }
        )


class LocalModelIntentInterpreter:
    """Use the model to choose a validated metric candidate, then route deterministically."""

    def __init__(
        self,
        model: LocalModelTransport,
        *,
        metric_names_provider: Callable[[AnswerQuestionRequest], Collection[str]],
        route_resolver: Callable[[AnswerQuestionRequest, str | None], AnalyticalRoute],
    ) -> None:
        self._model = model
        self._metric_names_provider = metric_names_provider
        self._route_resolver = route_resolver

    def interpret(self, request: AnswerQuestionRequest) -> AnalyticalIntent:
        available_metric_names = tuple(self._metric_names_provider(request))
        if not available_metric_names:
            raise LocalModelUnavailable(
                "No current validated semantic metrics are available for intent interpretation."
            )
        model_request = LocalModelIntentRequest(
            question=str(redact_identifiers(request.question)),
            requested_metric_name=str(redact_identifiers(request.requested_metric_name))
            if request.requested_metric_name is not None
            else None,
            available_metric_names=list(available_metric_names),
            conversation_context=request.conversation_context,
        )
        model_input = model_request.model_dump(mode="json", exclude={"task"})
        if model_request.conversation_context is None:
            model_input.pop("conversation_context", None)
        else:
            model_input = redact_identifiers(model_input)
        proposal = _request_and_validate(
            self._model,
            task=model_request.task,
            model_input=model_input,
            response_model=LocalModelIntentProposal,
        )
        if (
            proposal.ambiguity != "unambiguous"
            or proposal.metric_name is None
            or proposal.metric_name not in available_metric_names
        ):
            raise LocalModelOutputInvalid(
                "Local-model intent was ambiguous or selected a metric outside the validated "
                "semantic artifact."
            )
        route = self._route_resolver(request, proposal.metric_name)
        return AnalyticalIntent(route=route, metric_name=proposal.metric_name)


class LocalModelEvidenceDraftingAdapter:
    """Draft only from a redacted, already-authorized evidence citation set."""

    def __init__(self, model: LocalModelTransport) -> None:
        self._model = model

    def draft(self, response: GovernedAnalyticalResponse) -> LocalModelDraftProposal:
        context = build_local_model_evidence_context(response)
        proposal = _request_and_validate(
            self._model,
            task=context.task,
            model_input=context.model_dump(mode="json", exclude={"task"}),
            response_model=LocalModelDraftProposal,
        )
        _validate_local_model_proposal(proposal, context)
        return proposal


def build_local_model_evidence_context(
    response: GovernedAnalyticalResponse,
) -> CitedEvidenceDraft:
    """Project one governed response into the redacted evidence drafting context."""
    evidence = response.evidence
    if evidence is None or not evidence.citations:
        raise LocalModelOutputInvalid("Evidence drafting requires at least one citation.")
    if response.result_classification not in {
        ResultClassification.HYPOTHESIS,
        ResultClassification.INCONCLUSIVE,
    }:
        raise LocalModelOutputInvalid(
            "Evidence drafting requires a hypothesis or inconclusive response."
        )
    try:
        context = CitedEvidenceDraft(
            result_classification=(
                "hypothesis"
                if response.result_classification.value == "hypothesis"
                else "inconclusive"
            ),
            support_status=evidence.support_status,
            support_explanation=str(redact_identifiers(evidence.support_explanation)),
            citations=[
                build_local_model_citation(
                    document_id=citation.document_id,
                    title=citation.title,
                    product=citation.affected_scope.product,
                    region=citation.affected_scope.region,
                    tenant_scope=citation.affected_scope.tenant_scope,
                    relevant_date=citation.relevant_date.isoformat(),
                    support_status=citation.support_status,
                    support_explanation=citation.support_explanation,
                    source_revision=citation.source_revision,
                )
                for citation in evidence.citations
            ],
        )
    except ValidationError as error:
        raise LocalModelOutputInvalid(
            "Authorized evidence did not match the drafting schema."
        ) from error
    return context


def _validate_local_model_proposal(
    proposal: LocalModelDraftProposal,
    context: CitedEvidenceDraft,
) -> None:
    citation_ids = {citation.document_id for citation in context.citations}
    if len(set(proposal.citation_document_ids)) != len(proposal.citation_document_ids) or not set(
        proposal.citation_document_ids
    ).issubset(citation_ids):
        raise LocalModelOutputInvalid("Local-model prose cited an unapproved document.")
    if proposal.support_status != context.support_status:
        raise LocalModelOutputInvalid("Local-model prose changed the evidence status.")
    claim_sources = {
        _normalise_text(context.support_explanation),
        *(_normalise_text(citation.support_explanation) for citation in context.citations),
    }
    if any(_normalise_text(claim) not in claim_sources for claim in proposal.cited_claims):
        raise LocalModelOutputInvalid("Local-model prose cited an unapproved claim.")
    classification_label = (
        "Hypothesis" if context.result_classification == "hypothesis" else "Inconclusive"
    )
    safe_answer_forms = {
        _normalise_text(context.support_explanation),
        _normalise_text(f"{classification_label}: {context.support_explanation}"),
    }
    if _normalise_text(proposal.answer) not in safe_answer_forms:
        raise LocalModelOutputInvalid(
            "Local-model prose must be an extractive governed evidence draft."
        )
    if str(redact_identifiers(proposal.answer)) != proposal.answer:
        raise LocalModelOutputInvalid("Local-model prose contained a direct identifier.")


def validate_local_model_draft(
    response: GovernedAnalyticalResponse,
    draft: LocalModelDraftProposal,
) -> LocalModelDraftProposal:
    """Reapply the deterministic evidence-draft policy at the service seam."""
    if not isinstance(draft, LocalModelDraftProposal):
        raise LocalModelOutputInvalid("Local-model drafting returned an invalid typed result.")
    _validate_local_model_proposal(draft, build_local_model_evidence_context(response))
    return draft


def local_model_readiness(model: LocalModelTransport | None) -> dict[str, str | None]:
    """Return a safe readiness projection for the selected model boundary."""
    if model is None:
        return {"provider": "none", "status": "disabled", "model": None}
    if isinstance(model, OllamaLocalModel):
        return model.readiness()
    return {"provider": "custom", "status": "configured", "model": None}


def _request_and_validate(
    model: LocalModelTransport,
    *,
    task: str,
    model_input: Mapping[str, object],
    response_model: type[_ModelResponse],
) -> _ModelResponse:
    try:
        call = _LOCAL_MODEL_CALL_ADAPTER.validate_python({"task": task, "input": model_input})
    except ValidationError as error:
        raise LocalModelOutputInvalid(f"{task} input did not match its bounded schema.") from error
    try:
        raw_output = model.generate(call)
    except LocalModelError:
        raise
    except Exception as error:
        raise LocalModelUnavailable("The configured local model failed.") from error
    try:
        payload = json.loads(raw_output)
    except (TypeError, ValueError) as error:
        raise LocalModelOutputInvalid("Local-model output was not valid JSON.") from error
    if not isinstance(payload, dict):
        raise LocalModelOutputInvalid("Local-model output must be a JSON object.")
    try:
        return response_model.model_validate(payload)
    except ValidationError as error:
        raise LocalModelOutputInvalid(
            "Local-model output did not match its bounded schema."
        ) from error


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())

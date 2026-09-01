"""Evaluate retrieval and grounded generation separately (issue #86).

Retrieval quality (Recall@K, Precision@K, MRR, nDCG@K) is scored against
gold relevance keyed by Evidence Revision — `(source_document_id,
source_revision)`, the same kind of revision identity `lightrag.py` already
authorizes and de-duplicates by (there it also carries `chunk_id`; here it
does not, since gold relevance is revision-level, not chunk-level) — never
by chunk_id alone. Generation quality (context relevance, faithfulness,
answer relevance) is scored by RAGAS through an optional, injectable judge:
`RagJudge.from_environment()` mirrors `OllamaLocalModel.from_environment()`
exactly — returns `None` when unconfigured, and a configured judge that
fails is reported `"unavailable"`, never a fabricated score. See
docs/adr/0014-rag-evaluation-separates-retrieval-from-generation.md.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Literal

from .evaluation_runner import EvaluatorFinding, ScorecardCategory
from .evidence import EvidenceDocument
from .rag_evaluation_dataset import RagEvaluationCase, RagEvaluationDataset

EVALUATOR_VERSION = "1.0.0"
CHUNKING_STRATEGY_VERSION = "fixed-chunk-v1"
"""There is no automated chunking pipeline in this codebase yet — every
Evidence Revision is already split into explicit, hand-assigned chunks. This
is an honest placeholder identifier for that fact, not a fabricated
versioning scheme; it exists so results stay comparable once a real
chunking configuration is introduced."""


# --- Retrieval (IR metrics) ---------------------------------------------


@dataclass(frozen=True)
class RagRetrievalResult:
    case_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    passed: bool
    retrieved_revisions: tuple[tuple[str, str], ...]


def _gold_relevance_key(document: EvidenceDocument) -> tuple[str, str]:
    """The Evidence Revision identity a retrieved document is matched against
    gold relevance by. Deliberately named distinctly from `lightrag.py`'s
    `_revision_key` — that one is a 3-tuple that also carries `chunk_id`; this
    one is revision-level only, matching `RelevantEvidenceRevision`'s gold
    label shape, not a drop-in equivalent."""
    return (document.source_document_id or document.document_id, document.source_revision)


def evaluate_rag_retrieval(
    cases: Sequence[RagEvaluationCase],
    retrieve: Callable[[RagEvaluationCase], Sequence[EvidenceDocument]],
) -> list[RagRetrievalResult]:
    """Score ranked retrieval against gold Evidence Revisions, independent of
    whatever the generated answer says."""
    results = []
    for case in cases:
        gold = {
            (item.source_document_id, item.source_revision)
            for item in case.gold_relevant_revisions
        }
        top_k = list(retrieve(case))[: case.k]
        retrieved_keys = [_gold_relevance_key(document) for document in top_k]
        hits = [key for key in retrieved_keys if key in gold]

        recall = len(set(hits)) / len(gold) if gold else 1.0
        precision = len(hits) / case.k if case.k else 1.0
        reciprocal_rank = next(
            (1.0 / (rank + 1) for rank, key in enumerate(retrieved_keys) if key in gold),
            0.0,
        )
        dcg = sum(
            1.0 / math.log2(rank + 2) for rank, key in enumerate(retrieved_keys) if key in gold
        )
        ideal_hit_count = min(len(gold), case.k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hit_count))
        ndcg = dcg / idcg if idcg else 1.0

        results.append(
            RagRetrievalResult(
                case_id=case.case_id,
                recall_at_k=recall,
                precision_at_k=precision,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                passed=recall >= case.minimum_recall_at_k,
                retrieved_revisions=tuple(retrieved_keys),
            )
        )
    return results


# --- Generation (RAGAS, optional judge) ---------------------------------


class RagJudgeUnavailable(RuntimeError):
    """Raised when a configured RAGAS judge cannot produce a result."""


class RagJudge:
    """Wraps RAGAS's context-relevance/faithfulness/answer-relevancy scorers
    against a configured LLM + embeddings pair. Constructed only via
    `from_environment()` in real usage, matching
    `OllamaLocalModel.from_environment()`'s optional-construction pattern —
    ragas/openai are imported lazily, only once a judge is actually
    configured, mirroring `observability.py`'s lazy `_load_mlflow()`."""

    def __init__(self, *, llm_model_name: str, embedding_model_name: str, base_url: str):
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.base_url = base_url
        self._scorers: tuple[Any, Any, Any] | None = None

    @classmethod
    def from_environment(cls) -> RagJudge | None:
        """Build the RAGAS judge only when explicitly configured — unset by
        default, matching every other optional-model boundary in this repo."""
        llm_model_name = os.environ.get("RAGAS_JUDGE_MODEL_NAME")
        if not llm_model_name:
            return None
        return cls(
            llm_model_name=llm_model_name,
            embedding_model_name=os.environ.get(
                "RAGAS_JUDGE_EMBEDDING_MODEL_NAME", "nomic-embed-text"
            ),
            base_url=os.environ.get("RAGAS_JUDGE_BASE_URL", "http://127.0.0.1:11434/v1"),
        )

    def _get_scorers(self) -> tuple[Any, Any, Any]:
        if self._scorers is None:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextRelevance,
                Faithfulness,
            )

            client = AsyncOpenAI(api_key="ollama", base_url=self.base_url)
            llm = llm_factory(self.llm_model_name, provider="openai", client=client)
            embeddings = embedding_factory(
                "openai", model=self.embedding_model_name, client=client
            )
            self._scorers = (
                ContextRelevance(llm=llm),
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
            )
        return self._scorers

    async def ascore(
        self, *, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> dict[str, float]:
        context_relevance, faithfulness, answer_relevancy = self._get_scorers()
        try:
            context_result = await context_relevance.ascore(
                user_input=user_input, retrieved_contexts=retrieved_contexts
            )
            faithfulness_result = await faithfulness.ascore(
                user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
            )
            relevancy_result = await answer_relevancy.ascore(
                user_input=user_input, response=response
            )
        except Exception as error:
            # RAGAS/openai/httpx can raise many transport- and provider-specific
            # exception types for an unreachable or misbehaving local judge; the
            # safety property that matters is "never let a broken judge crash the
            # run or silently return a score" — caught broadly and re-raised as one
            # typed boundary error, matching local_model.py's own broad catch
            # around an arbitrary LocalModelTransport.generate() call in
            # `_request_and_validate` (`except Exception as error: raise
            # LocalModelUnavailable(...) from error`), not the narrower
            # single-urllib-call catch in `_OllamaHttpClient._send`.
            raise RagJudgeUnavailable(
                f"RAGAS judge {self.llm_model_name!r} is unavailable."
            ) from error
        return {
            "context_quality": float(context_result.value),
            "faithfulness": float(faithfulness_result.value),
            "answer_relevance": float(relevancy_result.value),
        }


@dataclass(frozen=True)
class RagGenerationResult:
    case_id: str
    status: Literal["scored", "not_configured", "unavailable"]
    context_quality: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    detail: str = ""


def evaluate_rag_generation(
    cases: Sequence[RagEvaluationCase],
    answer: Callable[[RagEvaluationCase], tuple[str, list[str]]],
    judge: RagJudge | None,
) -> list[RagGenerationResult]:
    """Score grounded generation quality via RAGAS when a judge is
    configured; otherwise honestly report `not_configured` for every case —
    never a fabricated pass or fail."""
    results = []
    for case in cases:
        if judge is None:
            results.append(
                RagGenerationResult(
                    case.case_id, "not_configured", detail="No RAGAS judge is configured."
                )
            )
            continue
        response_text, retrieved_contexts = answer(case)
        try:
            scores = asyncio.run(
                judge.ascore(
                    user_input=case.question,
                    response=response_text,
                    retrieved_contexts=retrieved_contexts,
                )
            )
        except RagJudgeUnavailable as error:
            results.append(RagGenerationResult(case.case_id, "unavailable", detail=str(error)))
            continue
        results.append(RagGenerationResult(case.case_id, "scored", **scores))
    return results


# --- Scorecard -----------------------------------------------------------


@dataclass(frozen=True)
class RagEvaluationScorecard:
    dataset_version: str
    evaluator_version: str
    configuration_versions: Mapping[str, str]
    generated_at: datetime
    retrieval: ScorecardCategory
    generation: ScorecardCategory
    # AC3 requires Recall@K, Precision@K, MRR, and nDCG@K to be *reported*, not
    # only used internally to decide pass/fail — these are the corpus-level
    # means (MRR is the mean of per-case reciprocal rank, matching this
    # codebase's existing evaluation.py convention).
    retrieval_metrics: Mapping[str, float]
    # Mean RAGAS scores over cases that were actually judge-scored; empty when
    # no case reached status="scored" (e.g. no judge configured).
    generation_metrics: Mapping[str, float]


def _retrieval_category(results: Sequence[RagRetrievalResult]) -> ScorecardCategory:
    findings = [
        EvaluatorFinding(
            "recall_at_k",
            result.passed,
            (
                ""
                if result.passed
                else f"{result.case_id}: recall@k={result.recall_at_k:.2f}"
            ),
        )
        for result in results
    ]
    passed = sum(1 for finding in findings if finding.passed)
    failed = len(findings) - passed
    return ScorecardCategory(
        name="retrieval",
        passed=passed,
        failed=failed,
        total=len(findings),
        pass_rate=(passed / len(findings)) if findings else 1.0,
        details=tuple(finding.detail for finding in findings if not finding.passed),
    )


def _retrieval_metrics(results: Sequence[RagRetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "recall_at_k": fmean(result.recall_at_k for result in results),
        "precision_at_k": fmean(result.precision_at_k for result in results),
        "mrr": fmean(result.reciprocal_rank for result in results),
        "ndcg_at_k": fmean(result.ndcg_at_k for result in results),
    }


def _generation_metrics(results: Sequence[RagGenerationResult]) -> dict[str, float]:
    scored = [result for result in results if result.status == "scored"]
    if not scored:
        return {}
    return {
        "context_quality": fmean(result.context_quality for result in scored),
        "faithfulness": fmean(result.faithfulness for result in scored),
        "answer_relevance": fmean(result.answer_relevance for result in scored),
    }


def _generation_category(results: Sequence[RagGenerationResult]) -> ScorecardCategory:
    passed = sum(1 for result in results if result.status != "unavailable")
    failed = len(results) - passed
    details = tuple(f"{result.case_id}: {result.status} — {result.detail}" for result in results)
    return ScorecardCategory(
        name="generation",
        passed=passed,
        failed=failed,
        total=len(results),
        pass_rate=(passed / len(results)) if results else 1.0,
        details=details,
    )


def run_rag_dataset(
    dataset: RagEvaluationDataset,
    *,
    retrieve: Callable[[RagEvaluationCase], Sequence[EvidenceDocument]],
    answer: Callable[[RagEvaluationCase], tuple[str, list[str]]],
    judge: RagJudge | None,
    evaluator_version: str = EVALUATOR_VERSION,
    configuration_versions: Mapping[str, str] | None = None,
) -> RagEvaluationScorecard:
    retrieval_results = evaluate_rag_retrieval(dataset.cases, retrieve)
    generation_results = evaluate_rag_generation(dataset.cases, answer, judge)
    return RagEvaluationScorecard(
        dataset_version=dataset.dataset_version,
        evaluator_version=evaluator_version,
        configuration_versions=dict(configuration_versions or {}),
        generated_at=datetime.now(UTC),
        retrieval=_retrieval_category(retrieval_results),
        generation=_generation_category(generation_results),
        retrieval_metrics=_retrieval_metrics(retrieval_results),
        generation_metrics=_generation_metrics(generation_results),
    )

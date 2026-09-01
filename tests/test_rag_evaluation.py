"""Tests for RAG evaluation (issue #86): IR-metric arithmetic, the optional
RAGAS judge's honest availability contract, and AC5's core proof that a
retrieval regression and a generation regression are independent scorecard
signals.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from growth_data_agent.evaluation_dataset import EvaluationCaseProvenance, EvaluationSplit
from growth_data_agent.evidence import EvidenceDocument, EvidenceSupportStatus
from growth_data_agent.rag_evaluation import (
    RagJudge,
    RagJudgeUnavailable,
    evaluate_rag_generation,
    evaluate_rag_retrieval,
    run_rag_dataset,
)
from growth_data_agent.rag_evaluation_dataset import (
    RagEvaluationCase,
    RagEvaluationDataset,
    RagEvaluationDatasetStore,
    RelevantEvidenceRevision,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "evaluations/rag_dataset/v1/cases.json"


def _document(document_id: str, *, revision: str = "synthetic-v1") -> EvidenceDocument:
    return EvidenceDocument(
        document_id=document_id,
        title=document_id,
        text="x",
        product="Jira",
        region="APAC",
        tenant_ids=[],
        tenant_scope="x",
        classification="internal",
        identifier_entitlement="none",
        relevant_date=date(2026, 6, 1),
        freshness=datetime(2026, 6, 1, tzinfo=UTC),
        support_status=EvidenceSupportStatus.SUPPORTS,
        support_explanation="x",
        source_document_id=document_id,
        source_revision=revision,
    )


def _case(**overrides) -> RagEvaluationCase:
    defaults = dict(
        case_id="t1",
        agent_user_id="data_analyst",
        product="Jira",
        region="APAC",
        retrieval_query="q",
        question="q?",
        permitted_scope="x",
        k=3,
        gold_relevant_revisions=[
            RelevantEvidenceRevision(source_document_id="a", source_revision="synthetic-v1"),
            RelevantEvidenceRevision(source_document_id="b", source_revision="synthetic-v1"),
        ],
        split=EvaluationSplit.DEVELOPMENT,
        provenance=EvaluationCaseProvenance(source_type="synthetic", source_reference="unit-test"),
    )
    defaults.update(overrides)
    return RagEvaluationCase(**defaults)


# --- IR metrics -----------------------------------------------------------


def test_recall_precision_mrr_ndcg_arithmetic() -> None:
    case = _case()
    # Irrelevant doc first, then both gold docs at rank 2 and rank 3.
    retrieved = [_document("c"), _document("a"), _document("b")]

    [result] = evaluate_rag_retrieval([case], lambda _c: retrieved)

    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 2 / 3
    assert result.reciprocal_rank == 0.5
    import math

    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(result.ndcg_at_k - dcg / idcg) < 1e-9


def test_recall_at_k_below_minimum_fails_the_case() -> None:
    case = _case(k=3, minimum_recall_at_k=1.0)
    # Only one of two gold documents retrieved.
    retrieved = [_document("a"), _document("c"), _document("d")]

    [result] = evaluate_rag_retrieval([case], lambda _c: retrieved)

    assert result.recall_at_k == 0.5
    assert not result.passed


def test_gold_matching_is_keyed_by_revision_not_just_document_id() -> None:
    case = _case(
        gold_relevant_revisions=[
            RelevantEvidenceRevision(source_document_id="a", source_revision="v2"),
        ]
    )
    # Same document_id, but a stale/superseded revision — must not count as a hit.
    retrieved = [_document("a", revision="v1")]

    [result] = evaluate_rag_retrieval([case], lambda _c: retrieved)

    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0


def test_empty_retrieval_scores_zero_not_a_crash() -> None:
    case = _case()

    [result] = evaluate_rag_retrieval([case], lambda _c: [])

    assert result.recall_at_k == 0.0
    assert result.ndcg_at_k == 0.0


# --- RAGAS judge availability ----------------------------------------------


def test_from_environment_returns_none_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("RAGAS_JUDGE_MODEL_NAME", raising=False)
    assert RagJudge.from_environment() is None


def test_from_environment_reads_configured_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_JUDGE_MODEL_NAME", "llama3.1:8b")
    monkeypatch.setenv("RAGAS_JUDGE_EMBEDDING_MODEL_NAME", "nomic-embed-text")
    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "http://127.0.0.1:11434/v1")

    judge = RagJudge.from_environment()

    assert judge is not None
    assert judge.llm_model_name == "llama3.1:8b"
    assert judge.embedding_model_name == "nomic-embed-text"
    assert judge.base_url == "http://127.0.0.1:11434/v1"


class _FakeJudge:
    """A duck-typed stand-in for RagJudge — evaluate_rag_generation only calls
    `.ascore(...)`, so tests never need a real Ollama server or ragas call."""

    def __init__(self, *, fail: bool = False, scores: dict[str, float] | None = None):
        self.fail = fail
        self.scores = scores or {
            "context_quality": 0.9,
            "faithfulness": 0.85,
            "answer_relevance": 0.8,
        }

    async def ascore(self, *, user_input: str, response: str, retrieved_contexts: list[str]):
        if self.fail:
            raise RagJudgeUnavailable("fake judge is unavailable")
        return self.scores


def test_evaluate_rag_generation_without_a_judge_is_honestly_not_configured() -> None:
    case = _case()

    [result] = evaluate_rag_generation([case], lambda _c: ("answer", ["context"]), judge=None)

    assert result.status == "not_configured"
    assert result.context_quality is None


def test_evaluate_rag_generation_with_a_working_judge_records_real_scores() -> None:
    case = _case()
    judge = _FakeJudge()

    [result] = evaluate_rag_generation([case], lambda _c: ("answer", ["context"]), judge=judge)

    assert result.status == "scored"
    assert result.faithfulness == 0.85


def test_a_failing_judge_records_unavailable_not_a_fabricated_pass() -> None:
    case = _case()
    judge = _FakeJudge(fail=True)

    [result] = evaluate_rag_generation([case], lambda _c: ("answer", ["context"]), judge=judge)

    assert result.status == "unavailable"
    assert result.context_quality is None


# --- AC5: retrieval and generation are independent scorecard signals -------


def _real_dataset() -> RagEvaluationDataset:
    return RagEvaluationDatasetStore(_DATASET_PATH).load()


def test_a_retrieval_only_regression_does_not_fail_generation() -> None:
    dataset = _real_dataset()

    scorecard = run_rag_dataset(
        dataset,
        retrieve=lambda _case: [],  # nothing retrieved for any case
        answer=lambda _case: ("a governed answer", ["a supporting context"]),
        judge=None,  # not_configured always "passes" — isolates the retrieval break
    )

    assert scorecard.retrieval.failed == len(dataset.cases)
    assert scorecard.generation.failed == 0


def test_scorecard_reports_all_four_ir_metrics_and_ragas_measures() -> None:
    """AC3/AC2: Recall@K, Precision@K, MRR, and nDCG@K must be *reported* on
    the scorecard, not only consumed internally to decide pass/fail — same
    for the three RAGAS measures once a judge actually scores a case."""
    dataset = _real_dataset()
    correct_documents = {
        case.case_id: [
            _document(revision.source_document_id, revision=revision.source_revision)
            for revision in case.gold_relevant_revisions
        ]
        for case in dataset.cases
    }

    scorecard = run_rag_dataset(
        dataset,
        retrieve=lambda case: correct_documents[case.case_id],
        answer=lambda _case: ("a governed answer", ["a supporting context"]),
        judge=_FakeJudge(),
    )

    assert set(scorecard.retrieval_metrics) == {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
    }
    assert scorecard.retrieval_metrics["recall_at_k"] == 1.0
    assert set(scorecard.generation_metrics) == {
        "context_quality",
        "faithfulness",
        "answer_relevance",
    }
    assert scorecard.generation_metrics["faithfulness"] == 0.85


def test_generation_metrics_are_empty_when_no_judge_is_configured() -> None:
    dataset = _real_dataset()

    scorecard = run_rag_dataset(
        dataset,
        retrieve=lambda _case: [],
        answer=lambda _case: ("a governed answer", ["a supporting context"]),
        judge=None,
    )

    assert scorecard.generation_metrics == {}


def test_a_generation_only_regression_does_not_fail_retrieval() -> None:
    dataset = _real_dataset()
    correct_documents = {
        case.case_id: [
            _document(revision.source_document_id, revision=revision.source_revision)
            for revision in case.gold_relevant_revisions
        ]
        for case in dataset.cases
    }

    scorecard = run_rag_dataset(
        dataset,
        retrieve=lambda case: correct_documents[case.case_id],
        answer=lambda _case: ("a governed answer", ["a supporting context"]),
        judge=_FakeJudge(fail=True),  # every case's judge call fails
    )

    assert scorecard.retrieval.failed == 0
    assert scorecard.generation.failed == len(dataset.cases)

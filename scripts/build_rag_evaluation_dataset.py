"""Build the versioned RAG Evaluation Dataset (issue #86) from the same
grounded default-corpus scenarios already proven by #84/#85 — each case's
gold-relevant Evidence Revision is a real document in
`growth_data_agent.synthetic.evidence_corpus()`, not invented.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from growth_data_agent.evaluation_dataset import EvaluationCaseProvenance, EvaluationSplit
from growth_data_agent.rag_evaluation_dataset import (
    RagEvaluationCase,
    RagEvaluationDataset,
    RelevantEvidenceRevision,
)

_DATASET_PATH = Path("evaluations/rag_dataset/v1/cases.json")
_DATASET_VERSION = "1.0.0"


def _build_dataset() -> RagEvaluationDataset:
    cases = [
        RagEvaluationCase(
            case_id="rag-jira-apac-provisioning-incident",
            agent_user_id="data_analyst",
            product="Jira",
            region="APAC",
            retrieval_query="Jira APAC 51-200 paid provisioning June 2026 decline",
            question="What evidence may explain the APAC 51–200-seat Tenant decline?",
            permitted_scope="data_analyst — unrestricted",
            k=3,
            gold_relevant_revisions=[
                RelevantEvidenceRevision(
                    source_document_id="jira-apac-paid-provisioning-incident",
                    source_revision="synthetic-v1",
                    chunk_id="jira-apac-paid-provisioning-incident:chunk:0",
                )
            ],
            split=EvaluationSplit.DEVELOPMENT,
            provenance=EvaluationCaseProvenance(
                source_type="synthetic",
                source_reference="evaluations/fixtures.json#apac-incident-retrieval",
                notes="Migrated to Evidence-Revision-keyed gold relevance for issue #86.",
            ),
        ),
        RagEvaluationCase(
            case_id="rag-confluence-americas-acquisition-campaign",
            agent_user_id="data_analyst",
            product="Confluence",
            region="Americas",
            retrieval_query="Confluence Americas 11-50 acquisition campaign June 2026",
            question=(
                "What evidence may explain the Americas 11–50-seat Confluence New PEU "
                "movement after the acquisition campaign?"
            ),
            permitted_scope="data_analyst — unrestricted",
            k=3,
            gold_relevant_revisions=[
                RelevantEvidenceRevision(
                    source_document_id="confluence-americas-acquisition-campaign",
                    source_revision="synthetic-v1",
                    chunk_id="confluence-americas-acquisition-campaign:chunk:0",
                )
            ],
            split=EvaluationSplit.VALIDATION,
            provenance=EvaluationCaseProvenance(
                source_type="synthetic",
                source_reference=(
                    "tests/test_confluence_new_peu.py::"
                    "test_data_analyst_receives_scoped_confluence_campaign_evidence"
                ),
            ),
        ),
        RagEvaluationCase(
            case_id="rag-confluence-emea-onboarding-email-regression",
            agent_user_id="data_analyst",
            product="Confluence",
            region="EMEA",
            retrieval_query="Confluence EMEA 51-200 onboarding email regression June 2026",
            question=(
                "What evidence may explain the Confluence EMEA 51–200-seat New MAU "
                "decline after the onboarding-email regression?"
            ),
            permitted_scope="data_analyst — unrestricted",
            k=3,
            gold_relevant_revisions=[
                RelevantEvidenceRevision(
                    source_document_id="confluence-emea-onboarding-email-regression",
                    source_revision="synthetic-v1",
                    chunk_id="confluence-emea-onboarding-email-regression:chunk:0",
                )
            ],
            split=EvaluationSplit.HELD_OUT,
            provenance=EvaluationCaseProvenance(
                source_type="synthetic",
                source_reference=(
                    "tests/test_new_mau.py::"
                    "test_data_analyst_receives_emea_confluence_new_mau_regression_hypothesis"
                ),
            ),
        ),
    ]
    return RagEvaluationDataset(
        dataset_version=_DATASET_VERSION,
        published_at=date(2026, 9, 1),
        cases=cases,
    )


def main() -> None:
    dataset = _build_dataset()
    _DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATASET_PATH.write_text(dataset.model_dump_json(indent=2) + "\n")
    print(f"Built {_DATASET_PATH} with {len(dataset.cases)} RAG Evaluation Cases.")


if __name__ == "__main__":
    main()

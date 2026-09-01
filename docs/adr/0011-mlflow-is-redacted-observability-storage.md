# MLflow is redacted observability storage, not evidence or reviewer-data storage

MLflow stores redacted Execution Trace metadata, evaluator results, and versioned
experiment references, never raw prompts, answers, SQL, Evidence Revision text,
or direct identifiers. Authorised review of live content occurs through a
separate governed path that revalidates access to the underlying Conversation
and Evidence Revisions; observability delivery failure does not change a
governed analytical outcome.

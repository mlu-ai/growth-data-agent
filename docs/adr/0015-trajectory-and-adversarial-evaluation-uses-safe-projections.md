# Trajectory and adversarial evaluation uses safe projections

DeepEval-linked trajectory cases and Promptfoo adversarial tests are offline
evaluation artifacts. They exercise a private evaluator projection over the
same governed `answer_question` workflow and inspect only redacted response
fields plus allowlisted trace metadata. They
never receive private reasoning, raw SQL, evidence bodies, direct identifiers,
or credentials.

Trajectory evaluation reports request interpretation, tool selection, tool
argument metadata, tool execution, output handling, and final-goal outcome as
separate stage findings. Multi-turn continuity is a separate category that
requires one conversation reference and distinct per-turn traces; it is not a
composite with safety, retrieval, or generation quality.

Promptfoo cases declare the permitted Regions and tools for each adversarial
request. The deterministic boundary checker fails when the response or actual
observed tool metadata widens either boundary, and denied responses must not
carry evidence, Candidate Causal Factors, or direct identifiers. The matrix
contains no token values and its provider targets only the evaluator endpoint,
which returns a safe response projection and actual tool-span names/statuses
but never a response body. The endpoint fails closed unless both an ordinary
Agent User bearer token and a separate evaluator capability token are supplied.

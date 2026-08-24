# Authorize before data retrieval

The semantic gateway derives row and column constraints, and the evidence
retriever derives document filters, from a user's entitlement before any data
reaches a model. Those constraints use the same permitted product, region, and
Tenant scope across structured data, documents, and graph traversal. Agent
guardrails and output redaction supplement this control but do not replace it.

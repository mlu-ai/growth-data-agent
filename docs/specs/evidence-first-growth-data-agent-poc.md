# Spec: evidence-first growth data agent POC

## Objective

Deliver a local-first, read-only analytical service that lets an authenticated
Agent User ask for a governed metric definition, a Driver Decomposition, or an
evidence-backed Hypothesis. The service must never present a Driver
Decomposition or Hypothesis as a causal conclusion.

The first delivery is intentionally narrow: it proves the workflow for **Jira
New Paid Enabled User (New PEU)** using synthetic data. It answers a canonical
definition, explains a known month-on-month decline by approved dimensions,
and retrieves access-filtered evidence that may explain the affected segment.

## Product boundary

The agent is an evidence-first analyst, not a general chatbot. It is read-only
and returns a governed analytical response for this primary seam:

`answer_question(authenticated Agent User, question, request time) -> governed analytical response`

Every response includes the applicable access scope, freshness, and caveats.
When a canonical metric is involved it also includes its semantic definition
and version. The response may include only data and evidence allowed by the
Agent User's Access Profile.

## First vertical

### Metric and semantic contract

The first canonical metric is **Jira New PEU**. A Product User qualifies when
they receive Jira paid access for the first time. A later restoration of access
does not make that Product User new again. The grain is a Product User in a
Tenant and product, not a Person. The same Person can therefore contribute a
New PEU to Jira and separately to Confluence.

dbt and MetricFlow are the semantic authority for the metric's formula,
dimensions, grain, time logic, version, and validation status. Postgres is the
initial analytical serving store. The service invokes MetricFlow to compile the
validated semantic definition to SQL and runs that SQL against Postgres; it
must not recreate the formula in agent prompts or application code.

### Scenario

Synthetic data has eighteen months of daily events and is reported monthly.
For the first scenario, Jira New PEU falls from 4,000 in May to 3,440 in June
(-14%). The APAC, 51–200 Seat Tier Tenant segment accounts for 420 of the 560
decline (75%). An evidence document describes a paid-provisioning incident in
that scope and period. It supports a possible explanation, not causal proof.

### Acceptance questions

The following are the required first-vertical questions:

1. “What is Jira New PEU?”
2. “Why did Jira New PEU fall from May to June?”
3. “What evidence may explain the APAC 51–200-seat Tenant decline?”

The service must answer each question correctly for a Data Analyst and must
also demonstrate governed scope for an APAC Regional Manager.

## User stories

1. As a Data Analyst, I can ask for Jira New PEU and receive the dbt/MetricFlow
   definition, its semantic version, grain, time rule, source freshness, and
   a citation to the semantic authority.
2. As a Data Analyst, I can ask why Jira New PEU changed between two periods
   and receive a ranked Driver Decomposition by permitted semantic dimensions.
3. As a Data Analyst, I can see each driver's absolute and percentage
   contribution, the comparison period, and the residual or reconciliation
   result.
4. As a Data Analyst, I can ask for evidence about an affected segment and
   receive only retrieved documents that are in scope, with a short evidence
   chain and source freshness.
5. As a Data Analyst, I receive the APAC 51–200 Seat Tier result as an
   observation and the provisioning incident as a Hypothesis, never as an
   established root cause.
6. As an APAC Regional Manager, I can investigate permitted APAC data and
   evidence but cannot infer or retrieve other regions through a broad query,
   decomposition, graph traversal, or document citation.
7. As a Jira Product Manager, I can later use Jira-scoped data and evidence
   across permitted regions, but not Confluence-scoped material.
8. As a Confluence Product Manager, I can later use Confluence-scoped data and
   evidence, but not Jira-scoped material.
9. As a Customer Success Manager with Tenant-portfolio entitlement, I can
   later view permitted direct identifiers only in a bounded, audited answer.
10. As any Agent User without direct-identifier entitlement, I receive no
    direct identifiers even if a retrieved document or graph node contains
    them.
11. As an Agent User, I see the resolved product, region, Tenant, and column
    scope that governed my answer, without exposing forbidden policy details.
12. As an Agent User, I receive an explicit safe refusal when the requested
    data, evidence, or identifier is outside my entitlement.
13. As an Agent User, I receive a clear limitation when the semantic artifact
    is stale or failed validation rather than an answer represented as
    canonical.
14. As an Agent User, I can request a metric absent from the semantic
    authority and receive either a refusal or a clearly labelled Provisional
    Metric, calculated only from permitted inputs and accompanied by its
    formula and caveats.
15. As an Agent User, I can explicitly approve creation of a data-team
    verification request for a Metric Definition Gap; the agent does not send
    it autonomously.
16. As a Data Analyst, I can ask who owns a metric, model, or dataset and get
    an ownership answer from published DataHub metadata when available.
17. As a Data Analyst, I can ask for related evidence after a driver is known;
    the evidence chain may traverse metric, segment, Tenant, incident,
    campaign, or accountable team in the derived graph.
18. As an Agent User, I receive an explicit incomplete-evidence outcome when
    retrieval cannot support a Hypothesis.
19. As an Agent User asking outside the analytical scope, I receive a concise
    redirection to the supported question types instead of speculative chat.
20. As an operator, I can see request, policy, route, tool, retrieval, and
    evaluation traces in MLflow without raw unauthorized identifiers in logs.
21. As an evaluator, I can replay deterministic synthetic fixtures to check
    semantic provenance, arithmetic, authorization, retrieval, evidence
    wording, and safe refusal behaviour.
22. As a future experiment owner, I can ask about an eligible registered
    randomized experiment only after it passes the causal eligibility gate;
    otherwise I receive descriptive findings or a reviewed analysis plan.
23. As an operator, I can keep the service usable if DataHub is unavailable:
    canonical metric logic still comes from the last validated dbt artifact,
    while catalog-dependent answers disclose degraded freshness.
24. As an operator, I can change the locally served generation model only after
    comparing it with the baseline evaluation suite.

## Functional requirements

### Request routing and specialist liaison

The lead orchestrator classifies each request as easy, medium, or difficult and
routes it to a tightly scoped specialist:

- **Easy:** semantic-definition specialist for canonical definitions.
- **Medium:** data specialist for Driver Decomposition; evidence specialist
  follows only after a quantified driver is known.
- **Difficult:** lead combines the bounded outputs, or returns an explicit
  limitation when evidence, entitlement, or causal eligibility is inadequate.
- **Causal:** an ML specialist is not part of ordinary routing. It is invoked
  only after a causal eligibility gate and required human review.

The specialists exchange typed, attributed results rather than unrestricted
free-form deliberation. The lead remains responsible for the final response
contract and safety wording.

### Governed semantic querying

The semantic gateway accepts the Agent User's resolved entitlement and a
metric request. It verifies the last validated dbt semantic artifact, asks
MetricFlow to plan the query, derives allowed product, region, Tenant, and
column constraints, and executes only the resulting constrained query against
Postgres.

The gateway rejects canonical answers if semantic validation is not current.
DataHub may enrich ownership, classification, and discovery; it is never a
second metric-logic authority.

For the POC, the approved dimensions include Product, Region, Paid Tenant
Tenure, and Seat Tier. Paid Tenant Tenure is time since the Tenant began a paid
subscription; Seat Tier is one of 1–10, 11–50, 51–200, or 201+ purchased
seats. Region is the Tenant billing or home region.

### Driver Decomposition

The data specialist calculates the movement between the named periods then
decomposes it through eligible dimensions. It returns ranked contributions,
counts, period labels, and reconciliation. A decomposition is explicitly an
observed quantitative explanation; it must not use causal language.

The initial scenario must surface APAC / 51–200 Seat Tier Tenants as 420 of
the 560 Jira New PEU decline. The response calls this a leading driver and
states that it does not establish cause.

### Evidence retrieval and graph

LlamaIndex retrieves from Qdrant through a `VectorEvidenceStore` boundary.
Document payload filters for product, region, Tenant, classification, and
identifier entitlement are derived before retrieval, so unauthorized content
does not enter the model context.

Apache AGE stores a derived evidence graph populated from approved dbt,
DataHub, and document-ingestion metadata. It is used after driver analysis for
ownership and multi-hop evidence chains. It cannot act as a semantic authority
or permissions engine. The relational entitlement evaluation remains the
authority before a graph query is formed.

An evidence answer cites document identity, relevant date or freshness, the
affected scope, and why the material supports, contradicts, or leaves a
Hypothesis inconclusive. The first scenario has one relevant incident document,
two plausible distractors, and one access-restricted document to prove ranking
and authorization.

### Identity and authorization

The synthetic system starts with five Agent Users, each assigned an Access
Profile: Data Analyst, Regional Manager, Jira Product Manager, Confluence
Product Manager, and Customer Success Manager. Profiles control row, column,
document, graph, and direct-identifier entitlements. The first vertical tests
the Data Analyst and APAC Regional Manager profiles.

Authorization is enforced before SQL execution, document retrieval, and graph
traversal. Output redaction and deterministic guardrails provide a second
layer; they do not substitute for pre-retrieval authorization. Direct
identifiers may be returned only where the requesting profile has the explicit
entitlement, classification permits it, the result is bounded, and the action
is audited.

### Provisional metrics and data-team workflow

If a requested metric is absent from the validated semantic layer, the service
marks a Metric Definition Gap. It may produce a Provisional Metric only when a
safe calculation can be made from permitted inputs. The response must name its
formula, inputs, scope, unverified status, and material caveats, then offer a
verification request to the data team. Creating that request requires explicit
Agent User confirmation.

### Causal analysis

The POC supports causal eligibility assessment and safe language, not general
causal automation. All-user pre/post comparisons are descriptive. A registered
randomized experiment may eventually run a pre-approved estimator after its
support and diagnostics pass. Observational approaches (matching, inverse
propensity weighting, DML, or related methods) first create a reviewable
analysis plan; they do not automatically yield a Causal Estimate.

The final result vocabulary is: evidence supports the Hypothesis, evidence
does not support it, or evidence is inconclusive. A Causal Estimate is allowed
only when the eligible design, diagnostics, and review are recorded.

### Models, guardrails, and observability

Generation uses a locally served small baseline model, initially `qwen3:8b`,
through Ollama. Embeddings use `embeddinggemma`. The model is not a source of
metric truth and may not bypass gateway, retrieval, or output policy checks.

Deterministic intent, tool, and output policies are in the first vertical.
NeMo Guardrails is deferred until measured POC failure modes justify its
operational cost. MLflow records observability and evaluation traces, including
route, policy fingerprint, source versions, tool outcomes, retrieval scores,
and redacted failure detail. Raw unauthorized identifiers are excluded.

## Synthetic data requirements

Create a reproducible synthetic dataset with approximately 1,000 Tenants,
10,000 Persons, and 15,000–20,000 Product Users over Jira and Confluence. It
contains eighteen months of immutable Paid Enablement and Visit events, plus
Tenant region, paid-subscription start, and seat tier. It must preserve the
domain rule that the same Person can be a separate Product User per product
and Tenant.

Seed, in addition to the first Jira New PEU incident, three future scenarios:

1. Confluence New Monthly Active User (New MAU) decline in EMEA enterprise
   Tenants after an onboarding-email regression.
2. Jira New MAU lift in Americas small Tenants in a registered onboarding
   treatment/control experiment.
3. Confluence New PEU lift in Americas 11–50 Seat Tier Tenants after a
   targeted acquisition campaign.

New MAU is reserved for subsequent delivery: it is a New PEU who has at least
one Visit in the same calendar month and product as first paid enablement. A
visit in another product does not qualify it.

## Response contract

The service returns a typed governed analytical response with these fields:

- answer text and result classification: canonical definition, Driver
  Decomposition, Hypothesis, Provisional Metric, limitation, or safe refusal;
- resolved metric definition and semantic version when applicable;
- effective access scope and source freshness;
- structured calculation or decomposition results when applicable;
- evidence citations with support status and scope;
- caveats, uncertainty, and any next action requiring confirmation;
- trace identifier suitable for audited MLflow lookup.

The answer text must distinguish observed data, retrieved evidence, inference,
and causal estimate. No final answer may use "root cause" unless it says that
the result is not established as causal.

## Implementation decisions

- Build a thin, long-running FastAPI service around the primary response seam.
- Use LangGraph to coordinate the lead and bounded specialists.
- Use LlamaIndex for retrieval orchestration and Qdrant as the initial vector
  store; retain a vector-store interface so an implementation can later use
  pgvector without changing the domain workflow.
- Use Postgres as analytical serving storage. dbt owns transformations, tests,
  models, and MetricFlow semantic definitions.
- Publish validated dbt metadata to the locally deployed DataHub instance for
  catalog, classification, ownership, and discoverability.
- Use Apache AGE only as the derived evidence graph.
- Use MLflow as the trace, observability, and evaluation system of record.
- Keep all services private-network only and use service identities for
  Postgres, Qdrant, DataHub, and MLflow access.
- Add interfaces around semantic querying, policy resolution, structured data,
  evidence retrieval, graph lookup, tracing, and model invocation so they can
  be tested independently.

## Testing and evaluation

Test primarily through the `answer_question` response seam. Tests should use
fixed synthetic fixtures and assert observable response properties rather than
internal LangGraph steps.

Required automated checks:

1. Jira New PEU definition matches the validated semantic artifact and includes
   version, grain, freshness, and source attribution.
2. The May-to-June driver answer reconciles from 4,000 to 3,440 and reports
   APAC / 51–200 Seat Tier Tenants as 420 of the 560 decline.
3. The evidence answer retrieves the relevant permitted incident ahead of the
   distractors and labels the conclusion as a Hypothesis rather than a cause.
4. The APAC Regional Manager cannot obtain non-APAC rows, documents, graph
   paths, citations, or inference through any broad or indirect wording.
5. A profile without direct-identifier entitlement cannot obtain identifiers
   through structured results, retrieved chunks, graph nodes, citations, or
   generated prose.
6. An entitled Customer Success Manager can receive only permitted, bounded,
   audited identifiers.
7. Failed or stale dbt semantic validation blocks a canonical semantic answer.
8. Missing semantic definitions follow the Provisional Metric / Metric
   Definition Gap contract and never claim canonical status.
9. Off-topic and unsupported-causal requests receive safe redirects or
   limitations.
10. MLflow traces contain required observability fields and no raw unauthorized
    identifiers.

Start evaluation with deterministic fixture checks and human-labelled expected
responses. Measure retrieval separately before judging answer generation:
Ragas can assess retrieval recall/precision or ranking and grounding;
DeepEval is reserved for agent/tool-trajectory regression; Promptfoo is added
for adversarial authorization and prompt-injection testing as the scope grows.
Use error analysis to add targeted fixtures iteratively. A local model upgrade
must not regress semantic provenance, arithmetic, entitlement safety,
retrieval quality, or evidence wording.

## Out of scope for the first delivery

- Revenue, Activation Rate, and any metric other than Jira New PEU.
- A production Teamwork Graph integration.
- pgvector as the POC retrieval store.
- General autonomous causal inference or automatic use of DML, matching, or
  inverse-propensity methods.
- A general conversational assistant, write actions, or autonomous data-team
  tickets.
- Real production data or public network exposure.
- A dependency on DataHub availability for canonical metric computation.
- NeMo Guardrails unless evaluation shows a concrete need.

## Delivery sequence

1. Establish dbt models, tests, MetricFlow definition, and synthetic Postgres
   data for Jira New PEU.
2. Implement entitlement resolution and the constrained semantic-query path.
3. Expose the canonical-definition response through FastAPI and test it at the
   response seam.
4. Add Driver Decomposition and the scripted May-to-June scenario.
5. Add Qdrant evidence, Apache AGE derived links, and the evidence-backed
   Hypothesis response.
6. Add the APAC Regional Manager authorization tests, MLflow tracing, and the
   initial evaluation fixtures.
7. Expand to Confluence and New MAU only after the first vertical is reliable.

## Open follow-ups

- Agree the exact SQL/RLS enforcement mechanism for Access Profiles before
  connecting any non-synthetic data.
- Define the data-team request endpoint and approval audit trail before
  enabling Metric Definition Gap submission.
- Set evaluation thresholds after observing the initial fixture and human
  review results; zero unauthorized disclosure and semantic provenance are
  non-negotiable.
- Define causal eligibility forms, estimator approval, and reviewer roles
  before enabling a Causal Estimate.

# Growth Data Agent

An evidence-first analytical assistant that explains governed business metrics,
their drivers, and evidence-backed hypotheses without presenting unverified
claims as facts.

## Analysis Language

**Person**:
One platform identity that may have Product User relationships in multiple
Tenants and products.
_Avoid_: User

**Product User**:
One person's access relationship to a named product within a Tenant. The
same person is a separate Product User in each product they can access.
_Avoid_: Platform user, unique person

**Tenant**:
The workspace that owns a paid subscription and contains Product Users.
_Avoid_: Customer, account, workspace

**New Paid Enabled User**:
A Product User granted paid access to that product for the first time ever.
Regaining access does not make the Product User new again.
_Avoid_: Reactivated user, newly active user

**New Monthly Active User**:
A New Paid Enabled User who visits that product at least once in the same
calendar month as first paid enablement. A visit to one product does not
activate another product.
_Avoid_: New user, active user

**Paid Enablement**:
An immutable event recording when a Product User receives paid access to a
product.
_Avoid_: Current access, entitlement snapshot

**Visit**:
An event recording that a Product User visited a product at a particular time.
_Avoid_: Login, engagement

**Canonical Metric**:
A metric declared in the validated semantic authority and available for governed
querying.
_Avoid_: Metric, official number

**Provisional Metric**:
An explicitly unverified calculation from permitted inputs, returned with its
formula, scope, and caveats.
_Avoid_: Canonical metric, source of truth

**Metric Definition Gap**:
A request for a metric that is absent from the semantic authority and needs
data-team verification.
_Avoid_: Broken metric, missing field

**Driver Decomposition**:
A quantified explanation of a metric movement by approved dimensions such as
region, customer segment, or plan.
_Avoid_: Root cause, causal explanation

**Hypothesis**:
An evidence-backed possible explanation for a driver that has not been shown to
be causal.
_Avoid_: Root cause, finding

**Evidence Chain**:
The cited path from a metric movement through an affected segment to related
documents, incidents, campaigns, contracts, or accountable teams.
_Avoid_: Proof, causal chain

**Candidate Causal Factor**:
A cited, falsifiable Hypothesis about a specific documented change that may
help explain a known Driver Decomposition movement; it is not proof of
causation.
_Avoid_: Cause, root cause, confirmed factor

**Factor Occurrence Time**:
The date a Candidate Causal Factor's documented change is recorded as having
occurred, used to test whether it could plausibly explain an observed
movement.
_Avoid_: Timestamp, event time

**Provisional Factor Record**:
A revision-linked derived projection extracted from one Evidence Revision,
never a source or permission authority; deterministic validation decides
whether it is ranking eligible.
_Avoid_: Verified factor, confirmed record

**Factor Vocabulary**:
The small, reviewed set of categories (such as provisioning or entitlement,
campaign, or incident) used to guide evidence retrieval for a Candidate
Causal Factor; it is not causal or metric authority.
_Avoid_: Root-cause taxonomy, category authority

**Ranking Signal**:
One of five named, typed, and bounded inputs (temporal alignment, population
overlap, metric-mechanism fit, independent source coverage, counterevidence)
used to order and cap Candidate Causal Factor cards; never collapsed into a
single opaque confidence score.
_Avoid_: Confidence score, relevance score

**Factor Support Status**:
A candidate card's deterministic outcome — supported, contradicted, or
inconclusive — based on independent supporting Evidence Revision count, high
authority operational records, and material contradiction; distinct from a
single citation's Evidence Support Status.
_Avoid_: Verified, confirmed, causal confidence

**High-Authority Operational Record**:
An Evidence Revision explicitly marked as a system-of-record operational
document, sufficient on its own to reach supported status without a second
independent source.
_Avoid_: Trusted source, verified document

**Paid Tenant Tenure**:
Elapsed time since the Product User's Tenant began its paid subscription.
_Avoid_: Account age, user age, tenure

**Seat Tier**:
The purchased-seat band of a Product User's Tenant: 1–10, 11–50, 51–200,
or 201+ seats.
_Avoid_: Plan tier, customer size

**Region**:
The billing or home region recorded on a Tenant.
_Avoid_: User locale, inferred location

## Governance Language

**Agent User**:
An authenticated person who asks the Growth Data Agent questions and receives
answers within an assigned Access Profile.
_Avoid_: User, Product User

**Access Profile**:
A reusable set of an Agent User's row, column, document, and identifier
entitlements.
_Avoid_: Role, permission set

**Entitlement**:
The user-specific permission to access defined rows, columns, documents, or
direct identifiers.
_Avoid_: Role, access level

**Sensitive Identifier**:
A direct identifier that may be returned only to an entitled user through a
bounded, audited response.
_Avoid_: PII, personal data

**Active Investigation**:
A stored reference to one previously selected Candidate Causal Factor's
`factor_id`, carried forward only as a lookup key; every subsequent turn
reauthorizes the Agent User and revalidates it against freshly ranked,
currently authorized Candidate Causal Factors before it is ever presented
again.
_Avoid_: Saved factor, cached hypothesis, remembered selection

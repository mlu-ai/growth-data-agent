# Derived evidence graph

Apache AGE is a derived investigation index over approved dbt metadata,
catalog metadata, and document-ingestion metadata. It supports ownership and
multi-hop evidence chains after driver analysis, while relational metadata
remains authoritative and the graph does not enforce permissions.

The application role is deliberately non-superuser. For a dedicated AGE
database, an administrator must preload the AGE library for each session and
grant the application role usage on `ag_catalog`; for example, configure
`session_preload_libraries = 'age'` at the database level and grant `USAGE ON
SCHEMA ag_catalog`. The application then sets `APACHE_AGE_PRELOADED=true`, so
its query and materialization sessions do not issue the privileged `LOAD
'age'` statement. If AGE is not preloaded or available to the role, the
adapter fails with an actionable configuration error rather than elevating
the role or bypassing graph authorization. The adapter uses AGE graph
functions for graph creation and Cypher access; it does not require direct
`SELECT` access to `ag_catalog.ag_graph`.

# dbt is the semantic authority

dbt and MetricFlow own canonical metric logic, dimensions, grain, and tests;
Postgres is the initial analytical serving store. The query gateway uses the
last validated dbt artifact, while DataHub receives published metadata for
catalog, ownership, classification, and discovery rather than becoming a
second source of metric logic.

from pathlib import Path


def test_dbt_model_selects_only_first_jira_paid_enablement() -> None:
    repository = Path(__file__).resolve().parents[1]
    intermediate_path = repository / "dbt/models/intermediate/int_first_paid_enablement.sql"
    intermediate = intermediate_path.read_text()
    fact_model = (repository / "dbt/models/marts/fct_jira_new_peu.sql").read_text()
    semantic_yaml = (repository / "dbt/models/marts/jira_new_peu.yml").read_text()
    confluence_fact_model = (
        repository / "dbt/models/marts/fct_confluence_new_peu.sql"
    ).read_text()
    confluence_semantic_yaml = (
        repository / "dbt/models/marts/confluence_new_peu.yml"
    ).read_text()
    time_spine_yaml = (repository / "dbt/models/marts/metricflow_time_spine.yml").read_text()

    assert "partition by product_user_id, product" in intermediate
    assert "row_number()" in intermediate
    assert "product = 'Jira'" in fact_model
    assert "paid_enablement_ordinal = 1" in fact_model
    assert "product = 'Confluence'" in confluence_fact_model
    assert "paid_enablement_ordinal = 1" in confluence_fact_model
    assert "count_distinct" in semantic_yaml
    assert "confluence_new_peu" in confluence_semantic_yaml
    assert "count_distinct" in confluence_semantic_yaml
    assert "seat_tier" in semantic_yaml
    assert "paid_tenant_tenure_days" in semantic_yaml
    tenant_staging = (repository / "dbt/models/staging/stg_tenants.sql").read_text()
    assert "paid_subscription_started_at" in tenant_staging
    assert "standard_granularity_column: date_day" in time_spine_yaml
    loader = (repository / "scripts/load_postgres.py").read_text()
    assert 'if __name__ == "__main__":' in loader
    assert "POSTGRES_PORT" in loader
    assert "paid_enablements_are_immutable" in loader
    assert "Refusing to reload immutable events" in loader


def test_dbt_eligible_population_excludes_already_paid_enabled_users() -> None:
    repository = Path(__file__).resolve().parents[1]
    intermediate = (
        repository / "dbt/models/intermediate/int_eligible_population.sql"
    ).read_text()
    jira_fact_model = (
        repository / "dbt/models/marts/fct_jira_new_peu_eligible_population.sql"
    ).read_text()
    confluence_fact_model = (
        repository / "dbt/models/marts/fct_confluence_new_peu_eligible_population.sql"
    ).read_text()
    jira_semantic_yaml = (
        repository / "dbt/models/marts/jira_new_peu_eligible_population.yml"
    ).read_text()
    confluence_semantic_yaml = (
        repository / "dbt/models/marts/confluence_new_peu_eligible_population.yml"
    ).read_text()
    staging = (
        repository / "dbt/models/staging/stg_product_user_entitlements.sql"
    ).read_text()

    assert "stg_product_user_entitlements" in intermediate
    assert "int_first_paid_enablement" in intermediate
    assert "paid_enablement_ordinal = 1" in intermediate
    assert "enablements.product_user_id is null" in intermediate
    assert "product = 'Jira'" in jira_fact_model
    assert "product = 'Confluence'" in confluence_fact_model
    assert "count_distinct" in jira_semantic_yaml
    assert "count_distinct" in confluence_semantic_yaml
    assert "seat_tier" in jira_semantic_yaml
    assert "seat_tier" in confluence_semantic_yaml
    assert "name: jira_new_peu_eligible_population_product_user" in jira_semantic_yaml
    assert "name: confluence_new_peu_eligible_population_product_user" in confluence_semantic_yaml
    assert "product_user_entitlements" in staging

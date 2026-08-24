from pathlib import Path


def test_dbt_model_selects_only_first_jira_paid_enablement() -> None:
    repository = Path(__file__).resolve().parents[1]
    intermediate_path = repository / "dbt/models/intermediate/int_first_paid_enablement.sql"
    intermediate = intermediate_path.read_text()
    fact_model = (repository / "dbt/models/marts/fct_jira_new_peu.sql").read_text()
    semantic_yaml = (repository / "dbt/models/marts/jira_new_peu.yml").read_text()
    time_spine_yaml = (repository / "dbt/models/marts/metricflow_time_spine.yml").read_text()

    assert "partition by product_user_id, product" in intermediate
    assert "row_number()" in intermediate
    assert "product = 'Jira'" in fact_model
    assert "paid_enablement_ordinal = 1" in fact_model
    assert "count_distinct" in semantic_yaml
    assert "standard_granularity_column: date_day" in time_spine_yaml
    loader = (repository / "scripts/load_postgres.py").read_text()
    assert 'if __name__ == "__main__":' in loader
    assert "POSTGRES_PORT" in loader
    assert "paid_enablements_are_immutable" in loader
    assert "Refusing to reload immutable events" in loader

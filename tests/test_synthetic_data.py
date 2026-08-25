import csv

from growth_data_agent.synthetic import generate


def test_synthetic_dataset_is_reproducible_and_glossary_aligned(tmp_path) -> None:
    first_counts = generate(tmp_path / "first")
    second_counts = generate(tmp_path / "second")

    assert first_counts == second_counts
    assert first_counts.tenants == 1_000
    assert first_counts.persons == 10_000
    assert 15_000 <= first_counts.product_users <= 20_000
    assert first_counts.paid_enablements > first_counts.product_users
    assert (tmp_path / "first" / "paid_enablements.csv").read_bytes() == (
        tmp_path / "second" / "paid_enablements.csv"
    ).read_bytes()

    with (tmp_path / "first" / "product_users.csv").open() as handle:
        product_users = list(csv.DictReader(handle))
    pairs = {(row["person_id"], row["tenant_id"]) for row in product_users}
    assert any(
        {row["product"] for row in product_users if (row["person_id"], row["tenant_id"]) == pair}
        == {"Jira", "Confluence"}
        for pair in pairs
    )

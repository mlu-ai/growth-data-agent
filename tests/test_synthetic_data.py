import csv

from growth_data_agent.synthetic import evidence_corpus, generate


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

    tenants_by_id = {}
    with (tmp_path / "first" / "tenants.csv").open() as handle:
        for tenant in csv.DictReader(handle):
            tenants_by_id[tenant["tenant_id"]] = (
                tenant["billing_region"],
                tenant["seat_tier"],
            )
    product_users_by_id = {row["product_user_id"]: row for row in product_users}
    first_jira_enablements = {}
    with (tmp_path / "first" / "paid_enablements.csv").open() as handle:
        for event in csv.DictReader(handle):
            product_user = product_users_by_id[event["product_user_id"]]
            if product_user["product"] != "Jira":
                continue
            product_user_id = event["product_user_id"]
            if event["paid_enabled_at"] < first_jira_enablements.get(product_user_id, "~"):
                first_jira_enablements[product_user_id] = event["paid_enabled_at"]

    may_june_counts = {}
    for product_user_id, paid_enabled_at in first_jira_enablements.items():
        month = paid_enabled_at[:7]
        if month not in {"2026-05", "2026-06"}:
            continue
        product_user = product_users_by_id[product_user_id]
        segment = tenants_by_id[product_user["tenant_id"]]
        key = (month, *segment)
        may_june_counts[key] = may_june_counts.get(key, 0) + 1

    may_total = sum(count for (month, *_), count in may_june_counts.items() if month == "2026-05")
    june_total = sum(count for (month, *_), count in may_june_counts.items() if month == "2026-06")
    assert may_total == 4000
    assert june_total == 3440
    assert may_june_counts[("2026-05", "APAC", "51-200")] == 800
    assert may_june_counts[("2026-06", "APAC", "51-200")] == 380


def test_synthetic_evidence_corpus_has_incident_distractors_and_restricted_case() -> None:
    documents = evidence_corpus()

    assert [document.document_id for document in documents] == [
        "jira-apac-paid-provisioning-incident",
        "jira-apac-small-tenant-maintenance",
        "jira-apac-tenant-migration-notice",
        "jira-apac-paid-provisioning-incident-restricted",
    ]
    assert documents[0].support_status.value == "supports"
    assert documents[0].tenant_scope == "APAC 51-200 Seat Tier Tenants"
    assert documents[1].support_status.value == "inconclusive"
    assert documents[2].support_status.value == "inconclusive"
    assert documents[3].classification == "restricted"
    assert documents[3].identifier_entitlement == "direct"

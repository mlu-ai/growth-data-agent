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


def test_synthetic_confluence_campaign_movement_is_reproducible_and_reconciled(
    tmp_path,
) -> None:
    generate(tmp_path / "campaign")
    with (tmp_path / "campaign" / "tenants.csv").open() as handle:
        tenants = {
            row["tenant_id"]: (row["billing_region"], row["seat_tier"])
            for row in csv.DictReader(handle)
        }
    with (tmp_path / "campaign" / "product_users.csv").open() as handle:
        product_users = {
            row["product_user_id"]: row for row in csv.DictReader(handle)
        }

    first_confluence_enablements = {}
    with (tmp_path / "campaign" / "paid_enablements.csv").open() as handle:
        for event in csv.DictReader(handle):
            product_user = product_users[event["product_user_id"]]
            if product_user["product"] != "Confluence":
                continue
            current = first_confluence_enablements.get(event["product_user_id"])
            if current is None or event["paid_enabled_at"] < current:
                first_confluence_enablements[event["product_user_id"]] = event[
                    "paid_enabled_at"
                ]

    counts = {}
    for product_user_id, enabled_at in first_confluence_enablements.items():
        if enabled_at[:7] not in {"2026-05", "2026-06"}:
            continue
        product_user = product_users[product_user_id]
        region, seat_tier = tenants[product_user["tenant_id"]]
        key = (enabled_at[:7], region, seat_tier)
        counts[key] = counts.get(key, 0) + 1

    assert sum(value for (month, _, _), value in counts.items() if month == "2026-05") == 2400
    assert sum(value for (month, _, _), value in counts.items() if month == "2026-06") == 2820
    assert counts[("2026-05", "Americas", "11-50")] == 1200
    assert counts[("2026-06", "Americas", "11-50")] == 1620


def test_synthetic_confluence_new_mau_emea_regression_uses_same_product_and_month(
    tmp_path,
) -> None:
    generate(tmp_path / "new-mau")
    with (tmp_path / "new-mau" / "tenants.csv").open() as handle:
        tenants = {
            row["tenant_id"]: (row["billing_region"], row["seat_tier"])
            for row in csv.DictReader(handle)
        }
    with (tmp_path / "new-mau" / "product_users.csv").open() as handle:
        product_users = {row["product_user_id"]: row for row in csv.DictReader(handle)}
    first_enablements = {}
    with (tmp_path / "new-mau" / "paid_enablements.csv").open() as handle:
        for event in csv.DictReader(handle):
            product_user = product_users[event["product_user_id"]]
            if product_user["product"] != "Confluence":
                continue
            current = first_enablements.get(event["product_user_id"])
            if current is None or event["paid_enabled_at"] < current:
                first_enablements[event["product_user_id"]] = event["paid_enabled_at"]

    qualifying_months = {}
    with (tmp_path / "new-mau" / "visits.csv").open() as handle:
        for visit in csv.DictReader(handle):
            product_user = product_users[visit["product_user_id"]]
            first_enabled_at = first_enablements.get(visit["product_user_id"])
            if (
                product_user["product"] == "Confluence"
                and tenants[product_user["tenant_id"]] == ("EMEA", "51-200")
                and visit["product"] == product_user["product"]
                and first_enabled_at is not None
                and first_enabled_at[:7] in {"2026-05", "2026-06"}
                and visit["visited_at"][:7] == first_enabled_at[:7]
            ):
                qualifying_months.setdefault(first_enabled_at[:7], set()).add(
                    visit["product_user_id"]
                )

    assert {month: len(users) for month, users in qualifying_months.items()} == {
        "2026-05": 600,
        "2026-06": 300,
    }


def test_synthetic_evidence_corpus_has_incident_distractors_and_restricted_case() -> None:
    documents = evidence_corpus()

    assert [document.document_id for document in documents] == [
        "jira-apac-paid-provisioning-incident",
        "jira-apac-small-tenant-maintenance",
        "jira-apac-tenant-migration-notice",
        "jira-apac-paid-provisioning-incident-restricted",
        "confluence-americas-acquisition-campaign",
        "confluence-americas-enterprise-campaign",
        "confluence-americas-provisioning-maintenance",
        "confluence-americas-acquisition-campaign-restricted",
        "confluence-emea-onboarding-email-regression",
        "confluence-emea-small-tenant-onboarding-email",
        "confluence-emea-201-plus-onboarding-email",
        "confluence-emea-onboarding-email-regression-restricted",
    ]
    assert documents[0].support_status.value == "supports"
    assert documents[0].tenant_scope == "APAC 51-200 Seat Tier Tenants"
    assert documents[1].support_status.value == "inconclusive"
    assert documents[2].support_status.value == "inconclusive"
    assert documents[3].classification == "restricted"
    assert documents[3].identifier_entitlement == "direct"
    assert documents[4].support_status.value == "supports"
    assert documents[4].tenant_scope == "Americas 11-50 Seat Tier Tenants"
    assert documents[7].classification == "restricted"
    assert documents[7].identifier_entitlement == "direct"

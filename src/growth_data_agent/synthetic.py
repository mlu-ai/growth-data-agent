"""Deterministic synthetic data for the local Postgres analytical store."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_START_DATE = date(2025, 1, 1)
_END_DATE = date(2026, 6, 30)
_REGIONS = ("Americas", "APAC", "EMEA")
_SEAT_TIERS = ("1-10", "11-50", "51-200", "201+")


@dataclass(frozen=True)
class DatasetCounts:
    tenants: int
    persons: int
    product_users: int
    paid_enablements: int
    visits: int


def generate(output_directory: Path) -> DatasetCounts:
    """Write a reproducible, glossary-aligned set of CSV source tables."""
    output_directory.mkdir(parents=True, exist_ok=True)
    tenants = _tenants()
    persons = [{"person_id": f"person-{number:05d}"} for number in range(1, 10_001)]
    product_users = _product_users()
    paid_enablements = _paid_enablements(product_users)
    visits = _visits(product_users)

    _write_csv(output_directory / "tenants.csv", tenants)
    _write_csv(output_directory / "persons.csv", persons)
    _write_csv(output_directory / "product_users.csv", product_users)
    _write_csv(output_directory / "paid_enablements.csv", paid_enablements)
    _write_csv(output_directory / "visits.csv", visits)
    return DatasetCounts(
        tenants=len(tenants),
        persons=len(persons),
        product_users=len(product_users),
        paid_enablements=len(paid_enablements),
        visits=len(visits),
    )


def _tenants() -> list[dict[str, str]]:
    return [
        {
            "tenant_id": f"tenant-{number:04d}",
            "billing_region": _REGIONS[(number - 1) % len(_REGIONS)],
            "paid_subscription_started_at": (
                _START_DATE - timedelta(days=30 * ((number - 1) % 18))
            ).isoformat(),
            "seat_tier": _SEAT_TIERS[(number - 1) % len(_SEAT_TIERS)],
        }
        for number in range(1, 1_001)
    ]


def _product_users() -> list[dict[str, str]]:
    product_users: list[dict[str, str]] = []
    sequence = 1
    # Each of these Persons has separate Product User relationships in both products.
    for person_number in range(1, 6_001):
        tenant_id = f"tenant-{((person_number * 17 - 1) % 1_000) + 1:04d}"
        for product in ("Jira", "Confluence"):
            product_users.append(
                {
                    "product_user_id": f"product-user-{sequence:05d}",
                    "person_id": f"person-{person_number:05d}",
                    "tenant_id": tenant_id,
                    "product": product,
                }
            )
            sequence += 1
    for person_number in range(6_001, 10_001):
        product_users.append(
            {
                "product_user_id": f"product-user-{sequence:05d}",
                "person_id": f"person-{person_number:05d}",
                "tenant_id": f"tenant-{((person_number * 29 - 1) % 1_000) + 1:04d}",
                "product": "Jira" if person_number % 2 else "Confluence",
            }
        )
        sequence += 1
    return product_users


def _paid_enablements(product_users: list[dict[str, str]]) -> list[dict[str, str]]:
    event_rows: list[dict[str, str]] = []
    days_in_period = (_END_DATE - _START_DATE).days + 1
    sequence = 1
    for index, product_user in enumerate(product_users, start=1):
        first_enabled = _START_DATE + timedelta(days=(index * 37) % days_in_period)
        event_rows.append(_enablement_event(sequence, product_user, first_enabled))
        sequence += 1
        # A later immutable Paid Enablement event proves that restoration does not requalify.
        restoration = first_enabled + timedelta(days=70)
        if index % 7 == 0 and restoration <= _END_DATE:
            event_rows.append(_enablement_event(sequence, product_user, restoration))
            sequence += 1
    return event_rows


def _enablement_event(
    sequence: int, product_user: dict[str, str], enabled_on: date
) -> dict[str, str]:
    return {
        "paid_enablement_id": f"paid-enablement-{sequence:05d}",
        "product_user_id": product_user["product_user_id"],
        "tenant_id": product_user["tenant_id"],
        "product": product_user["product"],
        "paid_enabled_at": datetime.combine(enabled_on, datetime.min.time(), UTC).isoformat(),
    }


def _visits(product_users: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "visit_id": f"visit-{index:05d}",
            "product_user_id": product_user["product_user_id"],
            "product": product_user["product"],
            "visited_at": datetime.combine(
                _START_DATE + timedelta(days=(index * 41) % ((_END_DATE - _START_DATE).days + 1)),
                datetime.min.time(),
                UTC,
            ).isoformat(),
        }
        for index, product_user in enumerate(product_users, start=1)
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

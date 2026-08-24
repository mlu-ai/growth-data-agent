"""Generate reproducible synthetic source data for local Postgres."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from growth_data_agent.synthetic import generate

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    counts = generate(arguments.output)
    print(
        "Generated "
        f"{counts.tenants} Tenants, {counts.persons} Persons, "
        f"{counts.product_users} Product Users, {counts.paid_enablements} Paid Enablements, "
        f"and {counts.visits} Visits in {arguments.output}."
    )


if __name__ == "__main__":
    main()

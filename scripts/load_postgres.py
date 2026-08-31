"""Load generated synthetic CSVs into the local Postgres source schema."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

_TABLES = {
    "tenants": "tenant_id, billing_region, paid_subscription_started_at, seat_tier",
    "persons": "person_id",
    "product_users": "product_user_id, person_id, tenant_id, product",
    "paid_enablements": "paid_enablement_id, product_user_id, tenant_id, product, paid_enabled_at",
    "visits": "visit_id, product_user_id, product, visited_at",
    "product_user_entitlements": "entitlement_id, product_user_id, tenant_id, product, entitled_at",
}

_DDL = """
create table if not exists public.tenants (
    tenant_id text primary key, billing_region text not null,
    paid_subscription_started_at date not null, seat_tier text not null
);
create table if not exists public.persons (person_id text primary key);
create table if not exists public.product_users (
    product_user_id text primary key, person_id text not null references public.persons,
    tenant_id text not null references public.tenants, product text not null
);
create table if not exists public.paid_enablements (
    paid_enablement_id text primary key,
    product_user_id text not null references public.product_users,
    tenant_id text not null references public.tenants, product text not null,
    paid_enabled_at timestamptz not null
);
create table if not exists public.visits (
    visit_id text primary key, product_user_id text not null references public.product_users,
    product text not null, visited_at timestamptz not null
);
create table if not exists public.product_user_entitlements (
    entitlement_id text primary key, product_user_id text not null,
    tenant_id text not null references public.tenants, product text not null,
    entitled_at timestamptz not null
);
create or replace function public.reject_paid_enablement_mutation()
returns trigger
language plpgsql
as $$
begin
    raise exception 'Paid Enablement events are immutable';
end;
$$;
drop trigger if exists paid_enablements_are_immutable on public.paid_enablements;
create trigger paid_enablements_are_immutable
before update or delete on public.paid_enablements
for each row execute function public.reject_paid_enablement_mutation();
"""


def main() -> None:
    data_directory = Path(os.environ.get("SYNTHETIC_DATA_DIRECTORY", "data"))
    missing = [name for name in _TABLES if not (data_directory / f"{name}.csv").exists()]
    if missing:
        raise SystemExit(
            f"Missing generated CSVs for: {', '.join(missing)}. Run make generate-data."
        )

    database_url = os.environ.get("DATABASE_URL") or (
        "postgresql://"
        f"{os.environ.get('POSTGRES_USER', 'growth_data')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'growth_data')}@"
        f"{os.environ.get('POSTGRES_HOST', '127.0.0.1')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'growth_data')}"
    )
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(_DDL)
            cursor.execute("select exists (select 1 from public.paid_enablements)")
            if cursor.fetchone()[0]:
                raise SystemExit(
                    "Paid Enablement events already exist. Refusing to reload immutable events; "
                    "use a fresh synthetic database."
                )
            for table in reversed(tuple(_TABLES)):
                cursor.execute(f"truncate table public.{table} cascade")
            for table, columns in _TABLES.items():
                with (data_directory / f"{table}.csv").open() as source:
                    copy_sql = f"copy public.{table} ({columns}) from stdin with csv header"
                    with cursor.copy(copy_sql) as copy:
                        for line in source:
                            copy.write(line)
    print("Loaded reproducible synthetic data into Postgres.")


if __name__ == "__main__":
    main()

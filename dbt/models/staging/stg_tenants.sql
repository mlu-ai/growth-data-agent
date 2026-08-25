select
    tenant_id,
    billing_region,
    paid_subscription_started_at,
    seat_tier
from {{ source('synthetic', 'tenants') }}

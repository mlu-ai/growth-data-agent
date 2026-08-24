select
    product_user_id,
    tenant_id,
    paid_enabled_at,
    date_trunc('day', paid_enabled_at)::date as paid_enablement_date
from {{ ref('int_first_paid_enablement') }}
where product = 'Jira'
  and paid_enablement_ordinal = 1

select
    enablements.product_user_id,
    enablements.tenant_id,
    'Confluence' as product,
    tenants.billing_region as region,
    tenants.seat_tier,
    (enablements.paid_enabled_at::date - tenants.paid_subscription_started_at) as paid_tenant_tenure_days,
    enablements.paid_enabled_at,
    date_trunc('day', enablements.paid_enabled_at)::date as paid_enablement_date
from {{ ref('int_first_paid_enablement') }} as enablements
inner join {{ ref('stg_tenants') }} as tenants using (tenant_id)
where enablements.product = 'Confluence'
  and enablements.paid_enablement_ordinal = 1

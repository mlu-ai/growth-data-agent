-- Eligible Population for New Paid Enabled User sizing: entitled Product Users
-- who have not previously qualified through Paid Enablement for that product.
select
    entitlements.entitlement_id,
    entitlements.product_user_id,
    entitlements.tenant_id,
    entitlements.product,
    tenants.billing_region as region,
    tenants.seat_tier,
    entitlements.entitled_at,
    date_trunc('day', entitlements.entitled_at)::date as entitled_date
from {{ ref('stg_product_user_entitlements') }} as entitlements
inner join {{ ref('stg_tenants') }} as tenants using (tenant_id)
left join {{ ref('int_first_paid_enablement') }} as enablements
    on enablements.product_user_id = entitlements.product_user_id
   and enablements.product = entitlements.product
   and enablements.paid_enablement_ordinal = 1
where enablements.product_user_id is null

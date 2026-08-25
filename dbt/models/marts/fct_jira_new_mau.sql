select distinct
    new_peu.product_user_id,
    new_peu.tenant_id,
    new_peu.product,
    new_peu.region,
    new_peu.seat_tier,
    new_peu.paid_tenant_tenure_days,
    new_peu.paid_enabled_at,
    date_trunc('day', new_peu.paid_enabled_at)::date as paid_enablement_date
from {{ ref('fct_jira_new_peu') }} as new_peu
inner join {{ ref('stg_visits') }} as visits
    on visits.product_user_id = new_peu.product_user_id
    and visits.product = new_peu.product
    and date_trunc('month', visits.visited_at) = date_trunc('month', new_peu.paid_enabled_at)
where visits.product = 'Jira'

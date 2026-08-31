select
    product_user_id,
    tenant_id,
    'Jira' as product,
    region,
    seat_tier,
    entitled_at,
    entitled_date
from {{ ref('int_eligible_population') }}
where product = 'Jira'

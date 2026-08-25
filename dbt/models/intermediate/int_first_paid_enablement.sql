select
    paid_enablement_id,
    product_user_id,
    tenant_id,
    product,
    paid_enabled_at,
    row_number() over (
        partition by product_user_id, product
        order by paid_enabled_at, paid_enablement_id
    ) as paid_enablement_ordinal
from {{ ref('stg_paid_enablements') }}

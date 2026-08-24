select
    paid_enablement_id,
    product_user_id,
    tenant_id,
    product,
    paid_enabled_at::timestamp as paid_enabled_at
from {{ source('synthetic', 'paid_enablements') }}

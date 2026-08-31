select
    entitlement_id,
    product_user_id,
    tenant_id,
    product,
    entitled_at::timestamp as entitled_at
from {{ source('synthetic', 'product_user_entitlements') }}

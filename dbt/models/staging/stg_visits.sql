select
    visit_id,
    product_user_id,
    product,
    visited_at::timestamp as visited_at
from {{ source('synthetic', 'visits') }}

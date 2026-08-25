select date_day::date as date_day
from generate_series(
    '2020-01-01'::date,
    '2030-12-31'::date,
    interval '1 day'
) as spine(date_day)

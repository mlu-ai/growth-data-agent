"""Small shared helpers for canonical metric identifiers."""

from __future__ import annotations

import re


def metric_identifier(value: str) -> str:
    """Normalize a user-supplied metric name to its bounded identifier form."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")

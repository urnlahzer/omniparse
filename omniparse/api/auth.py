"""API key authentication -- pure functions operating on a dict-like store.

The `store` parameter is a plain dict in tests and Modal Dict in production.
Duck-typed via __getitem__/__setitem__ so this module stays Modal-free.

Spend tracking uses time-bucketed keys to avoid read-modify-write race
conditions on a single spent_usd field.  Each call to record_spend writes
to an hourly bucket key (``spend:{api_key}:{YYYYMMDDHH}``), and
get_total_spend sums all buckets for a given key.

Functions:
- register_api_key: provision a new API key with budget
- validate_api_key: check key exists, is active, and within budget
- record_spend: write spend to an hourly bucket key
- get_total_spend: sum all hourly spend buckets for a key
"""

import time

SPEND_BUCKET_PREFIX = "spend:"


def _hour_bucket() -> str:
    """Return the current UTC hour as a 10-digit string ``YYYYMMDDHH``."""
    t = time.gmtime()
    return f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}{t.tm_hour:02d}"


def register_api_key(store: dict, api_key: str, budget_usd: float = 1.0) -> None:
    """Provision a new API key with the specified budget.

    Args:
        store: Key-value store (plain dict or Modal Dict).
        api_key: The API key string to register.
        budget_usd: Maximum spend allowed for this key (default $1.00).
    """
    store[api_key] = {"budget_usd": budget_usd, "active": True}


def get_total_spend(store: dict, api_key: str) -> float:
    """Sum all hourly spend buckets for *api_key*.

    Iterates store keys matching the prefix ``spend:{api_key}:`` and
    returns the cumulative spend.  Returns 0.0 if no buckets exist.

    Args:
        store: Key-value store containing spend bucket keys.
        api_key: The API key whose spend to total.

    Returns:
        Total spend in USD across all hourly buckets.
    """
    prefix = f"{SPEND_BUCKET_PREFIX}{api_key}:"
    total = 0.0
    for key in list(store.keys()):
        if key.startswith(prefix):
            total += store[key]
    return total


def validate_api_key(store: dict, api_key: str) -> dict:
    """Validate an API key against the store.

    Checks existence, active status, and remaining budget (via bucketed
    spend totals).

    Args:
        store: Key-value store containing registered keys.
        api_key: The API key to validate.

    Returns:
        The key's entry dict if valid.

    Raises:
        ValueError: If key is missing, deactivated, or budget exhausted.
    """
    try:
        entry = store[api_key]
    except KeyError:
        raise ValueError("Invalid API key")

    if not entry["active"]:
        raise ValueError("API key deactivated")

    if get_total_spend(store, api_key) >= entry["budget_usd"]:
        raise ValueError("Budget exhausted")

    return entry


def record_spend(store: dict, api_key: str, amount_usd: float) -> None:
    """Record spend in an hourly bucket key to avoid read-modify-write races.

    Writes to ``spend:{api_key}:{YYYYMMDDHH}`` so concurrent requests in
    the same hour accumulate into the same bucket.  The budget check in
    validate_api_key sums all buckets, so it always errs on the side of
    caution (over-counts, never allows overspend).

    Args:
        store: Key-value store containing registered keys.
        api_key: The API key to charge.
        amount_usd: Amount in USD to add to this hour's bucket.
    """
    bucket_key = f"{SPEND_BUCKET_PREFIX}{api_key}:{_hour_bucket()}"
    try:
        current = store[bucket_key]
    except KeyError:
        current = 0.0
    store[bucket_key] = current + amount_usd

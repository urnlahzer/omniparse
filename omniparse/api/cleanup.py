"""TTL-based cleanup of Modal Dict entries for jobs, HITL reviews, and spend buckets.

Implements automatic purging of expired entries from Modal Dict stores.
All functions are pure (no Modal dependency) -- they operate on any dict-like
store with __getitem__, __delitem__, and keys().

Constants:
- DEFAULT_TTL_DAYS: Default retention period (30 days), configurable via
  OMNIPARSE_DATA_TTL_DAYS env var at the scheduled function level.

Functions:
- purge_expired_entries: Delete job/HITL entries older than TTL from a store.
- purge_expired_spend_buckets: Delete spend bucket keys older than TTL.
"""

import calendar
import logging
import time

from omniparse.api.auth import SPEND_BUCKET_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_TTL_DAYS = 30


def purge_expired_entries(
    store: dict,
    ttl_days: int = DEFAULT_TTL_DAYS,
    *,
    now: float | None = None,
) -> int:
    """Delete job/HITL entries older than TTL from a dict-like store.

    Iterates all keys (snapshot via list() to allow deletion during iteration).
    Skips keys starting with SPEND_BUCKET_PREFIX (handled by
    purge_expired_spend_buckets). For dict entries: deletes if created_at is
    missing (per D-08) or older than cutoff.

    Args:
        store: Dict-like store (plain dict or Modal Dict).
        ttl_days: Number of days before an entry expires.
        now: Current time as epoch float (injectable for testing).

    Returns:
        Count of deleted entries.
    """
    if now is None:
        now = time.time()

    cutoff = now - (ttl_days * 86400)
    deleted = 0

    for key in list(store.keys()):
        # Skip spend bucket keys -- handled separately
        if key.startswith(SPEND_BUCKET_PREFIX):
            continue

        try:
            entry = store[key]
        except KeyError:
            continue

        if not isinstance(entry, dict):
            continue

        # Entries without created_at are treated as expired (D-08)
        created_at = entry.get("created_at")
        if created_at is None or created_at < cutoff:
            del store[key]
            deleted += 1

    logger.info("Purged %d expired entries (TTL=%d days)", deleted, ttl_days)
    return deleted


def purge_expired_spend_buckets(
    store: dict,
    ttl_days: int = DEFAULT_TTL_DAYS,
    *,
    now: float | None = None,
) -> int:
    """Delete spend bucket keys older than TTL from a dict-like store.

    Spend bucket keys have format ``spend:{api_key}:{YYYYMMDDHH}``.
    Parses the hour suffix to a timestamp and deletes buckets older than TTL.

    Args:
        store: Dict-like store containing spend bucket keys.
        ttl_days: Number of days before a bucket expires.
        now: Current time as epoch float (injectable for testing).

    Returns:
        Count of deleted spend buckets.
    """
    if now is None:
        now = time.time()

    cutoff = now - (ttl_days * 86400)
    deleted = 0

    for key in list(store.keys()):
        if not key.startswith(SPEND_BUCKET_PREFIX):
            continue

        # Extract YYYYMMDDHH suffix (last 10 chars after final ':')
        parts = key.rsplit(":", maxsplit=1)
        if len(parts) != 2 or len(parts[1]) != 10:
            continue

        hour_str = parts[1]
        try:
            t = time.strptime(hour_str, "%Y%m%d%H")
            bucket_ts = float(calendar.timegm(t))
        except (ValueError, OverflowError):
            continue

        if bucket_ts < cutoff:
            del store[key]
            deleted += 1

    logger.info("Purged %d expired spend buckets (TTL=%d days)", deleted, ttl_days)
    return deleted

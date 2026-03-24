"""Tests for TTL-based cleanup of Modal Dict entries.

Uses plain dicts as stores and fixed timestamps for deterministic testing.
Validates purge_expired_entries, purge_expired_spend_buckets, and DEFAULT_TTL_DAYS.
"""
import time

import pytest

from omniparse.api.auth import SPEND_BUCKET_PREFIX
from omniparse.api.cleanup import (
    DEFAULT_TTL_DAYS,
    purge_expired_entries,
    purge_expired_spend_buckets,
)


class TestDefaultTTL:
    """Verify TTL constant."""

    def test_default_ttl_days_equals_30(self):
        """DEFAULT_TTL_DAYS is 30."""
        assert DEFAULT_TTL_DAYS == 30


class TestPurgeExpiredEntries:
    """Verify TTL purge logic for job and HITL entries."""

    def test_deletes_entries_older_than_ttl(self):
        """Entries with created_at older than TTL are deleted."""
        now = 1_000_000_000.0
        old_ts = now - (31 * 86400)  # 31 days ago
        store = {
            "job1": {"created_at": old_ts, "status": "completed"},
            "job2": {"created_at": old_ts - 86400, "status": "completed"},
        }
        count = purge_expired_entries(store, ttl_days=30, now=now)
        assert count == 2
        assert len(store) == 0

    def test_keeps_entries_newer_than_ttl(self):
        """Entries with created_at within TTL are kept."""
        now = 1_000_000_000.0
        recent_ts = now - (29 * 86400)  # 29 days ago
        store = {
            "job1": {"created_at": recent_ts, "status": "completed"},
            "job2": {"created_at": now - 3600, "status": "processing"},
        }
        count = purge_expired_entries(store, ttl_days=30, now=now)
        assert count == 0
        assert len(store) == 2

    def test_entries_without_created_at_treated_as_expired(self):
        """Entries missing created_at key are treated as expired and deleted (D-08)."""
        now = 1_000_000_000.0
        store = {
            "job_old": {"status": "completed"},  # No created_at
            "job_new": {"created_at": now - 3600, "status": "processing"},
        }
        count = purge_expired_entries(store, ttl_days=30, now=now)
        assert count == 1
        assert "job_old" not in store
        assert "job_new" in store

    def test_returns_count_of_deleted_entries(self):
        """purge_expired_entries returns the number of deleted entries."""
        now = 1_000_000_000.0
        old_ts = now - (31 * 86400)
        store = {
            "job1": {"created_at": old_ts, "status": "completed"},
            "job2": {"created_at": now - 3600, "status": "processing"},
            "job3": {"created_at": old_ts, "status": "completed"},
        }
        count = purge_expired_entries(store, ttl_days=30, now=now)
        assert count == 2

    def test_empty_store_returns_zero(self):
        """purge_expired_entries with empty store returns 0."""
        store = {}
        count = purge_expired_entries(store, ttl_days=30, now=1_000_000_000.0)
        assert count == 0

    def test_skips_spend_bucket_keys(self):
        """Keys starting with SPEND_BUCKET_PREFIX are skipped by purge_expired_entries."""
        now = 1_000_000_000.0
        old_ts = now - (31 * 86400)
        store = {
            "job1": {"created_at": old_ts, "status": "completed"},
            f"{SPEND_BUCKET_PREFIX}key1:2026010100": 1.50,
        }
        count = purge_expired_entries(store, ttl_days=30, now=now)
        assert count == 1
        assert f"{SPEND_BUCKET_PREFIX}key1:2026010100" in store


class TestPurgeExpiredSpendBuckets:
    """Verify spend bucket cleanup logic."""

    def test_deletes_old_spend_buckets(self):
        """Spend bucket keys older than TTL are deleted."""
        # 2026-01-01 00:00 UTC as "now"
        now = 1_767_225_600.0  # 2026-01-01 00:00:00 UTC
        store = {
            # Old bucket: 2025-11-01 00:00 (>30 days before now)
            f"{SPEND_BUCKET_PREFIX}key1:2025110100": 1.50,
            # Recent bucket: 2025-12-25 00:00 (<30 days before now)
            f"{SPEND_BUCKET_PREFIX}key1:2025122500": 0.75,
            # Non-spend key (should be ignored)
            "api_key_1": {"budget_usd": 10.0, "active": True},
        }
        count = purge_expired_spend_buckets(store, ttl_days=30, now=now)
        assert count == 1
        assert f"{SPEND_BUCKET_PREFIX}key1:2025110100" not in store
        assert f"{SPEND_BUCKET_PREFIX}key1:2025122500" in store
        assert "api_key_1" in store

    def test_keeps_recent_spend_buckets(self):
        """Spend bucket keys within TTL are kept."""
        now = 1_767_225_600.0  # 2026-01-01 00:00:00 UTC
        store = {
            f"{SPEND_BUCKET_PREFIX}key1:2025122500": 0.75,
            f"{SPEND_BUCKET_PREFIX}key1:2025123100": 1.00,
        }
        count = purge_expired_spend_buckets(store, ttl_days=30, now=now)
        assert count == 0
        assert len(store) == 2

    def test_empty_store_returns_zero(self):
        """purge_expired_spend_buckets with empty store returns 0."""
        store = {}
        count = purge_expired_spend_buckets(store, ttl_days=30, now=1_000_000_000.0)
        assert count == 0

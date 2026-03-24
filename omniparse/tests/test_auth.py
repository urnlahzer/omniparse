"""Tests for API key authentication -- register, validate, bucketed spend tracking."""
import pytest

from omniparse.api.auth import (
    register_api_key,
    validate_api_key,
    record_spend,
    get_total_spend,
    SPEND_BUCKET_PREFIX,
)


# --- Registration and validation ---


def test_register_and_validate():
    """Register key, validate succeeds, returns entry with budget_usd and active=True."""
    store = {}
    register_api_key(store, "key-001", budget_usd=5.0)
    entry = validate_api_key(store, "key-001")
    assert entry["budget_usd"] == 5.0
    assert entry["active"] is True


def test_register_does_not_include_spent_usd():
    """After register, entry has no spent_usd field (budget tracked via buckets)."""
    store = {}
    register_api_key(store, "key-reg", budget_usd=5.0)
    entry = store["key-reg"]
    assert "spent_usd" not in entry


def test_invalid_api_key_rejected():
    """validate_api_key with unknown key raises ValueError."""
    store = {}
    with pytest.raises(ValueError, match="Invalid API key"):
        validate_api_key(store, "nonexistent-key")


def test_deactivated_key_rejected():
    """Deactivated key raises ValueError."""
    store = {}
    register_api_key(store, "key-002")
    store["key-002"]["active"] = False
    with pytest.raises(ValueError, match="API key deactivated"):
        validate_api_key(store, "key-002")


# --- Bucketed spend tracking ---


def test_spend_bucket_prefix_constant():
    """SPEND_BUCKET_PREFIX constant is exported and equals 'spend:'."""
    assert SPEND_BUCKET_PREFIX == "spend:"


def test_record_spend_uses_bucketed_key():
    """record_spend writes a key matching pattern spend:{api_key}:{YYYYMMDDHH} into store."""
    store = {}
    register_api_key(store, "key-004", budget_usd=10.0)
    record_spend(store, "key-004", 0.50)

    # Find bucket keys in the store
    bucket_keys = [k for k in store if k.startswith(f"{SPEND_BUCKET_PREFIX}key-004:")]
    assert len(bucket_keys) == 1
    # Bucket key format: spend:key-004:YYYYMMDDHH (10-digit hour bucket)
    bucket_key = bucket_keys[0]
    suffix = bucket_key.split(":")[-1]
    assert len(suffix) == 10  # YYYYMMDDHH
    assert suffix.isdigit()
    # Value should be 0.50
    assert store[bucket_key] == pytest.approx(0.50)


def test_record_spend_same_hour_accumulates():
    """Two spends in the same hour bucket accumulate correctly."""
    store = {}
    register_api_key(store, "key-005", budget_usd=10.0)
    record_spend(store, "key-005", 0.50)
    record_spend(store, "key-005", 0.50)

    bucket_keys = [k for k in store if k.startswith(f"{SPEND_BUCKET_PREFIX}key-005:")]
    # Both writes go to the same hour bucket (test runs in < 1 hour)
    assert len(bucket_keys) == 1
    assert store[bucket_keys[0]] == pytest.approx(1.0)


def test_record_spend_different_hour_buckets():
    """Spends with different hour buckets create separate keys."""
    store = {}
    register_api_key(store, "key-006", budget_usd=10.0)
    # Manually write two different hour buckets to simulate
    store[f"{SPEND_BUCKET_PREFIX}key-006:2026032416"] = 0.50
    store[f"{SPEND_BUCKET_PREFIX}key-006:2026032417"] = 0.75
    assert get_total_spend(store, "key-006") == pytest.approx(1.25)


def test_get_total_spend_sums_all_buckets():
    """get_total_spend sums all bucket values for the given api_key prefix."""
    store = {}
    register_api_key(store, "key-007", budget_usd=10.0)
    # Write multiple buckets manually
    store[f"{SPEND_BUCKET_PREFIX}key-007:2026032400"] = 1.0
    store[f"{SPEND_BUCKET_PREFIX}key-007:2026032401"] = 2.0
    store[f"{SPEND_BUCKET_PREFIX}key-007:2026032402"] = 0.5
    assert get_total_spend(store, "key-007") == pytest.approx(3.5)


def test_get_total_spend_ignores_other_keys():
    """get_total_spend does not include buckets for other api_keys."""
    store = {}
    register_api_key(store, "key-A", budget_usd=10.0)
    register_api_key(store, "key-B", budget_usd=10.0)
    store[f"{SPEND_BUCKET_PREFIX}key-A:2026032400"] = 1.0
    store[f"{SPEND_BUCKET_PREFIX}key-B:2026032400"] = 5.0
    assert get_total_spend(store, "key-A") == pytest.approx(1.0)
    assert get_total_spend(store, "key-B") == pytest.approx(5.0)


def test_get_total_spend_zero_when_no_buckets():
    """get_total_spend returns 0.0 when no spend buckets exist."""
    store = {}
    register_api_key(store, "key-008", budget_usd=10.0)
    assert get_total_spend(store, "key-008") == 0.0


def test_validate_rejects_when_budget_exhausted_via_buckets():
    """validate_api_key rejects key when total spend across all buckets >= budget_usd."""
    store = {}
    register_api_key(store, "key-009", budget_usd=1.0)
    store[f"{SPEND_BUCKET_PREFIX}key-009:2026032416"] = 0.60
    store[f"{SPEND_BUCKET_PREFIX}key-009:2026032417"] = 0.40
    with pytest.raises(ValueError, match="Budget exhausted"):
        validate_api_key(store, "key-009")


def test_validate_allows_under_budget_with_buckets():
    """validate_api_key returns entry when total spend is below budget."""
    store = {}
    register_api_key(store, "key-010", budget_usd=5.0)
    store[f"{SPEND_BUCKET_PREFIX}key-010:2026032416"] = 2.0
    entry = validate_api_key(store, "key-010")
    assert entry["budget_usd"] == 5.0


def test_record_spend_does_not_modify_entry_dict():
    """record_spend does not touch the main entry dict (no spent_usd field)."""
    store = {}
    register_api_key(store, "key-011", budget_usd=10.0)
    record_spend(store, "key-011", 1.0)
    entry = store["key-011"]
    assert "spent_usd" not in entry

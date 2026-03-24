"""Tests for cost guard -- budget enforcement, threshold alerting, alert building."""
import pytest

from omniparse.api.cost_guard import (
    BudgetExceededError,
    check_budget,
    should_alert,
    build_cost_alert,
)
from omniparse.api.models import CostAlert


def test_check_budget_no_limit():
    """check_budget with None budget does not raise."""
    check_budget(0.5, None)  # Should not raise


def test_check_budget_within_limit():
    """check_budget within limit does not raise."""
    check_budget(0.5, 1.0)  # Should not raise


def test_budget_exceeded_raises():
    """check_budget at exact limit raises BudgetExceededError."""
    with pytest.raises(BudgetExceededError) as exc_info:
        check_budget(1.0, 1.0)
    assert exc_info.value.spent == 1.0
    assert exc_info.value.budget == 1.0


def test_budget_exceeded_over_limit():
    """check_budget over limit raises BudgetExceededError."""
    with pytest.raises(BudgetExceededError):
        check_budget(1.5, 1.0)


def test_should_alert_below_threshold():
    """Below 80% threshold returns False."""
    assert should_alert(0.5, 1.0, 0.8) is False


def test_should_alert_at_threshold():
    """At exact 80% threshold returns True."""
    assert should_alert(0.8, 1.0, 0.8) is True


def test_should_alert_above_threshold():
    """Above 80% threshold returns True."""
    assert should_alert(0.9, 1.0, 0.8) is True


def test_build_cost_alert_fields():
    """build_cost_alert returns dict with all expected keys and correct message."""
    alert = build_cost_alert("job1", "hash1", 0.9, 1.0)
    assert alert["job_id"] == "job1"
    assert alert["api_key_hash"] == "hash1"
    assert alert["spent_usd"] == 0.9
    assert alert["budget_usd"] == 1.0
    assert alert["threshold_pct"] == 0.8
    assert "90.0%" in alert["message"]


def test_cost_alert_webhook_integration():
    """build_cost_alert output validates against CostAlert model."""
    assert should_alert(0.9, 1.0, 0.8) is True
    alert_dict = build_cost_alert("job-int", "hash-int", 0.9, 1.0, 0.8)
    # Validate against Pydantic model
    alert = CostAlert(**alert_dict)
    assert alert.job_id == "job-int"
    assert alert.spent_usd == 0.9
    assert "90.0%" in alert.message

"""Tests for API data models -- SubmitRequest, WebhookPayload, CostAlert validation."""
import pytest
from pydantic import ValidationError

from omniparse.api.models import (
    SubmitRequest,
    SubmitResponse,
    JobStatus,
    WebhookPayload,
    CostAlert,
)


def test_submit_request_defaults():
    """SubmitRequest() has correct defaults: ce_threshold=0.4, confidence_floor=0.0."""
    req = SubmitRequest()
    assert req.ce_threshold == 0.4
    assert req.confidence_floor == 0.0
    assert req.callback_url is None


def test_submit_request_custom_thresholds():
    """SubmitRequest with custom thresholds validates."""
    req = SubmitRequest(ce_threshold=0.8, confidence_floor=0.5)
    assert req.ce_threshold == 0.8
    assert req.confidence_floor == 0.5


def test_submit_request_ce_threshold_bounds():
    """ce_threshold rejects values outside [0.0, 2.0]."""
    with pytest.raises(ValidationError):
        SubmitRequest(ce_threshold=-0.1)
    with pytest.raises(ValidationError):
        SubmitRequest(ce_threshold=2.1)


def test_webhook_payload_serialization():
    """WebhookPayload with all fields serializes to dict with correct keys."""
    wp = WebhookPayload(
        job_id="job-123",
        status="completed",
        result_url="https://storage.example.com/results/job-123",
        summary={
            "page_count": 5,
            "hitl_flag_count": 1,
            "total_cost_usd": 0.042,
            "processing_time_s": 12.5,
        },
        error=None,
    )
    d = wp.model_dump()
    assert d["job_id"] == "job-123"
    assert d["status"] == "completed"
    assert d["result_url"].startswith("https://")
    assert d["summary"]["page_count"] == 5
    assert d["error"] is None


def test_cost_alert_fields():
    """CostAlert with sample data validates and has all expected fields."""
    alert = CostAlert(
        job_id="job-456",
        api_key_hash="a1b2c3d4e5f6" * 5 + "a1b2",
        spent_usd=0.85,
        budget_usd=1.0,
        threshold_pct=0.8,
        message="Job job-456 has used 85.0% of budget ($0.8500/$1.0000)",
    )
    assert alert.job_id == "job-456"
    assert alert.api_key_hash == "a1b2c3d4e5f6" * 5 + "a1b2"
    assert alert.spent_usd == 0.85
    assert alert.budget_usd == 1.0
    assert alert.threshold_pct == 0.8
    assert "85.0%" in alert.message

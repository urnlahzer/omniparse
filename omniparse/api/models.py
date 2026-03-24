"""API data models -- Pydantic request/response contracts for the production API.

These models define the HTTP interface for OmniParse:
- SubmitRequest: job submission with configurable thresholds
- SubmitResponse: immediate acknowledgment with job ID
- JobStatus: polling response for async job tracking
- WebhookPayload: callback delivery payload for completed jobs
- CostAlert: budget threshold notification
"""
from pydantic import BaseModel, Field


class SubmitRequest(BaseModel):
    """Job submission request with configurable quality thresholds.

    Callers can tune ce_threshold (cross-entropy divergence cutoff for LLM
    arbitration) and confidence_floor (minimum confidence to accept consensus).
    """
    callback_url: str | None = None
    ce_threshold: float = Field(default=0.4, ge=0.0, le=2.0)
    confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    # custom_dictionary: list[str] | None = None
    # Domain-specific terms for NW alignment bias (v2 feature)


class SubmitResponse(BaseModel):
    """Immediate acknowledgment after job submission."""
    job_id: str
    status: str = "processing"


class JobStatus(BaseModel):
    """Polling response for async job tracking."""
    job_id: str
    status: str  # processing, completed, failed, budget_exceeded
    result: dict | None = None
    error: str | None = None


class WebhookPayload(BaseModel):
    """Callback delivery payload for completed/failed jobs."""
    job_id: str
    status: str
    result_url: str
    summary: dict  # page_count, hitl_flag_count, total_cost_usd, processing_time_s
    error: str | None = None


class CostAlert(BaseModel):
    """Budget threshold notification sent when job cost exceeds configured percentage."""
    job_id: str
    api_key_hash: str
    spent_usd: float
    budget_usd: float
    threshold_pct: float
    message: str

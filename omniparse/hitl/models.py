"""HITL review data models -- contracts for human review of low-confidence OCR regions.

HitlRegion mirrors AlignedRegion fields relevant to review (hitl_flag, engine_texts, bounding_box).
ReviewItem provides job-level summary for the review list view.
CorrectionPayload validates POST body when submitting human corrections.
"""
from pydantic import BaseModel, Field
from typing import Optional


class HitlRegion(BaseModel):
    """A single OCR region flagged for human review.

    Contains engine outputs and bounding box for visual reference,
    plus optional corrected_text set after human review.
    """
    region_id: str
    element_type: str
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    engine_texts: dict[str, str] = Field(description="Engine name -> raw text content")
    consensus_text: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    corrected_text: Optional[str] = None


class ReviewItem(BaseModel):
    """Job-level summary for the HITL review list view."""
    job_id: str
    filename: str
    submitted_at: str
    total_regions: int = Field(ge=0)
    hitl_region_count: int = Field(ge=0)
    status: str = Field(default="pending", description="pending, reviewed, partial")


class CorrectionPayload(BaseModel):
    """POST body for submitting human corrections.

    Keys are region_id strings, values are the corrected text.
    """
    corrections: dict[str, str] = Field(description="region_id -> corrected text")

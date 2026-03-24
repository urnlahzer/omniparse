"""Consensus pipeline data models -- contracts between alignment, voting, arbitration, and Markdown."""
from pydantic import BaseModel, Field
from typing import Optional


class AlignedRegion(BaseModel):
    """A region after cross-engine alignment. May contain text from 1-3 engines."""
    region_id: str = Field(description="Unique region ID (from primary engine)")
    element_type: str = Field(description="Canonical element type from VALID_ELEMENT_TYPES")
    bounding_box: list[float] = Field(min_length=4, max_length=4, description="[x1,y1,x2,y2] 300 DPI pixel top-left")
    engine_texts: dict[str, str] = Field(description="Engine name -> raw text content")
    aligned_texts: Optional[dict[str, list[str]]] = Field(default=None, description="Engine name -> NW-aligned character list (after alignment)")
    consensus_text: Optional[str] = Field(default=None, description="Final resolved text after voting/arbitration")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="Consensus confidence score")
    source: str = Field(default="pending", description="How text was resolved: identical, voting, voting_fallback, arbitration, single_engine, hitl_fallback")
    needs_arbitration: bool = Field(default=False, description="True if CE exceeded threshold")
    hitl_flag: bool = Field(default=False, description="True if flagged for human review")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata (table_structure, hierarchy_level, etc.)")


class ConsensusResult(BaseModel):
    """Consensus output for a single page -- input to Markdown compiler."""
    page: int = Field(ge=0, description="Zero-indexed page number")
    regions: list[AlignedRegion] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list, description="Region IDs in reading order (Docling-authoritative)")
    page_metadata: Optional[dict] = Field(default=None, description="Page-level metadata (ground_truth status, etc.)")


class ArbitrationRequest(BaseModel):
    """Request payload for LLM arbiter -- contains ONLY what the LLM needs."""
    region_id: str
    image_bytes: bytes = Field(description="Cropped 300 DPI PNG of the disputed region")
    candidates: dict[str, str] = Field(description="Anonymous label (A/B/C) -> candidate text")
    element_type: str
    bounding_box: list[float] = Field(min_length=4, max_length=4)

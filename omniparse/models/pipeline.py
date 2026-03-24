"""Pipeline response data models -- contracts for pipeline result, processing log, and audit trail.

These models define the structured response from the OmniParse pipeline:
- RegionLog: per-region audit trail (which engines, how resolved, confidence)
- PageLog: per-page processing summary (durations, costs, region decisions)
- ProcessingLog: full job summary (totals, rates, engine versions)
- PipelineResult: top-level response (markdown + log + metadata)
"""
from pydantic import BaseModel, Field
from typing import Optional


class RegionLog(BaseModel):
    """Per-region audit trail for compliance traceability.

    Records which engines processed this region, how the text was resolved,
    and whether LLM arbitration or HITL escalation was triggered.
    Only coordinates are logged -- no image data (PII compliance).
    """
    region_id: str
    element_type: str
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    engines_ran: list[str]
    resolution: str  # identical, voting, voting_fallback, arbitration, hitl_fallback, single_engine
    confidence: float = Field(ge=0.0, le=1.0)
    ce_value: Optional[float] = None
    llm_invoked: bool = False
    hitl_flag: bool = False


class PageLog(BaseModel):
    """Per-page processing summary."""
    page_num: int = Field(ge=0)
    is_ground_truth: bool
    engine_durations: dict[str, float]  # engine name -> seconds
    region_count: int = Field(ge=0)
    regions: list[RegionLog]
    llm_invocation_count: int = Field(ge=0, default=0)
    hitl_escalation_count: int = Field(ge=0, default=0)
    specialist_dispatches: dict[str, int] = Field(default_factory=dict)  # trocr/dots_formula/dots_chart -> count
    estimated_cost_usd: float = Field(ge=0.0, default=0.0)


class ProcessingLog(BaseModel):
    """Full job processing log -- returned inline with Markdown.

    Aggregates all page-level data into job totals for quick inspection.
    """
    pages: list[PageLog]
    total_duration_s: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    total_llm_invocations: int = Field(ge=0, default=0)
    total_hitl_flags: int = Field(ge=0, default=0)
    llm_invocation_rate: float = Field(ge=0.0, default=0.0)
    engine_versions: dict[str, str] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Top-level response from OmniParse pipeline.

    Bundles the compiled Markdown output with a structured processing log
    and document metadata in a single response object.
    """
    markdown: str
    processing_log: ProcessingLog
    metadata: dict  # page_count, filename, processing_time_s, hitl_flag_count

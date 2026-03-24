"""Tests for pipeline data models -- PipelineResult, ProcessingLog, PageLog, RegionLog."""
from omniparse.models.pipeline import PipelineResult, ProcessingLog, PageLog, RegionLog


def test_region_log_creation():
    rl = RegionLog(
        region_id="r_001",
        element_type="printed_text",
        bounding_box=[100, 50, 2400, 120],
        engines_ran=["pdfplumber", "paddleocr", "docling"],
        resolution="voting",
        confidence=0.95,
        ce_value=0.12,
    )
    assert rl.resolution == "voting"
    assert rl.llm_invoked is False


def test_page_log_computed_fields():
    rl = RegionLog(
        region_id="r_001",
        element_type="printed_text",
        bounding_box=[100, 50, 2400, 120],
        engines_ran=["pdfplumber", "paddleocr", "docling"],
        resolution="arbitration",
        confidence=0.95,
        llm_invoked=True,
    )
    pl = PageLog(
        page_num=0,
        is_ground_truth=True,
        engine_durations={"pdfplumber": 0.5},
        region_count=1,
        regions=[rl],
        llm_invocation_count=1,
        hitl_escalation_count=0,
    )
    assert pl.llm_invocation_count == 1
    assert pl.region_count == 1


def test_processing_log_totals():
    log = ProcessingLog(
        pages=[],
        total_duration_s=5.0,
        total_cost_usd=0.01,
        total_llm_invocations=3,
        total_hitl_flags=1,
        llm_invocation_rate=0.05,
    )
    assert log.total_llm_invocations == 3
    assert log.llm_invocation_rate == 0.05


def test_pipeline_result_structure():
    log = ProcessingLog(pages=[], total_duration_s=1.0, total_cost_usd=0.001)
    result = PipelineResult(
        markdown="# Test",
        processing_log=log,
        metadata={"page_count": 1, "filename": "test.pdf", "processing_time_s": 1.0},
    )
    assert result.markdown == "# Test"
    assert result.metadata["page_count"] == 1


def test_pipeline_result_metadata_keys():
    log = ProcessingLog(pages=[], total_duration_s=1.0, total_cost_usd=0.001)
    result = PipelineResult(
        markdown="",
        processing_log=log,
        metadata={
            "page_count": 5,
            "filename": "doc.pdf",
            "processing_time_s": 12.5,
            "hitl_flag_count": 0,
        },
    )
    assert "page_count" in result.metadata
    assert "filename" in result.metadata
    assert "processing_time_s" in result.metadata

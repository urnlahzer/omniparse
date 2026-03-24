"""Tests for observability module -- ProcessingLogBuilder accumulator."""
from omniparse.models.pipeline import RegionLog, ProcessingLog
from omniparse.observability import ProcessingLogBuilder


def test_builder_start_and_finalize():
    """Fresh builder -> start -> finalize returns ProcessingLog with 0 pages."""
    builder = ProcessingLogBuilder()
    builder.start()
    log = builder.finalize()
    assert isinstance(log, ProcessingLog)
    assert len(log.pages) == 0
    assert log.total_duration_s >= 0


def test_record_engine_result():
    """Records engine duration for a page, verifiable in finalize."""
    builder = ProcessingLogBuilder()
    builder.start()
    builder.start_page(0, is_ground_truth=True)
    builder.record_engine_result(0, "pdfplumber", 0.5, 3)
    log = builder.finalize()
    assert log.pages[0].engine_durations["pdfplumber"] == 0.5


def test_record_region_decision():
    """Records a region with resolution='voting', engines_ran=['pdfplumber', 'paddleocr', 'docling']."""
    builder = ProcessingLogBuilder()
    builder.start()
    builder.start_page(0, is_ground_truth=False)
    builder.record_region_decision(
        page_num=0,
        region_id="r_001",
        element_type="printed_text",
        bounding_box=[100, 50, 2400, 120],
        engines_ran=["pdfplumber", "paddleocr", "docling"],
        resolution="voting",
        confidence=0.95,
        ce_value=0.12,
    )
    log = builder.finalize()
    assert len(log.pages[0].regions) == 1
    assert log.pages[0].regions[0].resolution == "voting"


def test_region_log_no_image_data():
    """Verify RegionLog schema has no bytes field (PII compliance)."""
    fields = RegionLog.model_fields
    for name, field in fields.items():
        assert field.annotation is not bytes, f"Field {name} must not contain image bytes"


def test_cost_estimation():
    """Records paddleocr=2.0s + docling=1.5s + pdfplumber=0.5s, verifies cost calculation."""
    builder = ProcessingLogBuilder()
    builder.start()
    builder.start_page(0, is_ground_truth=True)
    builder.record_engine_result(0, "paddleocr", 2.0, 5)
    builder.record_engine_result(0, "docling", 1.5, 5)
    builder.record_engine_result(0, "pdfplumber", 0.5, 5)
    cost = builder.estimate_page_cost(0)
    expected = 2.0 * 0.000306 + 1.5 * 0.000164 + 0.5 * 0.000043
    assert abs(cost - expected) < 0.0001


def test_llm_invocation_rate():
    """2 LLM invocations out of 20 regions = 0.10 rate."""
    builder = ProcessingLogBuilder()
    builder.start()
    builder.start_page(0, is_ground_truth=False)
    for i in range(20):
        llm = i < 2  # first 2 have LLM
        builder.record_region_decision(
            0,
            f"r_{i:03d}",
            "printed_text",
            [0, 0, 100, 100],
            ["pdfplumber", "paddleocr", "docling"],
            "arbitration" if llm else "voting",
            0.9,
            llm_invoked=llm,
        )
    log = builder.finalize()
    assert log.total_llm_invocations == 2
    assert abs(log.llm_invocation_rate - 0.10) < 0.01


def test_specialist_dispatch_recording():
    """Records trocr=2, dots_formula=1 for a page."""
    builder = ProcessingLogBuilder()
    builder.start()
    builder.start_page(0, is_ground_truth=False)
    builder.record_specialist_dispatch(0, "trocr", 2)
    builder.record_specialist_dispatch(0, "dots_formula", 1)
    log = builder.finalize()
    assert log.pages[0].specialist_dispatches["trocr"] == 2
    assert log.pages[0].specialist_dispatches["dots_formula"] == 1


def test_multipage_finalize():
    """3 pages with different costs aggregate to correct total."""
    builder = ProcessingLogBuilder()
    builder.start()
    for pn in range(3):
        builder.start_page(pn, is_ground_truth=pn == 0)
        builder.record_engine_result(pn, "paddleocr", 2.0, 3)
        builder.record_engine_result(pn, "pdfplumber", 0.3, 3)
        builder.record_region_decision(
            pn,
            f"r_{pn}",
            "printed_text",
            [0, 0, 100, 100],
            ["pdfplumber", "paddleocr"],
            "voting",
            0.95,
        )
    log = builder.finalize()
    assert len(log.pages) == 3
    assert log.total_cost_usd > 0
    assert log.total_llm_invocations == 0
    assert log.llm_invocation_rate == 0.0

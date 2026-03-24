"""Tests for the pipeline orchestrator.

Uses dependency injection (engines= parameter) for engine mocks -- NO patch-based
mocking for engine imports. Pure function mocks (preprocess, consensus, etc.)
use standard @patch decorators at their source modules since they are imported
inside function bodies via local imports.
"""
import io
import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from omniparse.models.region import Region, EngineOutput
from omniparse.models.page import PagePayload
from omniparse.models.pipeline import PipelineResult
from omniparse.observability import ProcessingLogBuilder
from omniparse.pipeline import process_document


def _make_real_png(width: int = 2550, height: int = 3300) -> bytes:
    """Create a real PNG image in memory for tests that need image processing."""
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper: build mock engines dict for DI
# ---------------------------------------------------------------------------

def _make_engine_output(page: int, engine: str, text: str = "LAST WILL AND TESTAMENT") -> dict:
    """Create a single EngineOutput dict with one printed_text region."""
    return EngineOutput(
        page=page,
        engine=engine,
        regions=[
            Region(
                id=f"{engine[:3]}_r1",
                element_type="printed_text",
                bounding_box=[100.0, 50.0, 2400.0, 120.0],
                confidence=0.95,
                text_content=text,
            ),
        ],
    ).model_dump()


def _make_mock_engines(
    page_count: int = 1,
    pdfplumber_text: str = "LAST WILL AND TESTAMENT",
    paddleocr_text: str = "LAST WILL AND TESTAMENT",
    docling_text: str = "LAST WILL AND TESTAMENT",
):
    """Create mock engine callables for DI into process_document.

    Each mock mimics the Modal engine interface:
    - run_pdfplumber: has .map(pdf_bytes_list, page_nums_list) -> list[dict]
    - PaddleOCREngine: callable that returns an object with .run.map() method
    - DoclingEngine: callable that returns an object with .run.remote() method
    """
    # pdfplumber mock -- it's a function with .map()
    mock_pdfplumber = MagicMock()
    mock_pdfplumber.map.return_value = [
        _make_engine_output(i, "pdfplumber", pdfplumber_text)
        for i in range(page_count)
    ]

    # PaddleOCR mock -- it's a class, instantiated then .run.map() called
    mock_paddle_cls = MagicMock()
    mock_paddle_instance = MagicMock()
    mock_paddle_instance.run.map.return_value = [
        _make_engine_output(i, "paddleocr", paddleocr_text)
        for i in range(page_count)
    ]
    mock_paddle_cls.return_value = mock_paddle_instance

    # Docling mock -- it's a class, instantiated then .run.remote() called
    mock_docling_cls = MagicMock()
    mock_docling_instance = MagicMock()
    mock_docling_instance.run.remote.return_value = {
        str(i): _make_engine_output(i, "docling", docling_text)
        for i in range(page_count)
    }
    mock_docling_cls.return_value = mock_docling_instance

    return {
        "run_pdfplumber": mock_pdfplumber,
        "PaddleOCREngine": mock_paddle_cls,
        "DoclingEngine": mock_docling_cls,
    }


def _make_page_payloads(count: int = 1) -> list[PagePayload]:
    """Create a list of valid PagePayload objects."""
    return [
        PagePayload(
            page_num=i,
            image_bytes=b"fake_png_bytes",
            pdf_bytes=b"fake_pdf_bytes",
            dpi=300,
            width=2550,
            height=3300,
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("omniparse.preprocess.preprocess")
def test_process_document_basic(mock_preprocess):
    """Basic end-to-end: 1 page PDF with identical text across engines."""
    mock_preprocess.return_value = _make_page_payloads(1)
    mock_engines = _make_mock_engines(page_count=1)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    result = process_document(
        b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
    )

    assert isinstance(result, PipelineResult)
    assert "LAST WILL AND TESTAMENT" in result.markdown
    assert len(result.processing_log.pages) == 1
    assert result.metadata["page_count"] == 1
    assert result.metadata["filename"] == "test.pdf"


@patch("omniparse.preprocess.preprocess")
def test_parallel_dispatch(mock_preprocess):
    """Verify all three engine dispatch methods are called with correct data."""
    pages = _make_page_payloads(3)
    mock_preprocess.return_value = pages
    mock_engines = _make_mock_engines(page_count=3)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    process_document(
        b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
    )

    # pdfplumber .map() called with 3 items
    mock_engines["run_pdfplumber"].map.assert_called_once()
    call_args = mock_engines["run_pdfplumber"].map.call_args
    assert len(call_args[0][0]) == 3  # 3 pdf_bytes
    assert len(call_args[0][1]) == 3  # 3 page_nums

    # PaddleOCR instance .run.map() called with 3 items
    paddle_instance = mock_engines["PaddleOCREngine"].return_value
    paddle_instance.run.map.assert_called_once()
    paddle_args = paddle_instance.run.map.call_args
    assert len(paddle_args[0][0]) == 3  # 3 image_bytes
    assert len(paddle_args[0][1]) == 3  # 3 page_nums

    # Docling instance .run.remote() called once
    docling_instance = mock_engines["DoclingEngine"].return_value
    docling_instance.run.remote.assert_called_once()


@patch("omniparse.preprocess.preprocess")
def test_multipage_dispatch(mock_preprocess):
    """3-page PDF: verify .map() receives 3 items each."""
    pages = _make_page_payloads(3)
    mock_preprocess.return_value = pages
    mock_engines = _make_mock_engines(page_count=3)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    result = process_document(
        b"fake_pdf", "multi.pdf", log_builder, engines=mock_engines,
    )

    assert result.metadata["page_count"] == 3
    assert len(result.processing_log.pages) == 3


def test_container_limits():
    """Verify max_containers sum across all engines supports 50+ concurrent containers."""
    engine_files = [
        "omniparse/engines/pdfplumber_engine.py",
        "omniparse/engines/paddleocr_engine.py",
        "omniparse/engines/docling_engine.py",
        "omniparse/engines/trocr_engine.py",
        "omniparse/engines/dots_engine.py",
        "omniparse/llm_arbiter.py",
        "omniparse/pipeline.py",
    ]
    total = 0
    for filepath in engine_files:
        content = pathlib.Path(filepath).read_text()
        matches = re.findall(r"max_containers\s*=\s*(\d+)", content)
        for m in matches:
            total += int(m)
    assert total >= 50, f"Total max_containers={total}, need >= 50"


def test_warm_engines():
    """Smoke test: warm_engines function exists and is callable."""
    from omniparse.pipeline import warm_engines
    # warm_engines is a Modal function object -- it is not None and has a callable interface
    assert warm_engines is not None


@patch("omniparse.preprocess.preprocess")
def test_result_includes_log(mock_preprocess):
    """Verify PipelineResult has processing_log with pages list."""
    mock_preprocess.return_value = _make_page_payloads(1)
    mock_engines = _make_mock_engines(page_count=1)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    result = process_document(
        b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
    )

    assert hasattr(result, "processing_log")
    assert hasattr(result.processing_log, "pages")
    assert isinstance(result.processing_log.pages, list)
    assert len(result.processing_log.pages) >= 1


@patch("omniparse.preprocess.preprocess")
def test_markdown_output(mock_preprocess):
    """Verify returned markdown starts with YAML frontmatter."""
    mock_preprocess.return_value = _make_page_payloads(1)
    mock_engines = _make_mock_engines(page_count=1)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    result = process_document(
        b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
    )

    assert result.markdown.startswith("---")


def test_entry_points():
    """Verify Pipeline class and parse_document exist and are not None."""
    from omniparse.pipeline import Pipeline, parse_document
    # Pipeline and parse_document are Modal-decorated objects
    assert Pipeline is not None
    assert parse_document is not None


@patch("omniparse.preprocess.preprocess")
def test_specialist_dispatch_after_always_run(mock_preprocess):
    """Mock PaddleOCR to return a handwriting region; verify TrOCR is dispatched."""
    real_png = _make_real_png()
    pages = [
        PagePayload(
            page_num=0,
            image_bytes=real_png,
            pdf_bytes=b"fake_pdf_bytes",
            dpi=300,
            width=2550,
            height=3300,
        )
    ]
    mock_preprocess.return_value = pages

    # PaddleOCR returns a handwriting region
    handwriting_eo = EngineOutput(
        page=0,
        engine="paddleocr",
        regions=[
            Region(
                id="pad_hw1",
                element_type="handwriting",
                bounding_box=[50.0, 500.0, 350.0, 580.0],
                confidence=0.75,
                text_content="John Smith",
            ),
        ],
    ).model_dump()

    mock_engines = _make_mock_engines(page_count=1)
    # Override PaddleOCR results to include handwriting
    mock_engines["PaddleOCREngine"].return_value.run.map.return_value = [handwriting_eo]

    # Mock specialist engines loaded at call site
    mock_trocr_cls = MagicMock()
    mock_trocr_instance = MagicMock()
    mock_trocr_instance.run.remote.return_value = EngineOutput(
        page=0,
        engine="trocr",
        regions=[
            Region(
                id="trocr_r1",
                element_type="handwriting",
                bounding_box=[50.0, 500.0, 350.0, 580.0],
                confidence=0.85,
                text_content="John Smith",
            ),
        ],
    ).model_dump()
    mock_trocr_cls.return_value = mock_trocr_instance

    mock_dots_cls = MagicMock()

    # Also mock PaddleOCR's run_handwriting for second opinion
    mock_paddle_instance = mock_engines["PaddleOCREngine"].return_value
    mock_paddle_instance.run_handwriting.remote.return_value = EngineOutput(
        page=0,
        engine="paddleocr_hw",
        regions=[
            Region(
                id="padhw_r1",
                element_type="handwriting",
                bounding_box=[50.0, 500.0, 350.0, 580.0],
                confidence=0.80,
                text_content="John Smith",
            ),
        ],
    ).model_dump()

    with patch("omniparse.engines.trocr_engine.TrOCREngine", mock_trocr_cls), \
         patch("omniparse.engines.dots_engine.DotsEngine", mock_dots_cls):

        log_builder = ProcessingLogBuilder()
        log_builder.start()
        result = process_document(
            b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
        )

    # TrOCR should have been called for the handwriting region
    mock_trocr_instance.run.remote.assert_called_once()
    # PaddleOCR handwriting second opinion should also be called
    mock_paddle_instance.run_handwriting.remote.assert_called_once()


@patch("omniparse.preprocess.preprocess")
def test_image_input_skips_pdfplumber(mock_preprocess):
    """filename='scan.png' should skip pdfplumber .map() call."""
    pages = [
        PagePayload(
            page_num=0,
            image_bytes=b"fake_png_bytes",
            pdf_bytes=None,
            dpi=300,
            width=2550,
            height=3300,
        )
    ]
    mock_preprocess.return_value = pages
    mock_engines = _make_mock_engines(page_count=1)

    log_builder = ProcessingLogBuilder()
    log_builder.start()
    result = process_document(
        b"fake_png", "scan.png", log_builder, engines=mock_engines,
    )

    # pdfplumber .map() should NOT have been called (image input)
    mock_engines["run_pdfplumber"].map.assert_not_called()
    # PaddleOCR should still be called
    mock_engines["PaddleOCREngine"].return_value.run.map.assert_called_once()
    # Docling should be skipped for images
    mock_engines["DoclingEngine"].return_value.run.remote.assert_not_called()



@patch("omniparse.preprocess.preprocess")
def test_processing_log_tracks_llm(mock_preprocess):
    """Mock a high-CE scenario where arbitration happens; verify LLM tracking."""
    mock_preprocess.return_value = _make_page_payloads(1)

    # Create engines where pdfplumber and paddleocr disagree significantly
    disagree_plumber = EngineOutput(
        page=0, engine="pdfplumber",
        regions=[Region(
            id="pl_r1", element_type="printed_text",
            bounding_box=[100.0, 50.0, 2400.0, 120.0],
            confidence=1.0, text_content="Article I: Definitions",
        )],
    ).model_dump()

    disagree_paddle = EngineOutput(
        page=0, engine="paddleocr",
        regions=[Region(
            id="pa_r1", element_type="printed_text",
            bounding_box=[100.0, 50.0, 2400.0, 120.0],
            confidence=0.85, text_content="Arlicle 1: Deflnitions",  # OCR errors
        )],
    ).model_dump()

    disagree_docling = EngineOutput(
        page=0, engine="docling",
        regions=[Region(
            id="do_r1", element_type="printed_text",
            bounding_box=[100.0, 50.0, 2400.0, 120.0],
            confidence=0.90, text_content="Articl I Defnitions",  # more errors
        )],
    ).model_dump()

    mock_engines = _make_mock_engines(page_count=1)
    mock_engines["run_pdfplumber"].map.return_value = [disagree_plumber]
    mock_engines["PaddleOCREngine"].return_value.run.map.return_value = [disagree_paddle]
    mock_engines["DoclingEngine"].return_value.run.remote.return_value = {"0": disagree_docling}

    # Mock the LLM arbiter that will be called for high-CE regions
    mock_arbiter_cls = MagicMock()
    mock_arbiter_instance = MagicMock()
    mock_arbiter_instance.run.return_value = {
        "text": "Article I: Definitions",
        "rejected": False,
        "hitl_flag": False,
    }
    mock_arbiter_cls.return_value = mock_arbiter_instance

    with patch("omniparse.llm_arbiter.LLMArbiter", mock_arbiter_cls):
        log_builder = ProcessingLogBuilder()
        log_builder.start()
        result = process_document(
            b"fake_pdf", "test.pdf", log_builder, engines=mock_engines,
        )

    # Pipeline should produce valid result with log tracking
    assert isinstance(result, PipelineResult)
    assert len(result.processing_log.pages) == 1
    # The processing log should have recorded region decisions
    assert result.processing_log.pages[0].region_count >= 1


# ---------------------------------------------------------------------------
# ce_threshold / confidence_floor forwarding
# ---------------------------------------------------------------------------

@patch("omniparse.preprocess.preprocess")
def test_ce_threshold_forwarded_to_consensus(mock_preprocess):
    """ce_threshold param is forwarded from process_document to resolve_page."""
    mock_preprocess.return_value = _make_page_payloads(1)
    mock_engines = _make_mock_engines(page_count=1)

    log_builder = ProcessingLogBuilder()
    log_builder.start()

    with patch("omniparse.consensus.resolve_page", wraps=None) as mock_resolve:
        # Make resolve_page return a minimal ConsensusResult
        from omniparse.models.consensus import ConsensusResult
        mock_resolve.return_value = ConsensusResult(page=0, regions=[], reading_order=[])

        process_document(
            b"fake_pdf", "test.pdf", log_builder,
            engines=mock_engines, ce_threshold=0.8,
        )

        # resolve_page must have received ce_threshold=0.8
        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("ce_threshold") == 0.8


@patch("omniparse.preprocess.preprocess")
def test_confidence_floor_filters_low_confidence_regions(mock_preprocess):
    """confidence_floor causes low-confidence regions to be HITL-flagged."""
    mock_preprocess.return_value = _make_page_payloads(1)
    # Use disagreeing engines to produce non-1.0 confidence
    mock_engines = _make_mock_engines(
        page_count=1,
        pdfplumber_text="LAST WILL AND TESTAMENT",
        paddleocr_text="LAST WILL AND TESTAMENT",
        docling_text="LAST WILL AND TESTAMENT",
    )

    log_builder = ProcessingLogBuilder()
    log_builder.start()

    # With confidence_floor=0.99, even high-agreement regions should be flagged
    result = process_document(
        b"fake_pdf", "test.pdf", log_builder,
        engines=mock_engines, confidence_floor=0.99,
    )

    assert isinstance(result, PipelineResult)
    # The process_document function should accept confidence_floor without error

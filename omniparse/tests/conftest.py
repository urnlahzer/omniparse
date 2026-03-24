import io

import pytest
from PIL import Image, ImageDraw

from omniparse.models.region import Region, EngineOutput
from omniparse.models.page import PagePayload


@pytest.fixture
def sample_region():
    return Region(
        id="r_001",
        element_type="printed_text",
        bounding_box=[72.0, 100.5, 540.0, 112.3],
        confidence=1.0,
        text_content="LAST WILL AND TESTAMENT",
        metadata={"coordinate_system": "pdf_points_topleft", "font_size": 14.0, "bold": True},
    )


@pytest.fixture
def sample_engine_output(sample_region):
    return EngineOutput(
        page=0,
        engine="pdfplumber",
        regions=[sample_region],
    )


@pytest.fixture
def sample_page_payload():
    return PagePayload(
        page_num=0,
        image_bytes=b"fake_png_bytes",
        pdf_bytes=b"fake_pdf_bytes",
        dpi=300,
        width=2550,
        height=3300,
        was_rotated=False,
    )


# --- PDF fixture generators for pdfplumber engine tests ---


@pytest.fixture
def born_digital_pdf_bytes():
    """Generate a simple born-digital PDF with text and a table."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "LAST WILL AND TESTAMENT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "I, John Smith, being of sound mind...", ln=True)

    # Add a simple table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Beneficiary", border=1)
    pdf.cell(60, 8, "Share", border=1, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 8, "Jane Smith", border=1)
    pdf.cell(60, 8, "50%", border=1, ln=True)
    pdf.cell(60, 8, "Bob Smith", border=1)
    pdf.cell(60, 8, "50%", border=1, ln=True)

    return pdf.output()


@pytest.fixture
def blank_pdf_bytes():
    """Generate a PDF with a blank page (no text content)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    return pdf.output()


@pytest.fixture
def multipage_pdf_bytes():
    """Generate a 3-page PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(3):
        pdf.add_page()
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Page {i + 1} content", ln=True)
    return pdf.output()


# ---------------------------------------------------------------------------
# Plan 02 fixtures -- programmatic test images and PDFs for preprocessing
# ---------------------------------------------------------------------------

def _make_image_with_text_lines(
    width: int, height: int, dpi: int = 72, fmt: str = "PNG",
) -> bytes:
    """Create an image with horizontal black bars (simulates text lines).

    Useful for skew detection -- OpenCV needs non-zero content.
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    bar_height = max(4, height // 20)
    for y in range(bar_height * 2, height - bar_height * 2, bar_height * 2):
        draw.rectangle(
            [width // 10, y, width - width // 10, y + bar_height],
            fill="black",
        )
    buf = io.BytesIO()
    img.save(buf, format=fmt, dpi=(dpi, dpi))
    return buf.getvalue()


@pytest.fixture
def portrait_png_72dpi() -> bytes:
    """100x150 portrait PNG at 72 DPI with text-like bars."""
    return _make_image_with_text_lines(100, 150, dpi=72, fmt="PNG")


@pytest.fixture
def portrait_png_300dpi() -> bytes:
    """100x150 portrait PNG at 300 DPI with text-like bars."""
    return _make_image_with_text_lines(100, 150, dpi=300, fmt="PNG")


@pytest.fixture
def landscape_png_300dpi() -> bytes:
    """200x100 landscape PNG at 300 DPI."""
    return _make_image_with_text_lines(200, 100, dpi=300, fmt="PNG")


@pytest.fixture
def portrait_jpg_bytes() -> bytes:
    """100x150 portrait JPEG at 300 DPI."""
    return _make_image_with_text_lines(100, 150, dpi=300, fmt="JPEG")


@pytest.fixture
def portrait_tiff_bytes() -> bytes:
    """100x150 portrait TIFF at 300 DPI."""
    return _make_image_with_text_lines(100, 150, dpi=300, fmt="TIFF")


@pytest.fixture
def skewed_png_10deg() -> bytes:
    """A portrait image deliberately rotated 10 degrees to simulate skew."""
    img = Image.new("RGB", (300, 400), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 360, 30):
        draw.rectangle([30, y, 270, y + 12], fill="black")
    img = img.rotate(10, expand=True, fillcolor="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    return buf.getvalue()


@pytest.fixture
def skewed_png_20deg() -> bytes:
    """A portrait image rotated 20 degrees -- beyond the 15-degree limit."""
    img = Image.new("RGB", (300, 400), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 360, 30):
        draw.rectangle([30, y, 270, y + 12], fill="black")
    img = img.rotate(20, expand=True, fillcolor="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    return buf.getvalue()


@pytest.fixture
def three_page_pdf_bytes() -> bytes:
    """A minimal 3-page PDF generated with fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(3):
        pdf.add_page()
        pdf.set_font("Helvetica", size=24)
        pdf.cell(text=f"Page {i + 1}")
    return pdf.output()


@pytest.fixture
def single_page_pdf_bytes() -> bytes:
    """A minimal 1-page PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=24)
    pdf.cell(text="Single page document")
    return pdf.output()


@pytest.fixture
def corrupted_pdf_bytes() -> bytes:
    """Bytes that look like a PDF header but are actually garbage."""
    return b"%PDF-1.4 this is not a valid pdf at all"


# ---------------------------------------------------------------------------
# Plan 03-01 fixtures -- consensus pipeline test data
# ---------------------------------------------------------------------------


@pytest.fixture
def three_engine_outputs():
    """Three EngineOutput objects with known overlapping bounding boxes.

    Region 1 (header/printed_text): Identical text across all engines.
    Region 2 (printed_text): Differing text (OCR errors in PaddleOCR).
    Table region: Identical simple text across all engines.

    Bounding boxes chosen so IoU > 0.85 between engines for the same region.
    """
    pdfplumber_output = EngineOutput(
        page=0,
        engine="pdfplumber",
        regions=[
            Region(
                id="pdf_r1",
                element_type="printed_text",
                bounding_box=[100.0, 50.0, 2400.0, 120.0],
                bounding_box_norm=[100.0/2550, 50.0/3300, 2400.0/2550, 120.0/3300],
                confidence=1.0,
                text_content="LAST WILL AND TESTAMENT",
            ),
            Region(
                id="pdf_r2",
                element_type="printed_text",
                bounding_box=[100.0, 150.0, 2400.0, 200.0],
                bounding_box_norm=[100.0/2550, 150.0/3300, 2400.0/2550, 200.0/3300],
                confidence=1.0,
                text_content="Article I: Definitions",
            ),
            Region(
                id="pdf_t1",
                element_type="table",
                bounding_box=[100.0, 300.0, 2400.0, 600.0],
                bounding_box_norm=[100.0/2550, 300.0/3300, 2400.0/2550, 600.0/3300],
                confidence=1.0,
                text_content="Beneficiary | Share\nJane Smith | 50%",
                table_structure={"rows": 2, "cols": 2, "has_merged_cells": False},
            ),
        ],
    )

    paddleocr_output = EngineOutput(
        page=0,
        engine="paddleocr",
        regions=[
            Region(
                id="pad_r1",
                element_type="printed_text",
                bounding_box=[105.0, 52.0, 2395.0, 118.0],
                bounding_box_norm=[105.0/2550, 52.0/3300, 2395.0/2550, 118.0/3300],
                confidence=0.95,
                text_content="LAST WILL AND TESTAMENT",
            ),
            Region(
                id="pad_r2",
                element_type="printed_text",
                bounding_box=[103.0, 152.0, 2397.0, 198.0],
                bounding_box_norm=[103.0/2550, 152.0/3300, 2397.0/2550, 198.0/3300],
                confidence=0.92,
                text_content="Artlcle I: Definltions",
            ),
            Region(
                id="pad_t1",
                element_type="table",
                bounding_box=[105.0, 305.0, 2395.0, 595.0],
                bounding_box_norm=[105.0/2550, 305.0/3300, 2395.0/2550, 595.0/3300],
                confidence=0.90,
                text_content="Beneficiary | Share\nJane Smith | 50%",
                table_structure={"rows": 2, "cols": 2, "has_merged_cells": False},
            ),
        ],
    )

    docling_output = EngineOutput(
        page=0,
        engine="docling",
        regions=[
            Region(
                id="doc_r1",
                element_type="printed_text",
                bounding_box=[102.0, 48.0, 2398.0, 122.0],
                bounding_box_norm=[102.0/2550, 48.0/3300, 2398.0/2550, 122.0/3300],
                confidence=0.98,
                text_content="LAST WILL AND TESTAMENT",
                metadata={"hierarchy_level": 1},
            ),
            Region(
                id="doc_r2",
                element_type="printed_text",
                bounding_box=[101.0, 149.0, 2399.0, 201.0],
                bounding_box_norm=[101.0/2550, 149.0/3300, 2399.0/2550, 201.0/3300],
                confidence=0.97,
                text_content="Article I: Definitions",
                metadata={"hierarchy_level": 2},
            ),
            Region(
                id="doc_t1",
                element_type="table",
                bounding_box=[102.0, 298.0, 2398.0, 602.0],
                bounding_box_norm=[102.0/2550, 298.0/3300, 2398.0/2550, 602.0/3300],
                confidence=0.96,
                text_content="Beneficiary | Share\nJane Smith | 50%",
                table_structure={"rows": 2, "cols": 2, "has_merged_cells": False},
            ),
        ],
    )

    return pdfplumber_output, paddleocr_output, docling_output


@pytest.fixture
def two_engine_outputs(three_engine_outputs):
    """Two EngineOutput objects (pdfplumber + paddleocr) for 2-engine voting tests."""
    pdfplumber_output, paddleocr_output, _ = three_engine_outputs
    return pdfplumber_output, paddleocr_output


# ---------------------------------------------------------------------------
# Plan 04-01 fixtures -- specialist dispatch and rendering test data
# ---------------------------------------------------------------------------


@pytest.fixture
def handwriting_region():
    """A handwriting region in the margin area (x1 < 15% of 2550 = 382.5)."""
    return Region(
        id="r_hw_001",
        element_type="handwriting",
        bounding_box=[50.0, 500.0, 350.0, 580.0],
        bounding_box_norm=[50.0/2550, 500.0/3300, 350.0/2550, 580.0/3300],
        confidence=0.75,
        text_content="John Smith",
        metadata={"page_width": 2550.0},
    )


@pytest.fixture
def inline_handwriting_region():
    """A handwriting region in the document body area (well inside margins)."""
    return Region(
        id="r_hw_002",
        element_type="handwriting",
        bounding_box=[500.0, 800.0, 1800.0, 870.0],
        bounding_box_norm=[500.0/2550, 800.0/3300, 1800.0/2550, 870.0/3300],
        confidence=0.80,
        text_content="Signed by the testator",
        metadata={"page_width": 2550.0},
    )


@pytest.fixture
def formula_region():
    """A formula region with high confidence."""
    return Region(
        id="r_fm_001",
        element_type="formula",
        bounding_box=[400.0, 600.0, 2000.0, 700.0],
        bounding_box_norm=[400.0/2550, 600.0/3300, 2000.0/2550, 700.0/3300],
        confidence=0.90,
        text_content="E = mc^2",
    )


@pytest.fixture
def chart_region():
    """A chart region with moderate confidence."""
    return Region(
        id="r_ch_001",
        element_type="chart",
        bounding_box=[200.0, 1000.0, 2200.0, 2000.0],
        bounding_box_norm=[200.0/2550, 1000.0/3300, 2200.0/2550, 2000.0/3300],
        confidence=0.85,
        text_content="",
        metadata={"svg_content": "<svg><rect width='100' height='50'/></svg>"},
    )


@pytest.fixture
def cross_granularity_outputs():
    """Cross-granularity engine outputs: line-level vs word-level regions.

    Engine A (pdfplumber): 1 large line-level region.
    Engine B (docling): 3 word-level regions fully inside engine A's box.
    Engine C (paddleocr): 1 line-level region matching engine A's box (IoU > 0.5).

    The word-level regions have IoU < 0.5 against the line-level region
    but containment >= 0.7 (they are fully inside).
    """
    pdfplumber_output = EngineOutput(
        page=0,
        engine="pdfplumber",
        regions=[
            Region(
                id="pdf_line1",
                element_type="printed_text",
                bounding_box=[127.5, 330.0, 2422.5, 495.0],
                bounding_box_norm=[0.05, 0.10, 0.95, 0.15],
                confidence=1.0,
                text_content="WHEREAS the parties agree",
            ),
        ],
    )

    docling_output = EngineOutput(
        page=0,
        engine="docling",
        regions=[
            Region(
                id="doc_word1",
                element_type="printed_text",
                bounding_box=[127.5, 330.0, 765.0, 495.0],
                bounding_box_norm=[0.05, 0.10, 0.30, 0.15],
                confidence=0.95,
                text_content="WHEREAS",
            ),
            Region(
                id="doc_word2",
                element_type="printed_text",
                bounding_box=[816.0, 330.0, 1402.5, 495.0],
                bounding_box_norm=[0.32, 0.10, 0.55, 0.15],
                confidence=0.93,
                text_content="the",
            ),
            Region(
                id="doc_word3",
                element_type="printed_text",
                bounding_box=[1453.5, 330.0, 2422.5, 495.0],
                bounding_box_norm=[0.57, 0.10, 0.95, 0.15],
                confidence=0.91,
                text_content="parties agree",
            ),
        ],
    )

    paddleocr_output = EngineOutput(
        page=0,
        engine="paddleocr",
        regions=[
            Region(
                id="pad_line1",
                element_type="printed_text",
                bounding_box=[140.0, 335.0, 2410.0, 490.0],
                bounding_box_norm=[0.055, 0.102, 0.945, 0.148],
                confidence=0.92,
                text_content="WHEREAS the parties agree",
            ),
        ],
    )

    return pdfplumber_output, docling_output, paddleocr_output


@pytest.fixture
def y_offset_outputs():
    """Y-offset engine outputs: same text, different vertical positions.

    Simulates the primary remaining gap from Phase 8: PaddleOCR and Docling
    detect the same text line but with 40-100px vertical shift. IoU=~0.21
    (too low for 0.5 threshold), but center_distance=~0.023 and IoU > 0.05.
    """
    paddleocr_output = EngineOutput(
        page=0,
        engine="paddleocr",
        regions=[
            Region(
                id="pad_line1",
                element_type="printed_text",
                bounding_box=[127.5, 330.0, 2422.5, 495.0],
                bounding_box_norm=[0.05, 0.10, 0.95, 0.15],
                confidence=0.92,
                text_content="WHEREAS the parties agree",
            ),
        ],
    )
    docling_output = EngineOutput(
        page=0,
        engine="docling",
        regions=[
            Region(
                id="doc_line1",
                element_type="printed_text",
                bounding_box=[140.0, 440.0, 2410.0, 587.0],
                bounding_box_norm=[0.055, 0.133, 0.945, 0.178],
                confidence=0.95,
                text_content="WHEREAS the parties agree",
            ),
        ],
    )
    return paddleocr_output, docling_output


@pytest.fixture
def no_overlap_close_centers():
    """Two regions with close centers but ZERO overlap (stacked text lines).

    This is the false-match scenario that caused the Phase 8 center-distance revert.
    Center distance is small (~0.03) but IoU is 0.0 because boxes don't overlap.
    """
    engine_a = EngineOutput(
        page=0,
        engine="engine_a",
        regions=[
            Region(
                id="a_line1",
                element_type="printed_text",
                bounding_box=[127.5, 330.0, 2422.5, 400.0],
                bounding_box_norm=[0.05, 0.10, 0.95, 0.121],
                confidence=0.92,
                text_content="Line one text",
            ),
        ],
    )
    engine_b = EngineOutput(
        page=0,
        engine="engine_b",
        regions=[
            Region(
                id="b_line2",
                element_type="printed_text",
                bounding_box=[127.5, 410.0, 2422.5, 480.0],
                bounding_box_norm=[0.05, 0.124, 0.95, 0.145],
                confidence=0.95,
                text_content="Line two text",
            ),
        ],
    )
    return engine_a, engine_b


@pytest.fixture
def clustering_orphan_outputs():
    """Regions that fail IoU, center-distance, containment, and WBF but overlap slightly.

    IoU=~0.11 (above 0.08 clustering threshold, distance=0.89 < 0.92).
    Center distance=0.316 > 0.05 (center rescue won't catch).
    Containment ratio=0.20 < 0.6 (containment won't catch).
    IoU=0.11 < 0.3 (WBF won't catch).
    These should be caught by agglomerative clustering.
    """
    engine_a = EngineOutput(
        page=0,
        engine="engine_a",
        regions=[
            Region(
                id="a_r1",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 1275.0, 660.0],
                bounding_box_norm=[0.0, 0.0, 0.50, 0.20],
                confidence=0.90,
                text_content="Cluster candidate A",
            ),
        ],
    )
    engine_b = EngineOutput(
        page=0,
        engine="engine_b",
        regions=[
            Region(
                id="b_r1",
                element_type="printed_text",
                bounding_box=[765.0, 330.0, 2040.0, 990.0],
                bounding_box_norm=[0.30, 0.10, 0.80, 0.30],
                confidence=0.85,
                text_content="Cluster candidate B",
            ),
        ],
    )
    return engine_a, engine_b


@pytest.fixture
def mixed_paddleocr_output(handwriting_region, formula_region, chart_region):
    """EngineOutput with handwriting + formula + chart + printed_text regions."""
    printed_region = Region(
        id="r_pt_001",
        element_type="printed_text",
        bounding_box=[100.0, 100.0, 2400.0, 180.0],
        bounding_box_norm=[100.0/2550, 100.0/3300, 2400.0/2550, 180.0/3300],
        confidence=0.95,
        text_content="Article I: Definitions",
    )
    return EngineOutput(
        page=0,
        engine="paddleocr",
        regions=[printed_region, handwriting_region, formula_region, chart_region],
    )

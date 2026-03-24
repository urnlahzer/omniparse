"""Baseline integration tests -- captures v1.0 multi-engine resolution rate.

Requirement: MEAS-01

Tests the alignment pipeline (match_regions_across_engines + align_region_group)
on synthetic engine outputs simulating diverse PDF characteristics.
Records per-PDF and aggregate multi-engine resolution rate as the v1.0
baseline for Phase 10 comparison.

Real PDF-based tests (using OMNIPARSE_SAMPLES_DIR env var) are marked with
@requires_samples and skip when the directory is not set or not available.
"""
import json
import os
from pathlib import Path

import pytest

from omniparse.models.region import Region, EngineOutput
from omniparse.alignment import match_regions_across_engines, align_region_group
from omniparse.normalization import normalize_to_unit


FIXTURES_DIR = Path(__file__).parent / "fixtures"
_samples_env = os.environ.get("OMNIPARSE_SAMPLES_DIR", "")
SAMPLES_DIR = Path(_samples_env) if _samples_env else None

requires_samples = pytest.mark.skipif(
    SAMPLES_DIR is None or not SAMPLES_DIR.exists(),
    reason="Test PDF samples not found (set OMNIPARSE_SAMPLES_DIR)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_resolution_rate(matched_groups: list[dict]) -> dict:
    """Compute multi-engine vs single-engine resolution rate."""
    multi = sum(1 for g in matched_groups if len(g["regions"]) > 1)
    single = sum(1 for g in matched_groups if len(g["regions"]) == 1)
    total = multi + single
    return {
        "multi_engine": multi,
        "single_engine": single,
        "total": total,
        "multi_engine_rate": round(multi / total, 4) if total > 0 else 0.0,
    }


def make_region(
    id, element_type, bbox_px, confidence, text,
    page_w=2550, page_h=3300, **kwargs,
):
    """Create a Region with both pixel and [0,1] normalized bounding boxes."""
    norm = normalize_to_unit(bbox_px, page_w, page_h)
    return Region(
        id=id, element_type=element_type, bounding_box=bbox_px,
        bounding_box_norm=norm, confidence=confidence, text_content=text,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# fpdf2-generated PDF for D-10
# ---------------------------------------------------------------------------


def _make_controlled_pdf() -> tuple[bytes, list[dict]]:
    """Generate a PDF with fpdf2 and return (pdf_bytes, known_regions).

    Each known_region is a dict with element_type, bbox_px (300 DPI pixel coords),
    and text content. The PDF is 8.5x11 inches = 612x792 points = 2550x3300 pixels
    at 300 DPI.
    """
    from fpdf import FPDF

    pdf = FPDF(unit="pt", format=(612, 792))
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    # Place 3 text elements at known coordinates (in PDF points, top-left origin)
    # Element 1: title at top
    pdf.set_xy(72, 36)
    pdf.cell(468, 24, "Trust Agreement", align="C")
    # Element 2: body text
    pdf.set_xy(72, 100)
    pdf.cell(468, 14, "This agreement is entered into on...")
    # Element 3: footer
    pdf.set_xy(72, 750)
    pdf.cell(468, 14, "Page 1 of 10")

    pdf_bytes = pdf.output()

    # Convert PDF point coords to 300 DPI pixel coords (multiply by 300/72)
    scale = 300.0 / 72.0
    known_regions = [
        {
            "element_type": "header",
            "bbox_px": [72 * scale, 36 * scale, 540 * scale, 60 * scale],
            "text": "Trust Agreement",
        },
        {
            "element_type": "printed_text",
            "bbox_px": [72 * scale, 100 * scale, 540 * scale, 114 * scale],
            "text": "This agreement is entered into on...",
        },
        {
            "element_type": "footer",
            "bbox_px": [72 * scale, 750 * scale, 540 * scale, 764 * scale],
            "text": "Page 1 of 10",
        },
    ]
    return pdf_bytes, known_regions


# ---------------------------------------------------------------------------
# Synthetic scenario fixtures
# ---------------------------------------------------------------------------


def _scanned_legal_filing():
    """3 engines, 5 regions each. Simulates scan jitter and type disagreement.

    pdfplumber and docling label title "header", paddleocr labels it "printed_text".
    One docling region has offset causing IoU < 0.5 (scan noise).
    Expected: 4 multi-engine + 2 single-engine = ~67% rate.
    """
    page_w, page_h = 2550, 3300

    # Region coordinates (pixel, 300 DPI)
    title_bbox = [300.0, 150.0, 2250.0, 210.0]
    body1_bbox = [300.0, 300.0, 2250.0, 360.0]
    body2_bbox = [300.0, 450.0, 2250.0, 510.0]
    footer_bbox = [300.0, 3100.0, 2250.0, 3160.0]
    table_bbox = [300.0, 600.0, 2250.0, 1200.0]

    # pdfplumber regions (labels title as "header")
    pdf_regions = [
        make_region("pdf-0", "header", title_bbox, 1.0, "DECLARATION OF TRUST", page_w, page_h),
        make_region("pdf-1", "printed_text", body1_bbox, 1.0, "Article I: Definitions", page_w, page_h),
        make_region("pdf-2", "printed_text", body2_bbox, 1.0, "Article II: Property", page_w, page_h),
        make_region("pdf-3", "footer", footer_bbox, 1.0, "Page 1 of 12", page_w, page_h),
        make_region("pdf-4", "table", table_bbox, 1.0, "Beneficiary | Share\nAlice | 50%", page_w, page_h,
                    table_structure={"rows": 2, "cols": 2, "has_merged_cells": False}),
    ]

    # PaddleOCR regions (labels title as "printed_text", slight jitter)
    paddle_regions = [
        make_region("pad-0", "printed_text", [305.0, 153.0, 2245.0, 207.0], 0.93,
                    "DECLARATION OF TRUST", page_w, page_h),
        make_region("pad-1", "printed_text", [303.0, 302.0, 2247.0, 358.0], 0.91,
                    "Artlcle I: Definltions", page_w, page_h),
        make_region("pad-2", "printed_text", [304.0, 452.0, 2248.0, 508.0], 0.90,
                    "Article II: Property", page_w, page_h),
        make_region("pad-3", "footer", [302.0, 3102.0, 2248.0, 3158.0], 0.88,
                    "Page 1 of 12", page_w, page_h),
        make_region("pad-4", "table", [306.0, 603.0, 2246.0, 1197.0], 0.87,
                    "Beneficiary | Share\nAlice | 50%", page_w, page_h,
                    table_structure={"rows": 2, "cols": 2, "has_merged_cells": False}),
    ]

    # Docling regions (labels title as "header", one region severely offset)
    docling_regions = [
        make_region("doc-0", "header", [302.0, 148.0, 2248.0, 212.0], 0.96,
                    "DECLARATION OF TRUST", page_w, page_h,
                    metadata={"hierarchy_level": 1}),
        make_region("doc-1", "printed_text", [301.0, 298.0, 2249.0, 362.0], 0.95,
                    "Article I: Definitions", page_w, page_h,
                    metadata={"hierarchy_level": 2}),
        # Severely offset -- IoU with others < 0.5 (scan noise misalignment)
        make_region("doc-2", "printed_text", [800.0, 700.0, 2600.0, 760.0], 0.60,
                    "Article II: Property", page_w, page_h),
        make_region("doc-3", "footer", [303.0, 3099.0, 2247.0, 3161.0], 0.94,
                    "Page 1 of 12", page_w, page_h),
        make_region("doc-4", "table", [304.0, 598.0, 2248.0, 1202.0], 0.93,
                    "Beneficiary | Share\nAlice | 50%", page_w, page_h,
                    table_structure={"rows": 2, "cols": 2, "has_merged_cells": False}),
    ]

    return [
        EngineOutput(engine="pdfplumber", regions=pdf_regions, page=0),
        EngineOutput(engine="paddleocr", regions=paddle_regions, page=0),
        EngineOutput(engine="docling", regions=docling_regions, page=0),
    ]


def _born_digital_form():
    """3 engines, 4 regions each. Near-identical bboxes (IoU > 0.95).

    Expected: ~100% multi-engine rate.
    """
    page_w, page_h = 2550, 3300

    title_bbox = [300.0, 100.0, 2250.0, 160.0]
    field1_bbox = [300.0, 250.0, 1200.0, 290.0]
    field2_bbox = [300.0, 350.0, 1200.0, 390.0]
    footer_bbox = [300.0, 3100.0, 2250.0, 3140.0]

    engines = []
    for eng_name, jitter, conf in [("pdfplumber", 0, 1.0), ("paddleocr", 2, 0.95), ("docling", 1, 0.97)]:
        regions = [
            make_region(f"{eng_name}-0", "printed_text",
                        [b + jitter for b in title_bbox], conf,
                        "Application for Benefits", page_w, page_h),
            make_region(f"{eng_name}-1", "printed_text",
                        [b + jitter for b in field1_bbox], conf,
                        "Name: John Smith", page_w, page_h),
            make_region(f"{eng_name}-2", "printed_text",
                        [b + jitter for b in field2_bbox], conf,
                        "Date: 2026-01-15", page_w, page_h),
            make_region(f"{eng_name}-3", "footer",
                        [b + jitter for b in footer_bbox], conf,
                        "Form 1040-A", page_w, page_h),
        ]
        engines.append(EngineOutput(engine=eng_name, regions=regions, page=0))

    return engines


def _degraded_historical():
    """3 engines with varying region counts. Simulates degraded scan.

    pdfplumber: 2 regions (sparse text layer).
    PaddleOCR: 5 regions (aggressive detection).
    Docling: 4 regions (moderate detection).
    Expected: low multi-engine rate (~30-40%).
    """
    page_w, page_h = 2550, 3300

    pdf_regions = [
        make_region("pdf-0", "printed_text", [300.0, 200.0, 2200.0, 260.0], 1.0,
                    "Last Will and Testament", page_w, page_h),
        make_region("pdf-1", "printed_text", [300.0, 400.0, 2200.0, 460.0], 1.0,
                    "I hereby declare", page_w, page_h),
    ]

    paddle_regions = [
        make_region("pad-0", "printed_text", [305.0, 203.0, 2195.0, 257.0], 0.80,
                    "Last Will and Testament", page_w, page_h),
        make_region("pad-1", "printed_text", [303.0, 403.0, 2197.0, 457.0], 0.75,
                    "I hereby declare", page_w, page_h),
        make_region("pad-2", "printed_text", [300.0, 600.0, 1500.0, 650.0], 0.60,
                    "faded text fragment", page_w, page_h),
        make_region("pad-3", "handwriting", [1600.0, 600.0, 2400.0, 680.0], 0.55,
                    "John Smith", page_w, page_h),
        make_region("pad-4", "printed_text", [300.0, 800.0, 2200.0, 860.0], 0.50,
                    "partial text", page_w, page_h),
    ]

    docling_regions = [
        make_region("doc-0", "printed_text", [302.0, 198.0, 2198.0, 262.0], 0.90,
                    "Last Will and Testament", page_w, page_h,
                    metadata={"hierarchy_level": 1}),
        make_region("doc-1", "printed_text", [301.0, 398.0, 2199.0, 462.0], 0.88,
                    "I hereby declare", page_w, page_h),
        make_region("doc-2", "printed_text", [302.0, 602.0, 1498.0, 648.0], 0.65,
                    "faded text fragment", page_w, page_h),
        make_region("doc-3", "printed_text", [301.0, 802.0, 2199.0, 858.0], 0.60,
                    "partial text", page_w, page_h),
    ]

    return [
        EngineOutput(engine="pdfplumber", regions=pdf_regions, page=0),
        EngineOutput(engine="paddleocr", regions=paddle_regions, page=0),
        EngineOutput(engine="docling", regions=docling_regions, page=0),
    ]


def _mixed_content():
    """3 engines with printed_text, handwriting, table, formula.

    Tests that type compatibility correctly isolates handwriting and tables.
    Expected: printed_text matches, specialist regions stay isolated.
    """
    page_w, page_h = 2550, 3300

    engines = []
    for eng_name, jitter, conf in [("pdfplumber", 0, 1.0), ("paddleocr", 3, 0.92), ("docling", 1, 0.96)]:
        regions = [
            make_region(f"{eng_name}-pt", "printed_text",
                        [300.0 + jitter, 100.0 + jitter, 2200.0 + jitter, 160.0 + jitter],
                        conf, "Article I: Definitions", page_w, page_h),
            make_region(f"{eng_name}-hw", "handwriting",
                        [300.0 + jitter, 300.0 + jitter, 1500.0 + jitter, 380.0 + jitter],
                        conf * 0.8, "John Smith", page_w, page_h),
            make_region(f"{eng_name}-tbl", "table",
                        [300.0 + jitter, 500.0 + jitter, 2200.0 + jitter, 900.0 + jitter],
                        conf, "Beneficiary | Share\nAlice | 50%", page_w, page_h,
                        table_structure={"rows": 2, "cols": 2, "has_merged_cells": False}),
            make_region(f"{eng_name}-fm", "formula",
                        [300.0 + jitter, 1000.0 + jitter, 1500.0 + jitter, 1060.0 + jitter],
                        conf * 0.9, "E = mc^2", page_w, page_h),
        ]
        engines.append(EngineOutput(engine=eng_name, regions=regions, page=0))

    return engines


def _type_disagreement():
    """3 engines, 3 regions each. All engines disagree on element_type for same spatial region.

    Engine A: header, Engine B: printed_text, Engine C: page_number.
    All three are in Group A (text), so they should match.
    Expected: 100% multi-engine rate.
    """
    page_w, page_h = 2550, 3300

    # All three regions cover the same spatial area with slight jitter
    base_bboxes = [
        [300.0, 100.0, 2200.0, 160.0],
        [300.0, 250.0, 2200.0, 310.0],
        [300.0, 400.0, 2200.0, 460.0],
    ]

    engine_types = [
        ("engine_a", ["header", "header", "printed_text"], 0, 0.95),
        ("engine_b", ["printed_text", "printed_text", "footer"], 2, 0.92),
        ("engine_c", ["page_number", "page_number", "page_number"], 1, 0.90),
    ]

    engines = []
    for eng_name, types, jitter, conf in engine_types:
        regions = []
        for i, (bbox, etype) in enumerate(zip(base_bboxes, types)):
            regions.append(
                make_region(f"{eng_name}-{i}", etype,
                            [b + jitter for b in bbox],
                            conf, f"Text content {i}", page_w, page_h)
            )
        engines.append(EngineOutput(engine=eng_name, regions=regions, page=0))

    return engines


def _generated_controlled():
    """D-10: fpdf2-generated PDF with known layout -> synthetic EngineOutput fixtures."""
    _pdf_bytes, known_regions = _make_controlled_pdf()
    # _pdf_bytes proves we CAN generate the PDF (D-10). The EngineOutput fixtures
    # model what engines would produce from this known layout.
    page_w, page_h = 2550, 3300  # 8.5x11 at 300 DPI

    engine_a_regions = [
        make_region(f"ctrl-a-{i}", kr["element_type"], kr["bbox_px"],
                    0.95, kr["text"], page_w, page_h)
        for i, kr in enumerate(known_regions)
    ]
    engine_b_regions = [
        make_region(f"ctrl-b-{i}", kr["element_type"],
                    [c + (3 if j % 2 == 0 else -2) for j, c in enumerate(kr["bbox_px"])],
                    0.92, kr["text"], page_w, page_h)
        for i, kr in enumerate(known_regions)
    ]

    return [
        EngineOutput(engine="engine_a", regions=engine_a_regions, page=0),
        EngineOutput(engine="engine_b", regions=engine_b_regions, page=0),
    ]


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------


class TestBaselineMetrics:
    """Run alignment on all synthetic scenarios and capture baseline rates."""

    SCENARIOS = [
        ("scanned_legal_filing", _scanned_legal_filing),
        ("born_digital_form", _born_digital_form),
        ("degraded_historical", _degraded_historical),
        ("mixed_content", _mixed_content),
        ("type_disagreement", _type_disagreement),
        ("generated_controlled", _generated_controlled),
    ]

    def test_baseline_captures_all_scenarios(self):
        """Run all scenarios and record baseline metrics (MEAS-01)."""
        results = {}
        for name, scenario_fn in self.SCENARIOS:
            engine_outputs = scenario_fn()
            matched = match_regions_across_engines(engine_outputs)
            rate = compute_resolution_rate(matched)
            results[name] = rate
            # Print to stdout for visibility
            print(f"\n  {name}: {rate['multi_engine_rate']:.1%} "
                  f"({rate['multi_engine']}/{rate['total']})")

        # Compute aggregate
        total_multi = sum(r["multi_engine"] for r in results.values())
        total_all = sum(r["total"] for r in results.values())
        aggregate_rate = round(total_multi / total_all, 4) if total_all > 0 else 0.0
        results["_aggregate"] = {
            "multi_engine": total_multi,
            "total": total_all,
            "multi_engine_rate": aggregate_rate,
        }
        print(f"\n  AGGREGATE: {aggregate_rate:.1%} ({total_multi}/{total_all})")

        # Write baseline JSON fixture (per D-11)
        FIXTURES_DIR.mkdir(exist_ok=True)
        baseline_path = FIXTURES_DIR / "baseline_v1.json"
        baseline_path.write_text(json.dumps(results, indent=2) + "\n")

        # Assertions: verify the test actually ran all scenarios
        assert len(results) == len(self.SCENARIOS) + 1  # +1 for _aggregate
        assert all(r["total"] > 0 for name, r in results.items() if name != "_aggregate")

    def test_born_digital_high_resolution(self):
        """Born-digital form should have near-100% multi-engine resolution."""
        eo = _born_digital_form()
        matched = match_regions_across_engines(eo)
        rate = compute_resolution_rate(matched)
        assert rate["multi_engine_rate"] >= 0.9

    def test_type_disagreement_resolved(self):
        """Cross-type matching (header/printed_text/page_number) should all match."""
        eo = _type_disagreement()
        matched = match_regions_across_engines(eo)
        rate = compute_resolution_rate(matched)
        assert rate["multi_engine_rate"] == 1.0

    def test_table_isolation_in_mixed(self):
        """Table regions should not cross-match with printed_text."""
        eo = _mixed_content()
        matched = match_regions_across_engines(eo)
        for group in matched:
            if len(group["regions"]) > 1:
                types = {r.element_type for r in group["regions"].values()}
                # If a group has a table region, ALL regions in that group must be tables
                if "table" in types:
                    assert types == {"table"}, f"Table matched with non-table: {types}"

    def test_generated_controlled_uses_fpdf2(self):
        """D-10: generated_controlled scenario actually generates a PDF via fpdf2."""
        pdf_bytes, known_regions = _make_controlled_pdf()
        assert pdf_bytes[:5] == b"%PDF-", "fpdf2 should produce valid PDF"
        assert len(known_regions) == 3, "Should have 3 known regions"
        # Verify the EngineOutput fixtures derived from it match well
        eo = _generated_controlled()
        matched = match_regions_across_engines(eo)
        rate = compute_resolution_rate(matched)
        assert rate["multi_engine_rate"] >= 0.9, "Controlled layout should have high match rate"


# ---------------------------------------------------------------------------
# Real PDF slicing tests (gated by @requires_samples)
# ---------------------------------------------------------------------------


@requires_samples
class TestRealPDFSlicing:
    """Verify real PDF test infrastructure works (slicing + pdfplumber extraction)."""

    PDF_SLICES = [
        ("williams_trust.pdf", 4, 5),     # pages 5-6 (0-indexed: 4-5)
        ("Nixon.pdf", 0, 1),              # pages 1-2
        ("lyndonjohnson.pdf", 0, 1),      # pages 1-2
        ("california-durable-power-of-attorney-form.pdf", 0, 1),  # pages 1-2
        ("ProbateCodeAdvanceHealthCareDirectiveForm-fillable.pdf", 0, 1),  # pages 1-2
    ]

    def test_pdf_slicing_produces_bytes(self):
        """pypdf slicing produces valid PDF bytes for each test PDF."""
        from pypdf import PdfReader, PdfWriter
        import io

        for pdf_name, start, end in self.PDF_SLICES:
            assert SAMPLES_DIR is not None
            pdf_path = SAMPLES_DIR / pdf_name
            if not pdf_path.exists():
                pytest.skip(f"{pdf_name} not found")
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            for i in range(start, min(end + 1, len(reader.pages))):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            sliced = buf.getvalue()
            assert len(sliced) > 100, f"{pdf_name} slice too small"
            assert sliced[:5] == b"%PDF-", f"{pdf_name} slice not valid PDF"

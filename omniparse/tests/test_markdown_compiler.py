"""Tests for the Markdown compilation module -- ConsensusResult to GFM Markdown."""
import re
from datetime import datetime, timezone

import pytest
import yaml

from omniparse.models.consensus import AlignedRegion, ConsensusResult
from omniparse.markdown_compiler import (
    compile_document,
    compile_page,
    region_to_gfm_table,
    region_to_markdown,
)


def _region(**kwargs) -> AlignedRegion:
    """Helper to build AlignedRegion with sane defaults."""
    defaults = dict(
        region_id="r_001",
        element_type="printed_text",
        bounding_box=[100.0, 200.0, 300.0, 400.0],
        engine_texts={"docling": "text"},
        consensus_text="text",
        confidence=0.95,
        source="voting",
        needs_arbitration=False,
        hitl_flag=False,
        metadata=None,
    )
    defaults.update(kwargs)
    return AlignedRegion(**defaults)


class TestRegionToMarkdownHeaders:
    """Header rendering with hierarchy_level support."""

    def test_header_h1(self):
        r = _region(element_type="header", metadata={"hierarchy_level": 1}, consensus_text="Title")
        result = region_to_markdown(r, page_num=0)
        assert result == "# Title"

    def test_header_h2(self):
        r = _region(element_type="header", metadata={"hierarchy_level": 2}, consensus_text="Subtitle")
        result = region_to_markdown(r, page_num=0)
        assert result == "## Subtitle"

    def test_header_h3_through_h6(self):
        for level in range(3, 7):
            r = _region(element_type="header", metadata={"hierarchy_level": level}, consensus_text="Heading")
            result = region_to_markdown(r, page_num=0)
            expected = "#" * level + " Heading"
            assert result == expected, f"Failed for H{level}"

    def test_header_no_level_defaults_h2(self):
        r = _region(element_type="header", metadata={}, consensus_text="Text")
        result = region_to_markdown(r, page_num=0)
        assert result == "## Text"

    def test_header_no_metadata_defaults_h2(self):
        r = _region(element_type="header", metadata=None, consensus_text="Text")
        result = region_to_markdown(r, page_num=0)
        assert result == "## Text"

    def test_header_level_clamped_to_6(self):
        r = _region(element_type="header", metadata={"hierarchy_level": 10}, consensus_text="Deep")
        result = region_to_markdown(r, page_num=0)
        assert result == "###### Deep"


class TestRegionToMarkdownText:
    """Plain text paragraph and skipped element types."""

    def test_printed_text_paragraph(self):
        r = _region(element_type="printed_text", consensus_text="Hello world.")
        result = region_to_markdown(r, page_num=0)
        assert result == "Hello world."

    def test_footer_skipped(self):
        r = _region(element_type="footer", consensus_text="Page 1 of 5")
        result = region_to_markdown(r, page_num=0)
        assert result == ""

    def test_page_number_skipped(self):
        r = _region(element_type="page_number", consensus_text="42")
        result = region_to_markdown(r, page_num=0)
        assert result == ""


class TestRegionToGfmTable:
    """Table rendering: Docling markdown, PaddleOCR HTML, simple text."""

    def test_table_docling_markdown(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        r = _region(element_type="table", metadata={"table_markdown": md})
        result = region_to_gfm_table(r)
        assert result == md

    def test_table_paddleocr_html_complex(self):
        html = '<table><tr><td colspan="2">Merged</td></tr><tr><td>A</td><td>B</td></tr></table>'
        r = _region(element_type="table", metadata={"table_html": html})
        result = region_to_gfm_table(r)
        assert result == html

    def test_table_paddleocr_html_simple_no_colspan(self):
        """HTML without colspan/rowspan should NOT be returned as HTML passthrough."""
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        r = _region(element_type="table", metadata={"table_html": html}, consensus_text="A\tB")
        result = region_to_gfm_table(r)
        # Should attempt pipe-table from consensus_text, not HTML passthrough
        assert "<table>" not in result

    def test_table_simple_text(self):
        r = _region(element_type="table", consensus_text="Name\tAge\nAlice\t30\nBob\t25", metadata={})
        result = region_to_gfm_table(r)
        assert "| Name | Age |" in result
        assert "| --- | --- |" in result or "|---|---|" in result.replace(" ", "")
        assert "| Alice | 30 |" in result
        assert "| Bob | 25 |" in result

    def test_table_empty(self):
        r = _region(element_type="table", consensus_text=None, metadata={})
        result = region_to_gfm_table(r)
        assert result == ""


class TestHitlFlags:
    """HITL review flag rendering as HTML comments."""

    def test_hitl_flag_format(self):
        r = _region(
            hitl_flag=True,
            bounding_box=[100.0, 200.0, 300.0, 400.0],
            confidence=0.45,
            consensus_text="some text",
        )
        result = region_to_markdown(r, page_num=2)
        assert "<!-- REVIEW NEEDED: [100.0,200.0,300.0,400.0] page=2 confidence=0.45 -->" in result

    def test_hitl_flag_with_text(self):
        r = _region(hitl_flag=True, consensus_text="uncertain text", confidence=0.40)
        result = region_to_markdown(r, page_num=1)
        assert "uncertain text" in result
        assert "<!-- REVIEW NEEDED:" in result

    def test_hitl_flag_without_text(self):
        r = _region(hitl_flag=True, consensus_text=None, confidence=0.30)
        result = region_to_markdown(r, page_num=0)
        assert "<!-- REVIEW NEEDED:" in result
        # Should be ONLY the comment (no other text)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("<!-- REVIEW NEEDED:")


class TestCompilePage:
    """compile_page renders regions in reading_order."""

    def test_compile_page_reading_order(self):
        r1 = _region(region_id="r_001", element_type="header", metadata={"hierarchy_level": 1}, consensus_text="Title")
        r2 = _region(region_id="r_002", element_type="printed_text", consensus_text="Paragraph one.")
        r3 = _region(region_id="r_003", element_type="printed_text", consensus_text="Paragraph two.")
        page = ConsensusResult(
            page=0,
            regions=[r1, r2, r3],
            reading_order=["r_003", "r_001", "r_002"],
        )
        result = compile_page(page)
        # r_003 should come first, then r_001, then r_002
        pos_003 = result.index("Paragraph two.")
        pos_001 = result.index("# Title")
        pos_002 = result.index("Paragraph one.")
        assert pos_003 < pos_001 < pos_002

    def test_compile_page_skips_missing_regions(self):
        r1 = _region(region_id="r_001", consensus_text="Present")
        page = ConsensusResult(
            page=0,
            regions=[r1],
            reading_order=["r_999", "r_001"],  # r_999 does not exist
        )
        result = compile_page(page)
        assert "Present" in result
        # No error, graceful skip


class TestCompileDocument:
    """compile_document: YAML frontmatter, page breaks, single-page handling."""

    def test_compile_document_frontmatter(self):
        r1 = _region(region_id="r_001", element_type="header", metadata={"hierarchy_level": 1}, consensus_text="My Doc Title")
        r2 = _region(region_id="r_002", element_type="printed_text", consensus_text="Page 1 text.")
        page1 = ConsensusResult(page=0, regions=[r1, r2], reading_order=["r_001", "r_002"])
        r3 = _region(region_id="r_003", element_type="printed_text", consensus_text="Page 2 text.")
        page2 = ConsensusResult(page=1, regions=[r3], reading_order=["r_003"])

        result = compile_document([page1, page2])
        # Parse YAML frontmatter
        assert result.startswith("---\n")
        fm_end = result.index("---\n", 4)
        fm_text = result[4:fm_end]
        fm = yaml.safe_load(fm_text)
        assert fm["title"] == "My Doc Title"
        assert fm["pages"] == 2
        assert "processed" in fm

    def test_compile_document_page_breaks(self):
        r1 = _region(region_id="r_001", consensus_text="Page 1")
        r2 = _region(region_id="r_002", consensus_text="Page 2")
        page1 = ConsensusResult(page=0, regions=[r1], reading_order=["r_001"])
        page2 = ConsensusResult(page=1, regions=[r2], reading_order=["r_002"])
        result = compile_document([page1, page2])
        # Should contain horizontal rule page break
        assert "\n\n---\n\n" in result

    def test_compile_document_single_page(self):
        r1 = _region(region_id="r_001", consensus_text="Only page")
        page1 = ConsensusResult(page=0, regions=[r1], reading_order=["r_001"])
        result = compile_document([page1])
        # Content after frontmatter should NOT have page break
        # Split off frontmatter
        parts = result.split("---\n", 2)
        body = parts[2] if len(parts) >= 3 else parts[-1]
        assert "\n---\n" not in body

    def test_compile_document_title_override(self):
        r1 = _region(region_id="r_001", element_type="header", metadata={"hierarchy_level": 1}, consensus_text="Auto Title")
        page1 = ConsensusResult(page=0, regions=[r1], reading_order=["r_001"])
        result = compile_document([page1], title="Override Title")
        fm_end = result.index("---\n", 4)
        fm = yaml.safe_load(result[4:fm_end])
        assert fm["title"] == "Override Title"

    def test_compile_document_no_header_untitled(self):
        r1 = _region(region_id="r_001", element_type="printed_text", consensus_text="Just text")
        page1 = ConsensusResult(page=0, regions=[r1], reading_order=["r_001"])
        result = compile_document([page1])
        fm_end = result.index("---\n", 4)
        fm = yaml.safe_load(result[4:fm_end])
        assert fm["title"] == "Untitled Document"


class TestTableTedsProxy:
    """ACCY-02 proxy: table structure preserved through compilation."""

    def test_table_teds_structure_preserved(self):
        md = "| H1 | H2 | H3 |\n|---|---|---|\n| a | b | c |\n| d | e | f |"
        r = _region(element_type="table", metadata={"table_markdown": md})
        result = region_to_gfm_table(r)
        # Verify structure: 3 columns, header + 2 data rows
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 4  # header + separator + 2 data rows
        for line in lines:
            # Each line should have 3 pipe-delimited cells (4 pipes)
            assert line.count("|") >= 4


# ---------------------------------------------------------------------------
# Plan 04-01: MKDN-03 -- Formula rendering
# ---------------------------------------------------------------------------


class TestFormulaRendering:
    """MKDN-03: formula regions render with LaTeX delimiters."""

    def test_formula_display_mode(self):
        """Multiline LaTeX renders as $$...$$."""
        r = _region(
            element_type="formula",
            consensus_text="\\frac{a}{b}\n+ c",
        )
        result = region_to_markdown(r, page_num=0)
        assert result.startswith("$$")
        assert result.endswith("$$")
        assert "\\frac{a}{b}" in result

    def test_formula_inline_mode(self):
        """Short LaTeX renders as $...$."""
        r = _region(
            element_type="formula",
            consensus_text="E = mc^2",
        )
        result = region_to_markdown(r, page_num=0)
        assert result.startswith("$")
        assert result.endswith("$")
        assert not result.startswith("$$")
        assert "E = mc^2" in result

    def test_formula_long_inline_to_display(self):
        """LaTeX > 80 chars renders as display mode $$...$$."""
        long_latex = "a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u"
        assert len(long_latex) > 80
        r = _region(
            element_type="formula",
            consensus_text=long_latex,
        )
        result = region_to_markdown(r, page_num=0)
        assert result.startswith("$$")
        assert result.endswith("$$")

    def test_formula_empty_unreadable(self):
        """Empty formula text renders as placeholder with page reference."""
        r = _region(
            element_type="formula",
            consensus_text="",
        )
        result = region_to_markdown(r, page_num=2)
        assert "[Formula: unreadable, see page 3]" in result

    def test_formula_none_unreadable(self):
        """None formula text renders as placeholder."""
        r = _region(
            element_type="formula",
            consensus_text=None,
        )
        result = region_to_markdown(r, page_num=0)
        assert "[Formula: unreadable, see page 1]" in result


# ---------------------------------------------------------------------------
# Plan 04-01: MKDN-04 -- Handwriting rendering
# ---------------------------------------------------------------------------


class TestHandwritingRendering:
    """MKDN-04: handwriting regions render as italic inline or marginal annotations."""

    def test_inline_handwriting(self):
        """Inline handwriting renders as '*text* <!-- handwritten -->'."""
        r = _region(
            element_type="handwriting",
            consensus_text="Signed by the testator",
            bounding_box=[500.0, 800.0, 1800.0, 870.0],
            metadata={"page_width": 2550.0},
        )
        result = region_to_markdown(r, page_num=0)
        assert "*Signed by the testator*" in result
        assert "<!-- handwritten -->" in result
        assert "handwritten-margin" not in result

    def test_marginal_handwriting(self):
        """Marginal handwriting renders as '*[Margin note: text]* <!-- handwritten-margin [bbox] -->'."""
        r = _region(
            element_type="handwriting",
            consensus_text="See addendum",
            bounding_box=[50.0, 500.0, 350.0, 580.0],
            metadata={"page_width": 2550.0},
        )
        result = region_to_markdown(r, page_num=0)
        assert "*[Margin note: See addendum]*" in result
        assert "<!-- handwritten-margin" in result
        assert "50.0" in result

    def test_classify_handwriting_position_marginal_left(self):
        """x1 < 15% page width -> marginal."""
        from omniparse.markdown_compiler import classify_handwriting_position
        # 15% of 2550 = 382.5; x1=50 < 382.5
        assert classify_handwriting_position([50.0, 500.0, 350.0, 580.0], 2550.0) == "marginal"

    def test_classify_handwriting_position_marginal_right(self):
        """x2 > 85% page width -> marginal."""
        from omniparse.markdown_compiler import classify_handwriting_position
        # 85% of 2550 = 2167.5; x2=2200 > 2167.5
        assert classify_handwriting_position([2100.0, 500.0, 2200.0, 580.0], 2550.0) == "marginal"

    def test_classify_handwriting_position_inline(self):
        """Region in body area -> inline."""
        from omniparse.markdown_compiler import classify_handwriting_position
        assert classify_handwriting_position([500.0, 800.0, 1800.0, 870.0], 2550.0) == "inline"


# ---------------------------------------------------------------------------
# Plan 04-01: Chart SVG rendering
# ---------------------------------------------------------------------------


class TestChartRendering:
    """Chart regions render as inline SVG blocks with XSS sanitization."""

    def test_chart_svg_passthrough(self):
        """Chart with svg_content in metadata renders as raw SVG block."""
        svg = "<svg><rect width='100' height='50'/></svg>"
        r = _region(
            element_type="chart",
            metadata={"svg_content": svg},
        )
        result = region_to_markdown(r, page_num=0)
        assert "<svg>" in result
        assert "<rect" in result

    def test_chart_svg_xss_sanitization(self):
        """Chart SVG with <script> tags has them stripped."""
        svg = "<svg><script>alert('xss')</script><rect width='100' height='50'/></svg>"
        r = _region(
            element_type="chart",
            metadata={"svg_content": svg},
        )
        result = region_to_markdown(r, page_num=0)
        assert "<script>" not in result
        assert "alert" not in result
        assert "<rect" in result

    def test_chart_svg_event_handler_stripped(self):
        """Chart SVG with on* event handlers has them removed."""
        svg = '<svg><rect onclick="alert(1)" width="100" height="50"/></svg>'
        r = _region(
            element_type="chart",
            metadata={"svg_content": svg},
        )
        result = region_to_markdown(r, page_num=0)
        assert "onclick" not in result
        assert "<rect" in result

    def test_chart_no_svg_content_fallback(self):
        """Chart with no svg_content renders as placeholder."""
        r = _region(
            element_type="chart",
            metadata={},
        )
        result = region_to_markdown(r, page_num=3)
        assert "[Chart: extraction failed, see page 4]" in result

    def test_chart_no_metadata_fallback(self):
        """Chart with no metadata at all renders as placeholder."""
        r = _region(
            element_type="chart",
            metadata=None,
        )
        result = region_to_markdown(r, page_num=0)
        assert "[Chart: extraction failed, see page 1]" in result

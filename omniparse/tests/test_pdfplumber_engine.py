"""Tests for pdfplumber text extraction engine.

Covers ENGN-01 (text extraction with bounding boxes, font metadata, tables)
and ENGN-05 (canonical Region/EngineOutput schema conformance).
"""
import re
import pytest
from pydantic import ValidationError
from omniparse.engines.pdfplumber_engine import extract_page
from omniparse.models.region import Region, EngineOutput


class TestTextExtraction:
    """ENGN-01: pdfplumber extracts text from born-digital PDFs."""

    def test_extract_returns_text_regions(self, born_digital_pdf_bytes):
        """Test 1: extract_page on born-digital PDF returns EngineOutput with printed_text regions."""
        result = extract_page(born_digital_pdf_bytes, 0)
        assert isinstance(result, EngineOutput)
        text_regions = [r for r in result.regions if r.element_type == "printed_text"]
        assert len(text_regions) >= 1, "Expected at least one printed_text region"

    def test_bounding_boxes_are_valid(self, born_digital_pdf_bytes):
        """Test 2: Each text region has 4-float bbox with x1 < x2 and y1 < y2."""
        result = extract_page(born_digital_pdf_bytes, 0)
        text_regions = [r for r in result.regions if r.element_type == "printed_text"]
        assert len(text_regions) >= 1

        for region in text_regions:
            bbox = region.bounding_box
            assert len(bbox) == 4, f"Bounding box must have 4 values, got {len(bbox)}"
            assert all(isinstance(v, float) for v in bbox), "All bbox values must be floats"
            x1, y1, x2, y2 = bbox
            assert x1 < x2, f"x1 ({x1}) must be less than x2 ({x2})"
            assert y1 < y2, f"y1 ({y1}) must be less than y2 ({y2})"

    def test_font_metadata_present(self, born_digital_pdf_bytes):
        """Test 3: Each text region has metadata with font_size, bold, italic."""
        result = extract_page(born_digital_pdf_bytes, 0)
        text_regions = [r for r in result.regions if r.element_type == "printed_text"]
        assert len(text_regions) >= 1

        for region in text_regions:
            assert region.metadata is not None, "metadata must not be None"
            assert "font_size" in region.metadata, "metadata must contain font_size"
            assert isinstance(region.metadata["font_size"], float), "font_size must be float"
            assert region.metadata["font_size"] > 0, "font_size must be positive"
            assert "bold" in region.metadata, "metadata must contain bold"
            assert isinstance(region.metadata["bold"], bool), "bold must be bool"
            assert "italic" in region.metadata, "metadata must contain italic"
            assert isinstance(region.metadata["italic"], bool), "italic must be bool"


class TestSchemaConformance:
    """ENGN-05: Output conforms to canonical Region/EngineOutput schema."""

    def test_regions_validate_against_schema(self, born_digital_pdf_bytes):
        """Test 4: Every region validates against the Region Pydantic model."""
        result = extract_page(born_digital_pdf_bytes, 0)
        for region in result.regions:
            # Re-validate by constructing from dict -- must not raise ValidationError
            try:
                Region(**region.model_dump())
            except ValidationError as e:
                pytest.fail(f"Region {region.id} failed validation: {e}")

    def test_confidence_is_one(self, born_digital_pdf_bytes):
        """Test 5: All pdfplumber regions have confidence=1.0."""
        result = extract_page(born_digital_pdf_bytes, 0)
        assert len(result.regions) >= 1

        for region in result.regions:
            assert region.confidence == 1.0, (
                f"Region {region.id} confidence={region.confidence}, expected 1.0"
            )

    def test_engine_name_and_page(self, born_digital_pdf_bytes):
        """Test 6: EngineOutput.engine equals 'pdfplumber' and page matches request."""
        result = extract_page(born_digital_pdf_bytes, 0)
        assert result.engine == "pdfplumber"
        assert result.page == 0

        # Also test with a different page number
        result_p1 = extract_page(born_digital_pdf_bytes, 1)
        assert result_p1.page == 1


class TestTableExtraction:
    """ENGN-01: Table detection and extraction."""

    def test_table_region_detected(self, born_digital_pdf_bytes):
        """Test 7: PDF with table produces at least one table region."""
        result = extract_page(born_digital_pdf_bytes, 0)
        table_regions = [r for r in result.regions if r.element_type == "table"]
        assert len(table_regions) >= 1, "Expected at least one table region"
        for tr in table_regions:
            assert tr.table_structure is not None, "table_structure must not be None"

    def test_table_structure_has_rows_and_cols(self, born_digital_pdf_bytes):
        """Test 8: table_structure dict contains 'rows' (int) and 'cols' (int)."""
        result = extract_page(born_digital_pdf_bytes, 0)
        table_regions = [r for r in result.regions if r.element_type == "table"]
        assert len(table_regions) >= 1

        for tr in table_regions:
            assert "rows" in tr.table_structure, "table_structure must contain 'rows'"
            assert "cols" in tr.table_structure, "table_structure must contain 'cols'"
            assert isinstance(tr.table_structure["rows"], int)
            assert isinstance(tr.table_structure["cols"], int)
            assert tr.table_structure["rows"] > 0
            assert tr.table_structure["cols"] > 0


class TestEdgeCases:
    """Edge cases: blank pages, image inputs, out-of-range pages."""

    def test_blank_page_returns_empty(self, blank_pdf_bytes):
        """Test 9: Blank PDF page returns EngineOutput with regions=[]."""
        result = extract_page(blank_pdf_bytes, 0)
        assert isinstance(result, EngineOutput)
        assert result.regions == []
        assert result.engine == "pdfplumber"
        assert result.page == 0

    def test_image_input_returns_empty(self):
        """Test 10: Non-PDF input with is_pdf=False returns empty EngineOutput."""
        # Simulate a PNG image (not a PDF)
        fake_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = extract_page(fake_png_bytes, 0, is_pdf=False)
        assert isinstance(result, EngineOutput)
        assert result.regions == []
        assert result.engine == "pdfplumber"

    def test_out_of_range_page_returns_empty(self, born_digital_pdf_bytes):
        """Test 11: Page number beyond PDF length returns empty EngineOutput."""
        result = extract_page(born_digital_pdf_bytes, 999)
        assert isinstance(result, EngineOutput)
        assert result.regions == []
        assert result.page == 999


class TestCoordinateSystem:
    """Coordinate system and region ID conventions."""

    def test_coordinate_system_metadata(self, born_digital_pdf_bytes):
        """Test 12: Region metadata contains coordinate_system='pixel_300dpi_topleft'."""
        result = extract_page(born_digital_pdf_bytes, 0)
        text_regions = [r for r in result.regions if r.element_type == "printed_text"]
        assert len(text_regions) >= 1

        for region in text_regions:
            assert region.metadata is not None
            assert region.metadata.get("coordinate_system") == "pixel_300dpi_topleft", (
                f"Expected coordinate_system='pixel_300dpi_topleft', got "
                f"'{region.metadata.get('coordinate_system')}'"
            )

    def test_region_ids_unique_and_formatted(self, born_digital_pdf_bytes):
        """Test 13: Each region has a unique id matching pattern r_NNN."""
        result = extract_page(born_digital_pdf_bytes, 0)
        assert len(result.regions) >= 1

        ids = set()
        pattern = re.compile(r"^r_\d{3}$")
        for region in result.regions:
            assert pattern.match(region.id), (
                f"Region id '{region.id}' does not match pattern 'r_NNN'"
            )
            assert region.id not in ids, f"Duplicate region id: {region.id}"
            ids.add(region.id)

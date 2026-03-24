"""Tests for canonical Region, EngineOutput, and PagePayload Pydantic models."""
import json
import pytest
from pydantic import ValidationError
from omniparse.models.region import Region, EngineOutput, VALID_ELEMENT_TYPES
from omniparse.models.page import PagePayload


class TestRegion:
    """Test Region model validation and serialization."""

    def test_region_accepts_all_required_fields(self, sample_region):
        """Test 1: Region model accepts all required fields and serializes correctly."""
        data = sample_region.model_dump()
        assert data["id"] == "r_001"
        assert data["element_type"] == "printed_text"
        assert data["bounding_box"] == [72.0, 100.5, 540.0, 112.3]
        assert data["confidence"] == 1.0
        assert data["text_content"] == "LAST WILL AND TESTAMENT"

    def test_region_rejects_confidence_below_zero(self):
        """Test 2a: Region rejects confidence < 0.0."""
        with pytest.raises(ValidationError):
            Region(
                id="r_bad",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 100.0, 100.0],
                confidence=-0.1,
                text_content="test",
            )

    def test_region_rejects_confidence_above_one(self):
        """Test 2b: Region rejects confidence > 1.0."""
        with pytest.raises(ValidationError):
            Region(
                id="r_bad",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 100.0, 100.0],
                confidence=1.1,
                text_content="test",
            )

    def test_region_optional_fields_default_none(self):
        """Test 3: Region optional fields default to None."""
        region = Region(
            id="r_002",
            element_type="table",
            bounding_box=[10.0, 20.0, 300.0, 400.0],
            confidence=0.95,
            text_content="<table>...</table>",
        )
        assert region.table_structure is None
        assert region.metadata is None

    def test_region_with_optional_fields_provided(self):
        """Test 3b: Region includes optional fields when provided."""
        table_meta = {"rows": 5, "cols": 3, "has_merged_cells": False}
        region = Region(
            id="r_003",
            element_type="table",
            bounding_box=[10.0, 20.0, 300.0, 400.0],
            confidence=0.92,
            text_content="<table>...</table>",
            table_structure=table_meta,
            metadata={"engine_version": "1.0"},
        )
        assert region.table_structure == table_meta
        assert region.metadata == {"engine_version": "1.0"}

    def test_region_bounding_box_exactly_four_floats(self):
        """Test 7: bounding_box must have exactly 4 elements."""
        with pytest.raises(ValidationError):
            Region(
                id="r_short",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 100.0],  # Only 3 elements
                confidence=1.0,
                text_content="test",
            )
        with pytest.raises(ValidationError):
            Region(
                id="r_long",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 100.0, 100.0, 50.0],  # 5 elements
                confidence=1.0,
                text_content="test",
            )

    def test_bounding_box_norm_defaults_none(self):
        """Region without bounding_box_norm has it as None."""
        region = Region(
            id="r_no_norm",
            element_type="printed_text",
            bounding_box=[0.0, 0.0, 100.0, 100.0],
            confidence=1.0,
            text_content="test",
        )
        assert region.bounding_box_norm is None

    def test_bounding_box_norm_accepts_valid(self):
        """Region with bounding_box_norm=[0.1, 0.2, 0.3, 0.4] is valid."""
        region = Region(
            id="r_norm",
            element_type="printed_text",
            bounding_box=[100.0, 200.0, 300.0, 400.0],
            confidence=1.0,
            text_content="test",
            bounding_box_norm=[0.1, 0.2, 0.3, 0.4],
        )
        assert region.bounding_box_norm == [0.1, 0.2, 0.3, 0.4]

    def test_bounding_box_norm_rejects_wrong_length(self):
        """bounding_box_norm with != 4 elements raises ValidationError."""
        with pytest.raises(ValidationError):
            Region(
                id="r_bad_norm",
                element_type="printed_text",
                bounding_box=[0.0, 0.0, 100.0, 100.0],
                confidence=1.0,
                text_content="test",
                bounding_box_norm=[0.1, 0.2],
            )

    def test_bounding_box_norm_coexists_with_pixel(self):
        """Region can have both bounding_box and bounding_box_norm set simultaneously."""
        region = Region(
            id="r_both",
            element_type="printed_text",
            bounding_box=[100.0, 200.0, 500.0, 600.0],
            confidence=1.0,
            text_content="test",
            bounding_box_norm=[0.0392, 0.0606, 0.1961, 0.1818],
        )
        assert region.bounding_box == [100.0, 200.0, 500.0, 600.0]
        assert region.bounding_box_norm == [0.0392, 0.0606, 0.1961, 0.1818]

    def test_region_all_valid_element_types(self):
        """Test 8: All 10 valid element types are accepted."""
        expected_types = {
            "printed_text", "table", "handwriting", "formula",
            "chart", "image", "header", "footer", "page_number", "seal",
        }
        assert VALID_ELEMENT_TYPES == expected_types
        for etype in expected_types:
            region = Region(
                id=f"r_{etype}",
                element_type=etype,
                bounding_box=[0.0, 0.0, 100.0, 100.0],
                confidence=0.5,
                text_content=f"content for {etype}",
            )
            assert region.element_type == etype


class TestEngineOutput:
    """Test EngineOutput model."""

    def test_engine_output_holds_regions(self, sample_region):
        """Test 4: EngineOutput holds page number, engine name, and list of Regions."""
        output = EngineOutput(
            page=0,
            engine="pdfplumber",
            regions=[sample_region],
        )
        assert output.page == 0
        assert output.engine == "pdfplumber"
        assert len(output.regions) == 1
        assert output.regions[0].id == "r_001"

    def test_engine_output_roundtrip_json(self, sample_engine_output):
        """Test 5: EngineOutput serializes to JSON and deserializes with all regions preserved."""
        json_str = sample_engine_output.model_dump_json()
        restored = EngineOutput.model_validate_json(json_str)
        assert restored.page == sample_engine_output.page
        assert restored.engine == sample_engine_output.engine
        assert len(restored.regions) == len(sample_engine_output.regions)
        assert restored.regions[0].id == sample_engine_output.regions[0].id
        assert restored.regions[0].text_content == sample_engine_output.regions[0].text_content
        assert restored.regions[0].bounding_box == sample_engine_output.regions[0].bounding_box

    def test_engine_output_empty_regions(self):
        """EngineOutput with empty regions list is valid."""
        output = EngineOutput(page=1, engine="docling", regions=[])
        assert output.regions == []


class TestPagePayload:
    """Test PagePayload model."""

    def test_page_payload_all_fields(self, sample_page_payload):
        """Test 6: PagePayload holds all expected fields."""
        assert sample_page_payload.page_num == 0
        assert sample_page_payload.image_bytes == b"fake_png_bytes"
        assert sample_page_payload.pdf_bytes == b"fake_pdf_bytes"
        assert sample_page_payload.dpi == 300
        assert sample_page_payload.width == 2550
        assert sample_page_payload.height == 3300
        assert sample_page_payload.was_rotated is False
        assert sample_page_payload.error is None

    def test_page_payload_optional_fields(self):
        """PagePayload with pdf_bytes=None and error=None."""
        payload = PagePayload(
            page_num=1,
            image_bytes=b"png_data",
            dpi=200,
            width=1700,
            height=2200,
        )
        assert payload.pdf_bytes is None
        assert payload.error is None
        assert payload.was_rotated is False

    def test_page_payload_with_error(self):
        """PagePayload with error string."""
        payload = PagePayload(
            page_num=2,
            image_bytes=b"png_data",
            dpi=300,
            width=2550,
            height=3300,
            error="DPI below minimum, resampled from 72 to 200",
        )
        assert payload.error == "DPI below minimum, resampled from 72 to 200"

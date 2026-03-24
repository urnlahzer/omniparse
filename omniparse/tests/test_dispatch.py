"""Tests for the smart dispatch module -- routing specialist regions to engines.

Covers requirements: SPEC-01 (specialist dispatch), SPEC-04 (handwriting routing).
"""
import pytest

from omniparse.models.region import Region, EngineOutput
from omniparse.dispatch import classify_dispatch, HANDWRITING_TYPES, FORMULA_TYPES, CHART_TYPES


class TestClassifyDispatch:
    """classify_dispatch routes regions by element_type to specialist engines."""

    def test_handwriting_routed_to_trocr(self, handwriting_region):
        """PaddleOCR-classified handwriting region -> trocr dispatch."""
        output = EngineOutput(
            page=0,
            engine="paddleocr",
            regions=[handwriting_region],
        )
        result = classify_dispatch(output)
        assert len(result["trocr"]) == 1
        assert result["trocr"][0].id == handwriting_region.id
        assert result["dots_formula"] == []
        assert result["dots_chart"] == []

    def test_formula_routed_to_dots_formula(self, formula_region):
        """PaddleOCR-classified formula region -> dots_formula dispatch."""
        output = EngineOutput(
            page=0,
            engine="paddleocr",
            regions=[formula_region],
        )
        result = classify_dispatch(output)
        assert len(result["dots_formula"]) == 1
        assert result["dots_formula"][0].id == formula_region.id
        assert result["trocr"] == []
        assert result["dots_chart"] == []

    def test_chart_routed_to_dots_chart(self, chart_region):
        """PaddleOCR-classified chart region -> dots_chart dispatch."""
        output = EngineOutput(
            page=0,
            engine="paddleocr",
            regions=[chart_region],
        )
        result = classify_dispatch(output)
        assert len(result["dots_chart"]) == 1
        assert result["dots_chart"][0].id == chart_region.id
        assert result["trocr"] == []
        assert result["dots_formula"] == []

    def test_mixed_regions_routed_correctly(self, mixed_paddleocr_output):
        """Mixed regions (handwriting + formula + chart + printed_text) -> correct routing."""
        result = classify_dispatch(mixed_paddleocr_output)
        # Handwriting -> trocr
        assert len(result["trocr"]) == 1
        assert result["trocr"][0].element_type == "handwriting"
        # Formula -> dots_formula
        assert len(result["dots_formula"]) == 1
        assert result["dots_formula"][0].element_type == "formula"
        # Chart -> dots_chart
        assert len(result["dots_chart"]) == 1
        assert result["dots_chart"][0].element_type == "chart"
        # printed_text should NOT be in any dispatch list
        all_dispatched = result["trocr"] + result["dots_formula"] + result["dots_chart"]
        assert all(r.element_type != "printed_text" for r in all_dispatched)

    def test_no_specialist_regions_empty_dispatch(self):
        """No specialist regions -> all dispatch lists empty."""
        output = EngineOutput(
            page=0,
            engine="paddleocr",
            regions=[
                Region(
                    id="r_001",
                    element_type="printed_text",
                    bounding_box=[100.0, 100.0, 500.0, 200.0],
                    confidence=0.95,
                    text_content="Normal text",
                ),
                Region(
                    id="r_002",
                    element_type="table",
                    bounding_box=[100.0, 300.0, 500.0, 600.0],
                    confidence=0.90,
                    text_content="Table content",
                ),
            ],
        )
        result = classify_dispatch(output)
        assert result["trocr"] == []
        assert result["dots_formula"] == []
        assert result["dots_chart"] == []

    def test_multiple_handwriting_regions_all_dispatched(self):
        """Multiple handwriting regions -> all in trocr list."""
        regions = [
            Region(
                id=f"r_{i:03d}",
                element_type="handwriting",
                bounding_box=[100.0, float(100 + i * 50), 500.0, float(150 + i * 50)],
                confidence=0.80,
                text_content=f"Handwritten text {i}",
            )
            for i in range(3)
        ]
        output = EngineOutput(page=0, engine="paddleocr", regions=regions)
        result = classify_dispatch(output)
        assert len(result["trocr"]) == 3
        assert result["dots_formula"] == []
        assert result["dots_chart"] == []


class TestDispatchConstants:
    """Verify dispatch type sets contain expected values."""

    def test_handwriting_types(self):
        assert "handwriting" in HANDWRITING_TYPES

    def test_formula_types(self):
        assert "formula" in FORMULA_TYPES

    def test_chart_types(self):
        assert "chart" in CHART_TYPES

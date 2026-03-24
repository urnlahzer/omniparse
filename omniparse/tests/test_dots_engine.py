"""Tests for Dots.ocr engine -- formula LaTeX extraction, chart SVG extraction, and batch dispatch.

Covers requirements: SPEC-05, SPEC-06.
"""
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock vLLM before importing the engine (GPU-only, not in local venv)
# Same sys.modules mock pattern as test_llm_arbiter.py.
# ---------------------------------------------------------------------------

_mock_vllm = MagicMock()
_mock_vllm.SamplingParams = MagicMock
sys.modules.setdefault("vllm", _mock_vllm)


# ---------------------------------------------------------------------------
# MockLLM that returns different responses based on prompt content
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock vLLM LLM that returns LaTeX for formula prompts and SVG for SVG prompts."""

    def __init__(self, response_override=None):
        self._response_override = response_override

    def chat(self, messages, sampling_params=None):
        if self._response_override is not None:
            text = self._response_override
        else:
            # Inspect prompt content to decide response
            prompt_text = ""
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_text = item["text"]
                elif isinstance(content, str):
                    prompt_text += content

            if "SVG" in prompt_text or "svg" in prompt_text.lower():
                text = '<svg viewBox="0 0 800 600"><rect/></svg>'
            else:
                text = r"\frac{a}{b} + c = 0"

        return [type("Output", (), {"outputs": [type("Choice", (), {"text": text})()]})]


class EmptyMockLLM(MockLLM):
    """Mock LLM that returns empty text."""

    def __init__(self):
        super().__init__(response_override="")


class PlainTextMockLLM(MockLLM):
    """Mock LLM that returns plain text (not SVG)."""

    def __init__(self):
        super().__init__(response_override="This is just a description of a chart, no SVG here.")


# ---------------------------------------------------------------------------
# Helper: import all public symbols from the module under test
# ---------------------------------------------------------------------------


def _import():
    from omniparse.engines.dots_engine import (
        extract_formula,
        extract_chart_svg,
        validate_latex,
        FORMULA_PROMPT,
        SVG_PROMPT_TEMPLATE,
        DOTS_MODEL_PATH,
        DotsEngine,
    )
    return {
        "extract_formula": extract_formula,
        "extract_chart_svg": extract_chart_svg,
        "validate_latex": validate_latex,
        "FORMULA_PROMPT": FORMULA_PROMPT,
        "SVG_PROMPT_TEMPLATE": SVG_PROMPT_TEMPLATE,
        "DOTS_MODEL_PATH": DOTS_MODEL_PATH,
        "DotsEngine": DotsEngine,
    }


# ---------------------------------------------------------------------------
# Helper: create a small test PNG as bytes
# ---------------------------------------------------------------------------


def _make_test_image_bytes() -> bytes:
    from PIL import Image
    import io
    img = Image.new("RGB", (10, 10), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1. Formula extraction (pure function)
# ===========================================================================


class TestExtractFormula:
    """extract_formula: converts formula region image to LaTeX via vLLM."""

    def test_extract_formula_returns_latex_key(self):
        """extract_formula with mock vLLM returns dict with 'latex' key containing formula text."""
        m = _import()
        result = m["extract_formula"](MockLLM(), _make_test_image_bytes())
        assert "latex" in result
        assert isinstance(result["latex"], str)
        assert len(result["latex"]) > 0

    def test_extract_formula_success_true(self):
        """extract_formula with successful output has 'success': True."""
        m = _import()
        result = m["extract_formula"](MockLLM(), _make_test_image_bytes())
        assert result["success"] is True

    def test_extract_formula_empty_output_fails(self):
        """extract_formula with empty vLLM output returns 'success': False."""
        m = _import()
        result = m["extract_formula"](EmptyMockLLM(), _make_test_image_bytes())
        assert result["success"] is False

    def test_extract_formula_has_confidence(self):
        """extract_formula returns confidence field."""
        m = _import()
        result = m["extract_formula"](MockLLM(), _make_test_image_bytes())
        assert "confidence" in result
        assert result["confidence"] > 0.0


# ===========================================================================
# 2. Chart SVG extraction (pure function)
# ===========================================================================


class TestExtractChartSvg:
    """extract_chart_svg: converts chart region image to SVG via vLLM."""

    def test_extract_chart_svg_returns_svg_key(self):
        """extract_chart_svg with mock vLLM returns dict with 'svg' key containing SVG markup."""
        m = _import()
        result = m["extract_chart_svg"](MockLLM(), _make_test_image_bytes())
        assert "svg" in result
        assert "<svg" in result["svg"].lower()

    def test_extract_chart_svg_success_true(self):
        """extract_chart_svg with successful output has 'success': True (contains '<svg')."""
        m = _import()
        result = m["extract_chart_svg"](MockLLM(), _make_test_image_bytes())
        assert result["success"] is True

    def test_extract_chart_svg_no_svg_output_fails(self):
        """extract_chart_svg with non-SVG output returns 'success': False."""
        m = _import()
        result = m["extract_chart_svg"](PlainTextMockLLM(), _make_test_image_bytes())
        assert result["success"] is False

    def test_extract_chart_svg_passes_dimensions_to_prompt(self):
        """extract_chart_svg passes width/height to prompt viewBox."""
        m = _import()
        # We verify by checking that the function accepts width/height params
        # and still returns successfully with custom dimensions
        result = m["extract_chart_svg"](MockLLM(), _make_test_image_bytes(), width=1024, height=768)
        assert result["success"] is True

    def test_extract_chart_svg_has_confidence(self):
        """extract_chart_svg returns confidence field."""
        m = _import()
        result = m["extract_chart_svg"](MockLLM(), _make_test_image_bytes())
        assert "confidence" in result
        assert result["confidence"] > 0.0


# ===========================================================================
# 3. LaTeX validation
# ===========================================================================


class TestValidateLatex:
    """validate_latex: checks balanced delimiters and non-empty content."""

    def test_validate_latex_balanced_double_dollar(self):
        """Balanced $$ pairs are valid."""
        m = _import()
        assert m["validate_latex"](r"$$\frac{a}{b}$$") is True

    def test_validate_latex_unbalanced_double_dollar(self):
        """Unbalanced $$ (odd count) is invalid."""
        m = _import()
        assert m["validate_latex"](r"$$\frac{a}{b}") is False

    def test_validate_latex_balanced_single_dollar(self):
        """Balanced single $ pairs are valid."""
        m = _import()
        assert m["validate_latex"](r"$x + y = z$") is True

    def test_validate_latex_empty_string_invalid(self):
        """Empty string is invalid."""
        m = _import()
        assert m["validate_latex"]("") is False

    def test_validate_latex_whitespace_only_invalid(self):
        """Whitespace-only string is invalid."""
        m = _import()
        assert m["validate_latex"]("   ") is False

    def test_validate_latex_no_delimiters_valid(self):
        """LaTeX without dollar delimiters is valid (plain formula text)."""
        m = _import()
        assert m["validate_latex"](r"\frac{a}{b} + c = 0") is True


# ===========================================================================
# 4. Engine integration (DotsEngine class)
# ===========================================================================


class TestDotsEngineIntegration:
    """DotsEngine Modal class delegates to pure functions.

    GPU engine stubs are tested via source inspection (Modal decorators prevent
    local calling) -- same pattern as test_app.py and test_llm_arbiter.py.
    """

    def test_dots_engine_run_batch_processes_multiple_regions(self):
        """run_batch logic processes multiple regions with correct dispatch."""
        m = _import()
        # Test the batch logic inline (same pure functions run_batch delegates to)
        image_bytes = _make_test_image_bytes()
        llm = MockLLM()

        regions = [
            {"image_bytes": image_bytes, "task": "formula", "region_id": "r_001"},
            {"image_bytes": image_bytes, "task": "chart", "region_id": "r_002", "width": 800, "height": 600},
            {"image_bytes": image_bytes, "task": "formula", "region_id": "r_003"},
        ]

        # Replicate run_batch logic (pure function calls)
        results = []
        for item in regions:
            task = item.get("task", "formula")
            img = item["image_bytes"]
            if task == "chart":
                result = m["extract_chart_svg"](llm, img, width=item.get("width", 800), height=item.get("height", 600))
            else:
                result = m["extract_formula"](llm, img)
            result["region_id"] = item.get("region_id", "unknown")
            results.append(result)

        assert len(results) == 3
        assert results[0]["region_id"] == "r_001"
        assert "latex" in results[0]
        assert results[1]["region_id"] == "r_002"
        assert "svg" in results[1]
        assert results[2]["region_id"] == "r_003"

    def test_dots_engine_run_formula_source_delegates(self):
        """DotsEngine.run_formula source delegates to extract_formula."""
        import pathlib
        source = pathlib.Path("omniparse/engines/dots_engine.py").read_text()
        assert "def run_formula" in source
        assert "extract_formula(self.llm" in source

    def test_dots_engine_run_chart_source_delegates(self):
        """DotsEngine.run_chart source delegates to extract_chart_svg."""
        import pathlib
        source = pathlib.Path("omniparse/engines/dots_engine.py").read_text()
        assert "def run_chart" in source
        assert "extract_chart_svg(self.llm" in source

    def test_dots_engine_has_map_compatible_batch(self):
        """DotsEngine.run_batch method exists and uses modal.method decorator."""
        import pathlib
        source = pathlib.Path("omniparse/engines/dots_engine.py").read_text()
        assert "def run_batch" in source
        assert "modal.method" in source
        assert "region_id" in source


# ===========================================================================
# 5. Constants
# ===========================================================================


class TestDotsConstants:
    """Verify prompts and model path constants."""

    def test_formula_prompt_defined(self):
        m = _import()
        assert "Extract" in m["FORMULA_PROMPT"] or "extract" in m["FORMULA_PROMPT"].lower()

    def test_svg_prompt_template_has_viewbox(self):
        m = _import()
        assert "viewBox" in m["SVG_PROMPT_TEMPLATE"] or "viewbox" in m["SVG_PROMPT_TEMPLATE"].lower()

    def test_dots_model_path_no_periods(self):
        """Model path uses underscores, not periods (research pitfall #4)."""
        m = _import()
        assert "." not in m["DOTS_MODEL_PATH"]
        assert "DotsOCR" in m["DOTS_MODEL_PATH"]

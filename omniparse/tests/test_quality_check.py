"""Tests for text quality check between pdfplumber and PaddleOCR."""
from omniparse.models.region import Region, EngineOutput
from omniparse.quality_check import check_text_quality


def _make_output(engine: str, texts: list[str], page: int = 0) -> EngineOutput:
    """Helper to create EngineOutput with text regions."""
    regions = [
        Region(
            id=f"r_{i:03d}",
            element_type="printed_text",
            bounding_box=[0.0, float(i * 50), 500.0, float(i * 50 + 40)],
            confidence=1.0,
            text_content=text,
        )
        for i, text in enumerate(texts)
    ]
    return EngineOutput(page=page, engine=engine, regions=regions)


class TestCheckTextQuality:
    """Test text quality check logic."""

    def test_identical_text_is_ground_truth(self):
        """Both engines return identical text -> ground truth."""
        plumber = _make_output("pdfplumber", ["LAST WILL AND TESTAMENT"])
        paddle = _make_output("paddleocr", ["LAST WILL AND TESTAMENT"])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is True
        assert result["similarity"] == 1.0
        assert result["reason"] == "agreement"

    def test_completely_different_text(self):
        """Completely different text -> not ground truth."""
        plumber = _make_output("pdfplumber", ["hello world"])
        paddle = _make_output("paddleocr", ["xyzzy abcde"])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is False
        assert result["reason"] == "disagreement"

    def test_high_similarity_is_ground_truth(self):
        """Minor OCR error still passes >90% threshold."""
        plumber = _make_output("pdfplumber", ["LAST WILL AND TESTAMENT"])
        paddle = _make_output("paddleocr", ["LAST WILL AND TESTAMANT"])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is True
        assert result["similarity"] > 0.90

    def test_low_similarity_not_ground_truth(self):
        """Very different text -> not ground truth."""
        plumber = _make_output("pdfplumber", ["alpha beta gamma"])
        paddle = _make_output("paddleocr", ["delta epsilon zeta"])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is False
        assert result["similarity"] < 0.90

    def test_empty_pdfplumber_text(self):
        """pdfplumber has no text regions -> no_pdfplumber_text."""
        plumber = EngineOutput(page=0, engine="pdfplumber", regions=[])
        paddle = _make_output("paddleocr", ["some text"])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is False
        assert result["similarity"] == 0.0
        assert result["reason"] == "no_pdfplumber_text"

    def test_empty_paddleocr_text(self):
        """pdfplumber has text but paddleocr empty -> disagreement."""
        plumber = _make_output("pdfplumber", ["hello world"])
        paddle = EngineOutput(page=0, engine="paddleocr", regions=[])
        result = check_text_quality(plumber, paddle)
        assert result["is_ground_truth"] is False
        assert result["similarity"] < 0.1

    def test_custom_threshold(self):
        """Custom threshold=0.50 with moderately similar text."""
        plumber = _make_output("pdfplumber", ["alpha beta gamma delta"])
        paddle = _make_output("paddleocr", ["alpha beta xxxxx delta"])
        result = check_text_quality(plumber, paddle, threshold=0.50)
        assert result["is_ground_truth"] is True
        assert result["similarity"] > 0.50

    def test_whitespace_only_pdfplumber(self):
        """Regions with only whitespace -> no_pdfplumber_text."""
        plumber = _make_output("pdfplumber", ["   ", "\t", "\n"])
        paddle = _make_output("paddleocr", ["some text"])
        result = check_text_quality(plumber, paddle)
        assert result["reason"] == "no_pdfplumber_text"

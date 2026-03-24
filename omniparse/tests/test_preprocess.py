"""Tests for the preprocessing pipeline (PREP-01 through PREP-05).

Every test calls preprocess(file_bytes, filename) and validates the returned
list[PagePayload] objects.
"""
import io

import numpy as np
import pytest
from PIL import Image

from omniparse.models.page import PagePayload
from omniparse.preprocess import preprocess


# ---------------------------------------------------------------------------
# PREP-01: Format acceptance
# ---------------------------------------------------------------------------

class TestFormatAcceptance:
    """preprocess() accepts PDF, PNG, JPG, JPEG, TIFF and rejects others."""

    def test_pdf_returns_page_payloads(self, single_page_pdf_bytes):
        """PREP-01: PDF input returns list[PagePayload]."""
        result = preprocess(single_page_pdf_bytes, "document.pdf")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(p, PagePayload) for p in result)

    def test_png_returns_single_page(self, portrait_png_300dpi):
        """PREP-01: PNG input returns list with exactly 1 PagePayload."""
        result = preprocess(portrait_png_300dpi, "page.png")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], PagePayload)

    def test_jpg_returns_single_page(self, portrait_jpg_bytes):
        """PREP-01: JPG input returns list with exactly 1 PagePayload."""
        result = preprocess(portrait_jpg_bytes, "photo.jpg")
        assert len(result) == 1
        assert isinstance(result[0], PagePayload)

    def test_tiff_returns_single_page(self, portrait_tiff_bytes):
        """PREP-01: TIFF input returns list with exactly 1 PagePayload."""
        result = preprocess(portrait_tiff_bytes, "scan.tiff")
        assert len(result) == 1
        assert isinstance(result[0], PagePayload)

    def test_unsupported_format_raises_value_error(self):
        """PREP-01: Unsupported format (.docx) raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            preprocess(b"fake content", "document.docx")


# ---------------------------------------------------------------------------
# PREP-02: DPI normalization
# ---------------------------------------------------------------------------

class TestDpiNormalization:
    """Pages below 200 DPI are upscaled to 300 DPI via Lanczos."""

    def test_low_dpi_image_is_upscaled(self, portrait_png_72dpi):
        """PREP-02: A 72-DPI image is upscaled -- output DPI >= 200."""
        result = preprocess(portrait_png_72dpi, "low_res.png")
        page = result[0]
        assert page.dpi >= 200
        # Image should be larger than the 100x150 original
        assert page.width > 100
        assert page.height > 150

    def test_high_dpi_image_not_upscaled(self, portrait_png_300dpi):
        """PREP-02: A 300-DPI image is NOT upscaled -- preserves original DPI."""
        result = preprocess(portrait_png_300dpi, "high_res.png")
        page = result[0]
        assert page.dpi == 300


# ---------------------------------------------------------------------------
# PREP-03: De-skew
# ---------------------------------------------------------------------------

class TestDeskew:
    """Skewed pages (up to 15 degrees) are corrected."""

    def test_skewed_image_is_corrected(self, skewed_png_10deg):
        """PREP-03: A 10-degree skewed image is corrected (no crash, returns valid payload)."""
        result = preprocess(skewed_png_10deg, "skewed.png")
        page = result[0]
        assert page.error is None
        assert page.image_bytes  # Non-empty image bytes

    def test_extreme_skew_not_corrected(self, skewed_png_20deg):
        """PREP-03: A 20-degree skew exceeds 15-degree limit -- image is left unchanged."""
        result = preprocess(skewed_png_20deg, "very_skewed.png")
        page = result[0]
        assert page.error is None
        assert page.image_bytes  # Still produces valid output


# ---------------------------------------------------------------------------
# PREP-04: Multi-page PDF chunking
# ---------------------------------------------------------------------------

class TestMultipagePdf:
    """Multi-page PDFs produce one PagePayload per page."""

    def test_three_page_pdf_produces_three_payloads(self, three_page_pdf_bytes):
        """PREP-04: A 3-page PDF produces 3 PagePayload objects with page_num 0, 1, 2."""
        result = preprocess(three_page_pdf_bytes, "multipage.pdf")
        assert len(result) == 3
        assert [p.page_num for p in result] == [0, 1, 2]

    def test_pdf_payloads_have_pdf_bytes(self, three_page_pdf_bytes):
        """PREP-04: Each PagePayload from a PDF has pdf_bytes set to the original PDF bytes."""
        result = preprocess(three_page_pdf_bytes, "multipage.pdf")
        for page in result:
            assert page.pdf_bytes == three_page_pdf_bytes


# ---------------------------------------------------------------------------
# PREP-05: Landscape rotation
# ---------------------------------------------------------------------------

class TestLandscapeRotation:
    """Landscape pages (width > height) are rotated to portrait."""

    def test_landscape_image_is_rotated(self, landscape_png_300dpi):
        """PREP-05: A landscape image is rotated -- output width < height, was_rotated=True."""
        result = preprocess(landscape_png_300dpi, "wide.png")
        page = result[0]
        assert page.was_rotated is True
        assert page.width <= page.height

    def test_portrait_image_not_rotated(self, portrait_png_300dpi):
        """PREP-05: A portrait image is NOT rotated -- was_rotated=False."""
        result = preprocess(portrait_png_300dpi, "tall.png")
        page = result[0]
        assert page.was_rotated is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Corrupted pages do not crash the pipeline."""

    def test_corrupted_pdf_returns_error_payload(self, corrupted_pdf_bytes):
        """A corrupted PDF does not crash -- returns PagePayload with error field set."""
        result = preprocess(corrupted_pdf_bytes, "broken.pdf")
        assert isinstance(result, list)
        assert len(result) >= 1
        # At least one page should have an error
        error_pages = [p for p in result if p.error is not None]
        assert len(error_pages) >= 1
        for ep in error_pages:
            assert ep.image_bytes == b""
            assert ep.dpi == 0
            assert ep.width == 0
            assert ep.height == 0

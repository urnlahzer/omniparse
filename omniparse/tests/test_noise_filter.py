"""Tests for pre-alignment noise filter."""
from omniparse.noise_filter import filter_noise_regions, _is_noise, _is_line_number
from omniparse.models.region import Region


def _make_region(x1, y1, x2, y2, text="test", element_type="printed_text"):
    """Helper: create Region with [0,1] normalized bbox."""
    return Region(
        id="r_001",
        element_type=element_type,
        bounding_box=[x1 * 2550, y1 * 3300, x2 * 2550, y2 * 3300],
        bounding_box_norm=[x1, y1, x2, y2],
        confidence=0.9,
        text_content=text,
    )


class TestIsLineNumber:
    def test_single_digit(self):
        assert _is_line_number("7")

    def test_double_digit(self):
        assert _is_line_number("28")

    def test_with_period(self):
        assert _is_line_number("14.")

    def test_text_is_not(self):
        assert not _is_line_number("Article")

    def test_three_digits_is_not(self):
        assert not _is_line_number("123")

    def test_mixed_is_not(self):
        assert not _is_line_number("7a")


class TestPageBottomNoise:
    def test_page_number_at_bottom(self):
        """Small region at y=0.97 is noise."""
        r = _make_region(0.48, 0.97, 0.52, 0.99, text="33")
        assert _is_noise(r)

    def test_footer_fragment_at_bottom(self):
        """Small footer-like region at bottom."""
        r = _make_region(0.06, 0.97, 0.15, 0.99, text="300781215.3")
        assert _is_noise(r)

    def test_large_bottom_region_kept(self):
        """Large region at bottom is NOT noise (could be last paragraph)."""
        r = _make_region(0.05, 0.93, 0.95, 0.99, text="Final paragraph text...")
        assert not _is_noise(r)

    def test_mid_page_small_region_kept(self):
        """Small region in middle of page is NOT noise."""
        r = _make_region(0.4, 0.5, 0.45, 0.52, text="42")
        assert not _is_noise(r)


class TestMarginLineNumbers:
    def test_line_number_in_left_margin(self):
        """Tiny digit in left margin is noise."""
        r = _make_region(0.09, 0.30, 0.11, 0.32, text="15")
        assert _is_noise(r)

    def test_line_number_single_digit(self):
        """Single digit line number."""
        r = _make_region(0.09, 0.10, 0.10, 0.11, text="7")
        assert _is_noise(r)

    def test_address_in_margin_kept(self):
        """Small address near left edge is NOT noise (has text, not just digits)."""
        r = _make_region(0.05, 0.30, 0.30, 0.35, text="123 Main St")
        assert not _is_noise(r)

    def test_larger_margin_region_kept(self):
        """Region in margin but too large to be a line number."""
        r = _make_region(0.05, 0.10, 0.12, 0.20, text="14")
        assert not _is_noise(r)  # area > 0.001

    def test_margin_text_not_digit_kept(self):
        """Non-digit text in margin kept."""
        r = _make_region(0.09, 0.30, 0.11, 0.32, text="Re:")
        assert not _is_noise(r)


class TestFilterNoiseRegions:
    def test_filters_line_numbers_and_page_numbers(self):
        regions = [
            _make_region(0.05, 0.10, 0.95, 0.30, text="Paragraph text"),  # real content
            _make_region(0.09, 0.30, 0.11, 0.32, text="15"),  # line number
            _make_region(0.48, 0.97, 0.52, 0.99, text="33"),  # page number
            _make_region(0.05, 0.40, 0.95, 0.60, text="More content"),  # real content
        ]
        filtered = filter_noise_regions(regions)
        assert len(filtered) == 2
        assert filtered[0].text_content == "Paragraph text"
        assert filtered[1].text_content == "More content"

    def test_keeps_all_when_no_noise(self):
        regions = [
            _make_region(0.05, 0.10, 0.95, 0.30, text="Content"),
            _make_region(0.05, 0.40, 0.95, 0.60, text="More"),
        ]
        filtered = filter_noise_regions(regions)
        assert len(filtered) == 2

    def test_no_bbox_norm_kept(self):
        """Regions without bounding_box_norm are kept (safe fallback)."""
        r = Region(
            id="r_001",
            element_type="printed_text",
            bounding_box=[0.0, 0.0, 10.0, 10.0],
            confidence=0.9,
            text_content="7",
        )
        assert not _is_noise(r)

    def test_empty_list(self):
        assert filter_noise_regions([]) == []

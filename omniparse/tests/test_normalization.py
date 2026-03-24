"""Tests for bounding box normalization utility."""
import random
import pytest
from omniparse.normalization import normalize_bbox, normalize_to_unit, unit_to_pixel


class TestNormalizeBbox:
    """Test coordinate conversion from engine-native to 300 DPI pixel top-left."""

    def test_pdf_points_to_pixels(self):
        """PDF points (72 DPI) scaled to 300 DPI pixels."""
        result = normalize_bbox([72.0, 100.0, 540.0, 700.0], "pdf_points_topleft", dpi=300)
        assert result == [300.0, 416.67, 2250.0, 2916.67]

    def test_pixel_topleft_identity(self):
        """Pixel coordinates at target DPI are identity (no conversion)."""
        result = normalize_bbox([100.0, 200.0, 500.0, 300.0], "pixel_topleft", dpi=300)
        assert result == [100.0, 200.0, 500.0, 300.0]

    def test_docling_bottomleft_flip(self):
        """Docling bottom-left origin scaled from PDF points and flipped to top-left."""
        # bbox = [l, b, r, t] in bottom-left PDF points (72 DPI)
        # scale = 300/72 = 4.1667
        # l_px = 50 * 4.1667 = 208.33, r_px = 500 * 4.1667 = 2083.33
        # t_px = 200 * 4.1667 = 833.33 -> y_top = 3300 - 833.33 = 2466.67
        # b_px = 100 * 4.1667 = 416.67 -> y_bot = 3300 - 416.67 = 2883.33
        result = normalize_bbox([50.0, 100.0, 500.0, 200.0], "docling_bottomleft", page_height=3300.0)
        assert result == [208.33, 2466.67, 2083.33, 2883.33]

    def test_unknown_system_raises(self):
        """Unknown coordinate system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown coordinate system"):
            normalize_bbox([0, 0, 100, 100], "unknown_system")

    def test_docling_missing_page_height_raises(self):
        """docling_bottomleft without page_height raises ValueError."""
        with pytest.raises(ValueError, match="page_height required"):
            normalize_bbox([0, 0, 100, 100], "docling_bottomleft")

    def test_pdf_points_rounding(self):
        """Values are rounded to 2 decimal places."""
        result = normalize_bbox([1.0, 1.0, 1.0, 1.0], "pdf_points_topleft", dpi=300)
        # 1.0 * 300/72 = 4.166666...
        assert result == [4.17, 4.17, 4.17, 4.17]

    def test_zero_bbox_pdf_points(self):
        """Zero bbox returns zero for pdf_points."""
        assert normalize_bbox([0, 0, 0, 0], "pdf_points_topleft") == [0.0, 0.0, 0.0, 0.0]

    def test_zero_bbox_pixel(self):
        """Zero bbox returns zero for pixel_topleft."""
        assert normalize_bbox([0, 0, 0, 0], "pixel_topleft") == [0.0, 0.0, 0.0, 0.0]

    def test_zero_bbox_docling(self):
        """Zero bbox for docling_bottomleft with page_height (scaled zeros remain zero)."""
        result = normalize_bbox([0, 0, 0, 0], "docling_bottomleft", page_height=3300.0)
        # All coords are 0 * scale = 0; Y-flip: 3300 - 0 = 3300
        assert result == [0.0, 3300.0, 0.0, 3300.0]


class TestNormalizeToUnit:
    """Test [0,1] coordinate normalization and round-trip conversion."""

    def test_basic_conversion(self):
        """normalize_to_unit divides pixel coords by page dimensions with full precision."""
        result = normalize_to_unit([100.0, 200.0, 500.0, 600.0], 2550, 3300)
        assert result == pytest.approx(
            [100.0 / 2550, 200.0 / 3300, 500.0 / 2550, 600.0 / 3300]
        )

    def test_zero_page_dimensions_raises(self):
        """page_width=0 or page_height=0 raises ValueError."""
        with pytest.raises(ValueError, match="Page dimensions must be positive"):
            normalize_to_unit([0, 0, 100, 100], 0, 3300)
        with pytest.raises(ValueError, match="Page dimensions must be positive"):
            normalize_to_unit([0, 0, 100, 100], 2550, 0)

    def test_negative_page_dimensions_raises(self):
        """Negative page dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Page dimensions must be positive"):
            normalize_to_unit([0, 0, 100, 100], -1, 3300)

    def test_full_page_bbox(self):
        """Full-page bbox normalizes to [0, 0, 1, 1]."""
        result = normalize_to_unit([0, 0, 2550, 3300], 2550, 3300)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_no_rounding(self):
        """Results preserve full float64 precision (no 2-decimal rounding)."""
        result = normalize_to_unit([100.0, 200.0, 500.0, 600.0], 2550, 3300)
        # Verify at least one value has more than 2 decimal places
        assert any(
            len(str(v).split(".")[-1]) > 2
            for v in result
            if "." in str(v)
        )

    def test_unit_to_pixel_basic(self):
        """unit_to_pixel converts [0,1] back to pixel coordinates."""
        result = unit_to_pixel([0.5, 0.5, 1.0, 1.0], 2550, 3300)
        assert result == [1275, 1650, 2550, 3300]

    def test_unit_to_pixel_uses_round_not_int(self):
        """unit_to_pixel uses round() not int() truncation."""
        # 0.3333333 * 3 = 0.9999999 -> round() = 1, int() = 0
        result = unit_to_pixel([0.3333333, 0, 0, 0], 3, 1)
        assert result[0] == 1  # round(0.9999999) = 1, not int(0.9999999) = 0

    def test_roundtrip_max_1_pixel_error(self):
        """Round-trip: normalize then denormalize has max 1-pixel error per coordinate."""
        rng = random.Random(42)
        page_w, page_h = 2550, 3300
        max_error = 0
        for _ in range(1000):
            x1 = rng.uniform(0, page_w)
            y1 = rng.uniform(0, page_h)
            x2 = rng.uniform(x1, page_w)
            y2 = rng.uniform(y1, page_h)
            bbox = [x1, y1, x2, y2]
            normed = normalize_to_unit(bbox, page_w, page_h)
            restored = unit_to_pixel(normed, page_w, page_h)
            for orig, rest in zip(bbox, restored):
                error = abs(orig - rest)
                if error > max_error:
                    max_error = error
                assert error <= 1.0, f"Round-trip error {error} > 1 pixel for {bbox}"
        assert max_error <= 1.0

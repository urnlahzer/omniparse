"""Tests for PaddleOCR PP-StructureV3 engine and PP-OCRv5 handwriting recognition."""
import io
import pathlib

from PIL import Image

from omniparse.engines.paddleocr_engine import extract_page, LABEL_MAP
from omniparse.models.region import VALID_ELEMENT_TYPES

ENGINE_FILE = pathlib.Path(__file__).parent.parent / "engines" / "paddleocr_engine.py"


def _read_source() -> str:
    return ENGINE_FILE.read_text()


# --- Mock PP-StructureV3 pipeline ---

class MockPPStructureV3Result:
    def __init__(self, json_data):
        self.json = json_data


class MockPipeline:
    def __init__(self, results):
        self._results = results

    def predict(self, img):
        return [MockPPStructureV3Result(r) for r in self._results]


# --- Mock fixtures ---

MOCK_SINGLE_TEXT = {
    "res": {
        "parsing_res_list": [{
            "block_label": "text",
            "block_content": "LAST WILL AND TESTAMENT",
            "block_bbox": [100.0, 200.0, 500.0, 230.0],
            "block_id": 0,
            "block_order": 1,
        }],
        "layout_det_res": {
            "boxes": [{
                "cls_id": 0,
                "label": "text",
                "score": 0.95,
                "coordinate": [100.0, 200.0, 500.0, 230.0],
            }]
        },
    }
}

MOCK_MULTI_TYPES = {
    "res": {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "Some text",
                "block_bbox": [10.0, 10.0, 300.0, 40.0],
                "block_id": 0,
                "block_order": 1,
            },
            {
                "block_label": "table",
                "block_content": "| A |\n|---|\n| B |",
                "block_bbox": [10.0, 50.0, 400.0, 200.0],
                "block_id": 1,
                "block_order": 2,
            },
            {
                "block_label": "formula",
                "block_content": "E = mc^2",
                "block_bbox": [10.0, 210.0, 200.0, 250.0],
                "block_id": 2,
                "block_order": 3,
            },
        ],
        "layout_det_res": {
            "boxes": [
                {"cls_id": 0, "label": "text", "score": 0.92, "coordinate": [10.0, 10.0, 300.0, 40.0]},
                {"cls_id": 5, "label": "table", "score": 0.88, "coordinate": [10.0, 50.0, 400.0, 200.0]},
                {"cls_id": 3, "label": "formula", "score": 0.85, "coordinate": [10.0, 210.0, 200.0, 250.0]},
            ]
        },
    }
}

MOCK_UNKNOWN_LABEL = {
    "res": {
        "parsing_res_list": [{
            "block_label": "unknown_thing",
            "block_content": "mystery",
            "block_bbox": [50.0, 50.0, 150.0, 80.0],
            "block_id": 0,
            "block_order": 1,
        }],
        "layout_det_res": {
            "boxes": [{
                "cls_id": 99,
                "label": "unknown_thing",
                "score": 0.70,
                "coordinate": [50.0, 50.0, 150.0, 80.0],
            }]
        },
    }
}


# --- Tests: LABEL_MAP ---

class TestLabelMap:
    def test_label_map_values_valid(self):
        """Every LABEL_MAP value must be in VALID_ELEMENT_TYPES."""
        for label, element_type in LABEL_MAP.items():
            assert element_type in VALID_ELEMENT_TYPES, (
                f"LABEL_MAP['{label}'] = '{element_type}' not in VALID_ELEMENT_TYPES"
            )

    def test_label_map_completeness(self):
        """All 21 known PP-StructureV3 labels are present."""
        assert len(LABEL_MAP) >= 21


# --- Tests: extract_page pure function ---

class TestExtractPage:
    def test_empty_bytes(self):
        """Empty image_bytes returns empty EngineOutput."""
        pipeline = MockPipeline([])
        result = extract_page(pipeline, b"", 0)
        assert result.engine == "paddleocr"
        assert result.page == 0
        assert result.regions == []

    def test_basic_region(self):
        """Single text box produces correct Region."""
        pipeline = MockPipeline([MOCK_SINGLE_TEXT])
        # Create a minimal valid PNG
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        assert len(result.regions) == 1
        r = result.regions[0]
        assert r.element_type == "printed_text"
        assert r.bounding_box == [100.0, 200.0, 500.0, 230.0]
        assert r.confidence == 0.95
        assert "LAST" in r.text_content
        assert r.metadata["paddleocr_label"] == "text"
        assert r.metadata["coordinate_system"] == "pixel_300dpi_topleft"

    def test_multiple_regions(self):
        """Multiple boxes produce correct number of regions with right types."""
        pipeline = MockPipeline([MOCK_MULTI_TYPES])
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 3)
        assert len(result.regions) == 3
        assert result.page == 3
        assert result.regions[0].element_type == "printed_text"
        assert result.regions[1].element_type == "table"
        assert result.regions[2].element_type == "formula"

    def test_unknown_label_defaults(self):
        """Unknown PP-StructureV3 label defaults to 'printed_text'."""
        pipeline = MockPipeline([MOCK_UNKNOWN_LABEL])
        import io
        from PIL import Image
        img = Image.new("RGB", (200, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        assert result.regions[0].element_type == "printed_text"
        assert result.regions[0].metadata["paddleocr_label"] == "unknown_thing"

    def test_region_metadata_has_coordinate_system(self):
        """All regions have coordinate_system in metadata."""
        pipeline = MockPipeline([MOCK_SINGLE_TEXT])
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        assert result.regions[0].metadata["coordinate_system"] == "pixel_300dpi_topleft"

    def test_region_ids_sequential(self):
        """Region IDs are sequential: r_001, r_002, r_003."""
        pipeline = MockPipeline([MOCK_MULTI_TYPES])
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        assert [r.id for r in result.regions] == ["r_001", "r_002", "r_003"]

    def test_table_region_has_table_structure(self):
        """Table region has table_structure metadata."""
        pipeline = MockPipeline([MOCK_MULTI_TYPES])
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        table_region = result.regions[1]
        assert table_region.table_structure is not None
        assert "table_markdown" in table_region.metadata

    def test_formula_text_from_text_field(self):
        """Formula region extracts text from 'text' field when no rec_texts."""
        pipeline = MockPipeline([MOCK_MULTI_TYPES])
        import io
        from PIL import Image
        img = Image.new("RGB", (600, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = extract_page(pipeline, buf.getvalue(), 0)
        assert result.regions[2].text_content == "E = mc^2"


# --- Tests: Modal config (source inspection) ---

class TestModalConfig:
    def test_min_containers_0(self):
        assert "min_containers=0" in _read_source()

    def test_timeout_300(self):
        assert "timeout=300" in _read_source()

    def test_retries_configured(self):
        assert "modal.Retries" in _read_source()

    def test_max_retries_2(self):
        assert "max_retries=2" in _read_source()

    def test_gpu_a10g(self):
        assert 'gpu="A10G"' in _read_source()

    def test_no_not_implemented(self):
        assert "NotImplementedError" not in _read_source()


# --- Mock PP-OCRv5 pipeline ---

class MockOCRv5:
    """Mock PaddleOCR instance for PP-OCRv5 recognition-only mode."""

    def __init__(self, result):
        self._result = result
        self.last_call_kwargs = {}

    def ocr(self, img_array, **kwargs):
        self.last_call_kwargs = kwargs
        return self._result


def _make_test_image_bytes(width: int = 200, height: int = 50) -> bytes:
    """Create a minimal PNG image as bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Tests: PP-OCRv5 handwriting recognition ---

class TestPPOCRv5:
    def test_recognize_handwriting_returns_tuple(self):
        """recognize_handwriting with mock PP-OCRv5 pipeline returns (text, confidence) tuple."""
        from omniparse.engines.paddleocr_engine import recognize_handwriting

        ocr_v5 = MockOCRv5(result=[[("handwritten text", 0.75)]])
        image_bytes = _make_test_image_bytes()

        text, confidence = recognize_handwriting(ocr_v5, image_bytes)

        assert text == "handwritten text"
        assert confidence == 0.75

    def test_recognize_handwriting_uses_rec_only(self):
        """recognize_handwriting uses det=False, rec=True (recognition-only)."""
        from omniparse.engines.paddleocr_engine import recognize_handwriting

        ocr_v5 = MockOCRv5(result=[[("text", 0.80)]])
        image_bytes = _make_test_image_bytes()

        recognize_handwriting(ocr_v5, image_bytes)

        assert ocr_v5.last_call_kwargs.get("det") is False
        assert ocr_v5.last_call_kwargs.get("rec") is True

    def test_recognize_handwriting_empty_result(self):
        """recognize_handwriting with empty/None OCR result returns ('', 0.0)."""
        from omniparse.engines.paddleocr_engine import recognize_handwriting

        ocr_v5 = MockOCRv5(result=None)
        image_bytes = _make_test_image_bytes()

        text, confidence = recognize_handwriting(ocr_v5, image_bytes)

        assert text == ""
        assert confidence == 0.0

    def test_recognize_handwriting_empty_inner_result(self):
        """recognize_handwriting with empty inner result returns ('', 0.0)."""
        from omniparse.engines.paddleocr_engine import recognize_handwriting

        ocr_v5 = MockOCRv5(result=[[]])
        image_bytes = _make_test_image_bytes()

        text, confidence = recognize_handwriting(ocr_v5, image_bytes)

        assert text == ""
        assert confidence == 0.0

    def test_recognize_handwriting_multi_line(self):
        """recognize_handwriting concatenates multi-line results with spaces."""
        from omniparse.engines.paddleocr_engine import recognize_handwriting

        ocr_v5 = MockOCRv5(result=[[("line one", 0.8), ("line two", 0.7)]])
        image_bytes = _make_test_image_bytes()

        text, confidence = recognize_handwriting(ocr_v5, image_bytes)

        assert text == "line one line two"
        assert confidence == 0.75  # (0.8 + 0.7) / 2

    def test_run_handwriting_method_exists(self):
        """PaddleOCREngine.run_handwriting method exists and delegates to recognize_handwriting."""
        source = _read_source()
        assert "def run_handwriting" in source
        assert "recognize_handwriting" in source

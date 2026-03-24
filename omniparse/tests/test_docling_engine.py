"""Tests for Docling hierarchical structure extraction engine."""
import pathlib
from unittest.mock import patch

from omniparse.engines.docling_engine import extract_pages, DOCLING_LABEL_MAP, DOCLING_CONFIDENCE
from omniparse.models.region import VALID_ELEMENT_TYPES

ENGINE_FILE = pathlib.Path(__file__).parent.parent / "engines" / "docling_engine.py"


def _read_source() -> str:
    return ENGINE_FILE.read_text()


# --- Mock Docling objects ---

class MockBBox:
    def __init__(self, l, b, r, t):
        self.l = l
        self.b = b
        self.r = r
        self.t = t


class MockProv:
    def __init__(self, page_no, bbox):
        self.page_no = page_no
        self.bbox = bbox


class MockItem:
    def __init__(self, label, text, prov, level=1):
        self.label = label
        self.text = text
        self.prov = prov


class MockTableItem:
    def __init__(self, label, text, prov, markdown, level=1):
        self.label = label
        self.text = text
        self.prov = prov
        self._markdown = markdown

    def export_to_markdown(self):
        return self._markdown


class MockDocument:
    def __init__(self, items):
        self._items = items  # list of (item, level)

    def iterate_items(self):
        return self._items


class MockResult:
    def __init__(self, document):
        self.document = document


class MockConverter:
    def __init__(self, document):
        self._document = document

    def convert(self, source):
        return MockResult(self._document)


# --- Tests: DOCLING_LABEL_MAP ---

class TestLabelMap:
    def test_label_map_values_valid(self):
        """Every DOCLING_LABEL_MAP value must be in VALID_ELEMENT_TYPES."""
        for label, element_type in DOCLING_LABEL_MAP.items():
            assert element_type in VALID_ELEMENT_TYPES, (
                f"DOCLING_LABEL_MAP['{label}'] = '{element_type}' not in VALID_ELEMENT_TYPES"
            )


# --- Tests: extract_pages pure function ---

class TestExtractPages:
    def _make_converter(self, items):
        """Create a MockConverter with given (item, level) pairs."""
        doc = MockDocument(items)
        return MockConverter(doc)

    def test_empty_bytes(self):
        """Empty pdf_bytes returns empty dict."""
        converter = self._make_converter([])
        result = extract_pages(converter, b"", {})
        assert result == {}

    @patch("omniparse.engines.docling_engine._make_source")
    def test_basic_region(self, mock_source):
        """Single text item on page 1 -> page 0 EngineOutput."""
        mock_source.return_value = "dummy"
        item = MockItem(
            label="text",
            text="LAST WILL AND TESTAMENT",
            prov=[MockProv(1, MockBBox(50.0, 100.0, 500.0, 200.0))],
        )
        converter = self._make_converter([(item, 1)])
        page_heights = {0: 3300.0}

        result = extract_pages(converter, b"fake-pdf", page_heights)
        assert 0 in result
        eo = result[0]
        assert eo.engine == "docling"
        assert eo.page == 0
        assert len(eo.regions) == 1
        r = eo.regions[0]
        assert r.element_type == "printed_text"
        assert r.text_content == "LAST WILL AND TESTAMENT"

    @patch("omniparse.engines.docling_engine._make_source")
    def test_multiple_pages(self, mock_source):
        """Items on pages 1 and 2 -> two EngineOutput entries."""
        mock_source.return_value = "dummy"
        item1 = MockItem("text", "Page 1 text", [MockProv(1, MockBBox(10, 10, 100, 50))])
        item2 = MockItem("text", "Page 2 text", [MockProv(2, MockBBox(10, 10, 100, 50))])
        converter = self._make_converter([(item1, 1), (item2, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0, 1: 3300.0})
        assert 0 in result
        assert 1 in result
        assert result[0].regions[0].text_content == "Page 1 text"
        assert result[1].regions[0].text_content == "Page 2 text"

    @patch("omniparse.engines.docling_engine._make_source")
    def test_coordinate_conversion(self, mock_source):
        """Docling bottom-left PDF point coords are scaled and converted to top-left pixels."""
        mock_source.return_value = "dummy"
        # bbox: l=50, b=100, r=500, t=200 in bottom-left PDF points (72 DPI)
        # scale = 300/72 = 4.1667
        # l_px=208.33, r_px=2083.33, t_px=833.33, b_px=416.67
        # top-left: [208.33, 3300-833.33=2466.67, 2083.33, 3300-416.67=2883.33]
        item = MockItem("text", "test", [MockProv(1, MockBBox(50.0, 100.0, 500.0, 200.0))])
        converter = self._make_converter([(item, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        r = result[0].regions[0]
        assert r.bounding_box == [208.33, 2466.67, 2083.33, 2883.33]

    @patch("omniparse.engines.docling_engine._make_source")
    def test_unknown_label(self, mock_source):
        """Unknown Docling label defaults to 'printed_text'."""
        mock_source.return_value = "dummy"
        item = MockItem("unknown_thing", "mystery", [MockProv(1, MockBBox(0, 0, 100, 50))])
        converter = self._make_converter([(item, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        assert result[0].regions[0].element_type == "printed_text"
        assert result[0].regions[0].metadata["docling_label"] == "unknown_thing"

    @patch("omniparse.engines.docling_engine._make_source")
    def test_confidence_fixed(self, mock_source):
        """All Docling regions have fixed confidence."""
        mock_source.return_value = "dummy"
        item = MockItem("text", "test", [MockProv(1, MockBBox(0, 0, 100, 50))])
        converter = self._make_converter([(item, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        assert result[0].regions[0].confidence == DOCLING_CONFIDENCE

    @patch("omniparse.engines.docling_engine._make_source")
    def test_metadata_coordinate_system(self, mock_source):
        """Metadata includes coordinate_system."""
        mock_source.return_value = "dummy"
        item = MockItem("text", "test", [MockProv(1, MockBBox(0, 0, 100, 50))])
        converter = self._make_converter([(item, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        assert result[0].regions[0].metadata["coordinate_system"] == "pixel_300dpi_topleft"

    @patch("omniparse.engines.docling_engine._make_source")
    def test_metadata_hierarchy_level(self, mock_source):
        """Metadata includes hierarchy_level from iterate_items."""
        mock_source.return_value = "dummy"
        item = MockItem("section_header", "Chapter 1", [MockProv(1, MockBBox(0, 0, 100, 50))])
        converter = self._make_converter([(item, 2)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        assert result[0].regions[0].metadata["hierarchy_level"] == 2

    @patch("omniparse.engines.docling_engine._make_source")
    def test_table_has_table_structure(self, mock_source):
        """Table item with export_to_markdown produces table_structure."""
        mock_source.return_value = "dummy"
        table_md = "| Col1 | Col2 |\n|------|------|\n| A | B |"
        item = MockTableItem(
            "table", "table content",
            [MockProv(1, MockBBox(0, 0, 400, 200))],
            markdown=table_md,
        )
        converter = self._make_converter([(item, 1)])

        result = extract_pages(converter, b"fake", {0: 3300.0})
        r = result[0].regions[0]
        assert r.table_structure is not None
        assert r.table_structure["rows"] > 0
        assert r.table_structure["cols"] > 0

    @patch("omniparse.engines.docling_engine._make_source")
    def test_region_ids_sequential_per_page(self, mock_source):
        """Regions on same page have sequential IDs."""
        mock_source.return_value = "dummy"
        items = [
            (MockItem("text", "first", [MockProv(1, MockBBox(0, 0, 100, 30))]), 1),
            (MockItem("text", "second", [MockProv(1, MockBBox(0, 40, 100, 70))]), 1),
            (MockItem("text", "third", [MockProv(1, MockBBox(0, 80, 100, 110))]), 1),
        ]
        converter = self._make_converter(items)

        result = extract_pages(converter, b"fake", {0: 3300.0})
        ids = [r.id for r in result[0].regions]
        assert ids == ["r_001", "r_002", "r_003"]


# --- Tests: Modal config (source inspection) ---

class TestModalConfig:
    def test_gpu_l4(self):
        assert 'gpu="L4"' in _read_source()

    def test_timeout_600(self):
        assert "timeout=600" in _read_source()

    def test_retries_configured(self):
        assert "modal.Retries" in _read_source()

    def test_max_retries_2(self):
        assert "max_retries=2" in _read_source()

    def test_cuda_fallback_pattern(self):
        source = _read_source()
        assert "has_cuda" in source or "AcceleratorDevice.CUDA" in source
        assert "DocumentConverter(" in source  # fallback always present

    def test_pypdfium2_backend(self):
        source = _read_source()
        assert "PyPdfiumDocumentBackend" in source
        assert "pypdfium2_backend" in source

    def test_no_not_implemented(self):
        assert "NotImplementedError" not in _read_source()

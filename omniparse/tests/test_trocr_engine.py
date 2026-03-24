"""Tests for TrOCR handwriting recognition engine (DBNet line segmentation + TrOCR inference).

Uses mock-based testing pattern: no GPU, no real models. Pure functions are tested
with mock pipelines that simulate DBNet and TrOCR behavior.

torch is not installed in the local venv (GPU-only), so we inject a mock torch
module via sys.modules before importing the engine. This follows the same pattern
used for vLLM in test_llm_arbiter.py.
"""
import io
import math
import pathlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from omniparse.models.region import EngineOutput


ENGINE_FILE = pathlib.Path(__file__).parent.parent / "engines" / "trocr_engine.py"

# ---------------------------------------------------------------------------
# Mock torch module -- injected into sys.modules so engine imports work
# without a real torch installation. Only the subset used by recognize_line
# is implemented.
# ---------------------------------------------------------------------------

_mock_torch = ModuleType("torch")


class _MockTensor:
    """Minimal tensor-like for test mocks."""

    def __init__(self, data):
        if isinstance(data, list):
            self._data = data
            self._flat = self._flatten(data)
        elif isinstance(data, (int, float)):
            self._data = data
            self._flat = [data]
        elif isinstance(data, _MockTensor):
            self._data = data._data
            self._flat = data._flat
        else:
            self._data = data
            self._flat = [data]

    def _flatten(self, d):
        if isinstance(d, list):
            out = []
            for x in d:
                out.extend(self._flatten(x))
            return out
        return [d]

    def item(self):
        if isinstance(self._data, (int, float)):
            return float(self._data)
        return float(self._flat[0])

    def mean(self):
        s = sum(self._flat)
        return _MockTensor(s / len(self._flat))

    def __getitem__(self, key):
        if isinstance(self._data, list):
            result = self._data
            if isinstance(key, tuple):
                for k in key:
                    # Convert _MockTensor keys to int (PyTorch tensor indexing)
                    if isinstance(k, _MockTensor):
                        k = int(k.item())
                    result = result[k]
            else:
                if isinstance(key, _MockTensor):
                    key = int(key.item())
                result = result[key]
            if isinstance(result, list):
                return _MockTensor(result)
            return _MockTensor(result)
        return _MockTensor(self._data)

    @property
    def shape(self):
        return [len(self._flat)]

    def to(self, device):
        return self


def _torch_tensor(data):
    return _MockTensor(data)


def _torch_zeros(*shape):
    import functools
    total = functools.reduce(lambda a, b: a * b, shape, 1)
    flat = [0.0] * total
    return _MockTensor(flat)


def _torch_full(shape, fill_value):
    import functools
    total = functools.reduce(lambda a, b: a * b, shape, 1)
    # For 2D shape (1, vocab_size), create nested list
    if len(shape) == 2:
        row = [fill_value] * shape[1]
        data = [row for _ in range(shape[0])]
        return _MockTensor(data)
    return _MockTensor([fill_value] * total)


def _torch_log_softmax(tensor, dim=-1):
    # For the mock, we return the tensor as-is (scores are already log-prob-like)
    return tensor


def _torch_exp(tensor):
    val = tensor.item()
    return _MockTensor(math.exp(val))


# Wire up mock torch
_mock_torch.tensor = _torch_tensor
_mock_torch.zeros = _torch_zeros
_mock_torch.full = _torch_full
_mock_torch.log_softmax = _torch_log_softmax
_mock_torch.exp = _torch_exp
_mock_torch.cuda = MagicMock()
_mock_torch.cuda.is_available = MagicMock(return_value=False)

# Inject mock before any engine import
if "torch" not in sys.modules:
    sys.modules["torch"] = _mock_torch


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_test_image(width: int = 200, height: int = 100) -> bytes:
    """Create a minimal PNG image as bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_dbnet(polygons: list | None = None):
    """Create a mock DBNet (PaddleOCR det-only) pipeline.

    Args:
        polygons: List of polygon bounding boxes that .ocr() returns.
                  Each polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
                  None means DBNet found no text lines.
    """
    mock = MagicMock()
    if polygons is None:
        mock.ocr.return_value = None
    else:
        # PaddleOCR det-only returns: [list_of_polygons]
        mock.ocr.return_value = [polygons]
    return mock


def _make_mock_processor():
    """Create a mock TrOCR processor."""
    mock = MagicMock()
    # processor(image, return_tensors="pt") returns object with .pixel_values
    pixel_values = _torch_zeros(1, 3, 384, 384)
    call_result = MagicMock()
    call_result.pixel_values = pixel_values
    mock.return_value = call_result
    mock.batch_decode.return_value = ["Hello World"]
    return mock


def _make_mock_model(scores: list[float] | None = None):
    """Create a mock TrOCR model.

    Args:
        scores: List of log-softmax score values to simulate per-token confidence.
                If None, uses default moderate-confidence scores.
    """
    mock = MagicMock()
    mock.device = "cpu"

    # Build mock generate output
    gen_output = MagicMock()
    gen_output.sequences = _torch_tensor([[0, 5, 10, 15]])  # 4 tokens (including BOS)

    if scores is None:
        scores = [-0.2, -0.3, -0.1]  # moderate confidence

    # Each score is a tensor of shape (1, vocab_size) -- we only need the right token probs
    mock_scores = []
    for i, log_p in enumerate(scores):
        score_tensor = _torch_full((1, 50000), -10.0)  # low prob for all tokens
        token_id = gen_output.sequences[0, i + 1].item()
        score_tensor._data[0][int(token_id)] = log_p  # assign the target token's score
        mock_scores.append(score_tensor)

    gen_output.scores = mock_scores
    mock.generate.return_value = gen_output
    return mock


# --- Tests: segment_lines ---


class TestSegmentLines:
    def test_segment_lines_returns_cropped_images(self):
        """segment_lines on an image with a mock DBNet returns list of cropped line images."""
        from omniparse.engines.trocr_engine import segment_lines

        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        # Two text line polygons (top-left, top-right, bottom-right, bottom-left format)
        polygons = [
            [[10, 5], [190, 5], [190, 25], [10, 25]],
            [[10, 40], [190, 40], [190, 60], [10, 60]],
        ]
        dbnet = _make_mock_dbnet(polygons)

        lines = segment_lines(dbnet, img)

        assert len(lines) == 2
        for line in lines:
            assert isinstance(line, Image.Image)

    def test_segment_lines_none_returns_empty(self):
        """segment_lines on an image where DBNet returns None returns empty list."""
        from omniparse.engines.trocr_engine import segment_lines

        img = Image.new("RGB", (200, 100))
        dbnet = _make_mock_dbnet(None)

        lines = segment_lines(dbnet, img)

        assert lines == []


# --- Tests: recognize_line ---


class TestRecognizeLine:
    def test_recognize_line_returns_text_confidence(self):
        """recognize_line with a mock TrOCR processor+model returns (text, confidence) tuple."""
        from omniparse.engines.trocr_engine import recognize_line

        processor = _make_mock_processor()
        model = _make_mock_model()
        line_img = Image.new("RGB", (200, 30))

        text, confidence = recognize_line(processor, model, line_img)

        assert isinstance(text, str)
        assert isinstance(confidence, float)
        assert len(text) > 0

    def test_recognize_line_confidence_bounded(self):
        """recognize_line confidence is between 0.0 and 1.0."""
        from omniparse.engines.trocr_engine import recognize_line

        processor = _make_mock_processor()
        model = _make_mock_model()
        line_img = Image.new("RGB", (200, 30))

        _, confidence = recognize_line(processor, model, line_img)

        assert 0.0 <= confidence <= 1.0


# --- Tests: extract_handwriting ---


class TestExtractHandwriting:
    def test_extract_handwriting_returns_engine_output(self):
        """extract_handwriting with mock pipelines returns EngineOutput with engine='trocr', element_type='handwriting'."""
        from omniparse.engines.trocr_engine import extract_handwriting

        polygons = [
            [[10, 5], [190, 5], [190, 25], [10, 25]],
        ]
        dbnet = _make_mock_dbnet(polygons)
        processor = _make_mock_processor()
        model = _make_mock_model()
        image_bytes = _make_test_image()

        result = extract_handwriting(
            dbnet, processor, model, image_bytes,
            region_bbox=[10.0, 5.0, 190.0, 60.0], region_id="r_001",
        )

        assert isinstance(result, EngineOutput)
        assert result.engine == "trocr"
        assert len(result.regions) == 1
        assert result.regions[0].element_type == "handwriting"

    def test_extract_handwriting_empty_image(self):
        """extract_handwriting with empty image_bytes returns EngineOutput with empty regions."""
        from omniparse.engines.trocr_engine import extract_handwriting

        dbnet = _make_mock_dbnet(None)
        processor = _make_mock_processor()
        model = _make_mock_model()

        result = extract_handwriting(
            dbnet, processor, model, b"",
            region_bbox=[0.0, 0.0, 0.0, 0.0], region_id="r_001",
        )

        assert isinstance(result, EngineOutput)
        assert result.regions == []

    def test_extract_handwriting_hitl_flag(self):
        """extract_handwriting confidence < 0.20 sets metadata['hitl_flag'] = True."""
        from omniparse.engines.trocr_engine import extract_handwriting

        polygons = [
            [[10, 5], [190, 5], [190, 25], [10, 25]],
        ]
        dbnet = _make_mock_dbnet(polygons)
        processor = _make_mock_processor()
        # Very low confidence scores (log probs near -5 => confidence ~0.007)
        model = _make_mock_model(scores=[-5.0, -5.0, -5.0])
        image_bytes = _make_test_image()

        result = extract_handwriting(
            dbnet, processor, model, image_bytes,
            region_bbox=[10.0, 5.0, 190.0, 60.0], region_id="r_001",
        )

        assert result.regions[0].metadata["hitl_flag"] is True

    def test_extract_handwriting_concatenates_lines(self):
        """extract_handwriting concatenates all line texts with spaces."""
        from omniparse.engines.trocr_engine import extract_handwriting

        # Two line polygons
        polygons = [
            [[10, 5], [190, 5], [190, 25], [10, 25]],
            [[10, 40], [190, 40], [190, 60], [10, 60]],
        ]
        dbnet = _make_mock_dbnet(polygons)
        processor = _make_mock_processor()
        model = _make_mock_model()
        image_bytes = _make_test_image()

        result = extract_handwriting(
            dbnet, processor, model, image_bytes,
            region_bbox=[10.0, 5.0, 190.0, 60.0], region_id="r_001",
        )

        # Two lines recognized -> text should contain a space (concatenation)
        text = result.regions[0].text_content
        assert " " in text  # concatenated with spaces

    def test_extract_handwriting_no_lines_empty_regions(self):
        """extract_handwriting returns empty regions when DBNet detects no lines."""
        from omniparse.engines.trocr_engine import extract_handwriting

        dbnet = _make_mock_dbnet([])  # empty list, not None
        processor = _make_mock_processor()
        model = _make_mock_model()
        image_bytes = _make_test_image()

        result = extract_handwriting(
            dbnet, processor, model, image_bytes,
            region_bbox=[10.0, 5.0, 190.0, 60.0], region_id="r_001",
        )

        assert result.regions == []

    def test_extract_handwriting_metadata_engine_detail(self):
        """extract_handwriting region metadata contains engine_detail and num_lines."""
        from omniparse.engines.trocr_engine import extract_handwriting

        polygons = [
            [[10, 5], [190, 5], [190, 25], [10, 25]],
        ]
        dbnet = _make_mock_dbnet(polygons)
        processor = _make_mock_processor()
        model = _make_mock_model()
        image_bytes = _make_test_image()

        result = extract_handwriting(
            dbnet, processor, model, image_bytes,
            region_bbox=[10.0, 5.0, 190.0, 60.0], region_id="r_001",
        )

        meta = result.regions[0].metadata
        assert meta["engine_detail"] == "trocr_dbnet"
        assert meta["num_lines"] == 1


# --- Tests: Container image (source inspection) ---


class TestTrOCRImageDefinition:
    def _read_app_source(self) -> str:
        app_file = pathlib.Path(__file__).parent.parent / "app.py"
        return app_file.read_text()

    def test_trocr_image_has_paddlepaddle_gpu(self):
        """trocr_image definition includes 'paddlepaddle-gpu' (for DBNet)."""
        source = self._read_app_source()
        # Find the trocr_image block and verify paddlepaddle-gpu is within it
        assert "paddlepaddle-gpu" in source

    def test_trocr_image_has_transformers(self):
        """trocr_image definition includes 'transformers' (for TrOCR)."""
        source = self._read_app_source()
        assert "transformers" in source

    def test_trocr_image_has_torch(self):
        """trocr_image definition includes 'torch' (for PyTorch)."""
        source = self._read_app_source()
        assert '"torch"' in source

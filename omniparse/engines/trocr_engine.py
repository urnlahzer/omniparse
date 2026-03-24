"""TrOCR handwriting recognition engine (GPU, L4).

Performs DBNet text-line segmentation then per-line TrOCR-large-handwritten inference.
Primary handwriting recognizer weighted 2x in consensus (see consensus.py HANDWRITING_WEIGHTS).

Architecture:
- DBNet (PaddleOCR det-only) segments handwriting region into individual text lines
- TrOCR-large-handwritten recognizes each line independently
- Confidence computed from generation log-probabilities (not hardcoded)
- Very low confidence (<0.20) triggers hitl_flag for human review

Pattern: pure functions (segment_lines, recognize_line, extract_handwriting) + Modal class
wrapper (TrOCREngine). Pure functions enable GPU-free testing with mocks.

Coordinate system: TrOCR operates on pre-cropped region images; bounding boxes are
passed through from the PaddleOCR layout classifier. No coordinate transformation needed.
"""
import io
import logging

import modal
import numpy as np
from PIL import Image

from omniparse.app import app, trocr_image, model_volume, MODELS_DIR
from omniparse.models.region import Region, EngineOutput

logger = logging.getLogger(__name__)

# Confidence threshold below which hitl_flag is set (per CONTEXT.md)
HITL_CONFIDENCE_THRESHOLD = 0.20


def segment_lines(dbnet_pipeline, region_image: Image.Image) -> list[Image.Image]:
    """Segment a handwriting region into individual text lines using DBNet.

    Uses PaddleOCR in det-only mode (no recognition, no classification) to find
    text line bounding boxes, then crops the region image to each line.

    Args:
        dbnet_pipeline: PaddleOCR instance configured for detection only.
        region_image: PIL Image of the handwriting region.

    Returns:
        List of cropped line PIL Images, sorted top-to-bottom by y-coordinate.
        Empty list if no lines detected.
    """
    img_array = np.array(region_image)
    result = dbnet_pipeline.ocr(img_array, cls=False, det=True, rec=False)

    if not result or not result[0]:
        return []

    polygons = result[0]
    if not polygons:
        return []

    # Extract bounding rects from polygons, sort by y-coordinate (top-to-bottom)
    line_rects = []
    for polygon in polygons:
        # polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(region_image.width, x2)
        y2 = min(region_image.height, y2)
        if x2 > x1 and y2 > y1:
            line_rects.append((y1, x1, y2, x2))

    # Sort top-to-bottom by y-coordinate
    line_rects.sort(key=lambda r: r[0])

    # Crop each line
    lines = []
    for y1, x1, y2, x2 in line_rects:
        line_img = region_image.crop((x1, y1, x2, y2))
        lines.append(line_img)

    return lines


def recognize_line(processor, model, line_image: Image.Image) -> tuple[str, float]:
    """Recognize text in a single line image using TrOCR.

    Args:
        processor: TrOCRProcessor instance.
        model: VisionEncoderDecoderModel instance.
        line_image: PIL Image of a single text line.

    Returns:
        (text, confidence) tuple. Confidence derived from generation log-probabilities,
        clamped to [0.0, 1.0].
    """
    import torch

    rgb_image = line_image.convert("RGB")
    pixel_values = processor(rgb_image, return_tensors="pt").pixel_values.to(model.device)

    outputs = model.generate(
        pixel_values,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=128,
    )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    # Compute confidence from log-probabilities
    if outputs.scores:
        log_probs = []
        for i, score in enumerate(outputs.scores):
            token_id = outputs.sequences[0, i + 1]
            log_prob = torch.log_softmax(score, dim=-1)[0, token_id].item()
            log_probs.append(log_prob)
        confidence = torch.exp(torch.tensor(log_probs).mean()).item()
    else:
        confidence = 0.5

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    return text, confidence


def extract_handwriting(
    dbnet_pipeline,
    processor,
    model,
    region_image_bytes: bytes,
    region_bbox: list[float],
    region_id: str,
) -> EngineOutput:
    """Extract handwriting text from a region image using DBNet + TrOCR pipeline.

    Args:
        dbnet_pipeline: PaddleOCR instance for DBNet line segmentation.
        processor: TrOCRProcessor for image preprocessing.
        model: VisionEncoderDecoderModel for text recognition.
        region_image_bytes: PNG bytes of the cropped handwriting region.
        region_bbox: [x1, y1, x2, y2] bounding box from layout classifier.
        region_id: Unique region identifier (e.g., "r_001").

    Returns:
        EngineOutput with engine="trocr" and handwriting regions.
        Empty regions if image is empty or no lines detected.
    """
    if not region_image_bytes:
        return EngineOutput(page=0, engine="trocr", regions=[])

    try:
        img = Image.open(io.BytesIO(region_image_bytes)).convert("RGB")
    except Exception:
        logger.warning("Failed to open region image for %s", region_id)
        return EngineOutput(page=0, engine="trocr", regions=[])

    # Segment into text lines
    lines = segment_lines(dbnet_pipeline, img)

    if not lines:
        return EngineOutput(page=0, engine="trocr", regions=[])

    # Recognize each line
    texts = []
    confidences = []
    for line_img in lines:
        text, conf = recognize_line(processor, model, line_img)
        texts.append(text)
        confidences.append(conf)

    # Aggregate
    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    avg_confidence = max(0.0, min(1.0, avg_confidence))

    # Build metadata
    metadata = {
        "engine_detail": "trocr_dbnet",
        "num_lines": len(lines),
        "hitl_flag": avg_confidence < HITL_CONFIDENCE_THRESHOLD,
    }

    region = Region(
        id=region_id,
        element_type="handwriting",
        bounding_box=region_bbox,
        confidence=avg_confidence,
        text_content=combined_text,
        metadata=metadata,
    )

    return EngineOutput(page=0, engine="trocr", regions=[region])


@app.cls(
    gpu="L4",
    image=trocr_image,
    volumes={MODELS_DIR: model_volume},
    timeout=120,
    min_containers=0,
    max_containers=10,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
class TrOCREngine:
    @modal.enter()
    def load_model(self):
        """Load DBNet (PaddleOCR det-only) and TrOCR-large-handwritten on container start."""
        # Load DBNet for line segmentation
        from paddleocr import PaddleOCR
        logger.info("Loading DBNet (PaddleOCR det-only)...")
        self.dbnet = PaddleOCR(lang="en", use_gpu=True, det=True, rec=False, cls=False)

        # Load TrOCR
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        model_path = f"{MODELS_DIR}/trocr-large-handwritten"
        logger.info("Loading TrOCR from %s...", model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        logger.info("TrOCR model loaded on %s.", self.model.device)

    @modal.method()
    def run(self, region_image: bytes, region_bbox: list = None, region_id: str = "r_001") -> dict:
        """Process a handwriting region image. Returns EngineOutput as dict."""
        bbox = region_bbox or [0.0, 0.0, 0.0, 0.0]
        result = extract_handwriting(
            self.dbnet, self.processor, self.model,
            region_image, bbox, region_id,
        )
        return result.model_dump()

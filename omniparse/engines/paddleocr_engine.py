"""PaddleOCR PP-StructureV3 engine (GPU, A10G).

Performs layout classification (23 categories) and OCR text extraction on every
page image. PaddleOCR is the primary layout authority for element_type classification.

Also provides PP-OCRv5 recognition-only mode for handwriting second opinion
(det=False, rec=True). Used as the second specialist engine for handwriting regions,
weighted 1x in consensus against TrOCR's 2x (see consensus.py HANDWRITING_WEIGHTS).

Coordinate system: PP-StructureV3 outputs pixel coordinates at input image DPI.
Since the preprocessor outputs 300 DPI images, coordinates are already at 300 DPI.
normalize_bbox with "pixel_topleft" is called for rounding consistency.

Pattern: pure function (extract_page, recognize_handwriting) + Modal class wrapper
(PaddleOCREngine).
"""
import io
import logging

import numpy as np
import modal
from PIL import Image

from omniparse.app import app, paddleocr_image, model_volume, MODELS_DIR
from omniparse.models.region import Region, EngineOutput, VALID_ELEMENT_TYPES
from omniparse.normalization import normalize_bbox

logger = logging.getLogger(__name__)

# PP-StructureV3 label -> canonical element_type mapping.
# All values MUST be in VALID_ELEMENT_TYPES.
# Original PP-StructureV3 label preserved in metadata["paddleocr_label"].
LABEL_MAP = {
    "text": "printed_text",
    "document_title": "header",
    "paragraph_title": "header",
    "table": "table",
    "formula": "formula",
    "formula_number": "formula",
    "image": "image",
    "chart": "chart",
    "seal": "seal",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "abstract": "printed_text",
    "references": "printed_text",
    "footnotes": "footer",
    "algorithm": "printed_text",
    "sidebar_text": "printed_text",
    "lists": "printed_text",
    "figure_title": "printed_text",
    "table_caption": "printed_text",
    "figure": "image",
}


def extract_page(pipeline, image_bytes: bytes, page_num: int) -> EngineOutput:
    """Extract layout-classified regions from a page image using PP-StructureV3.

    Args:
        pipeline: Initialized PPStructureV3 instance (loaded in @modal.enter).
        image_bytes: PNG-encoded page image from preprocessor (300 DPI).
        page_num: Zero-indexed page number.

    Returns:
        EngineOutput with Region objects for each detected layout region.
    """
    if not image_bytes:
        return EngineOutput(page=page_num, engine="paddleocr", regions=[])

    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    results = pipeline.predict(img_array)
    regions: list[Region] = []
    counter = 1

    for res in results:
        json_data = res.json
        if not json_data or "res" not in json_data:
            continue

        res_data = json_data["res"]

        # PPStructureV3 output: parsing_res_list has layout blocks with text.
        # layout_det_res.boxes has layout detection (coordinates + labels, no text).
        # overall_ocr_res has raw OCR (rec_texts + rec_boxes, no layout).
        # We use parsing_res_list as primary source (layout + text combined),
        # falling back to layout_det_res.boxes for score/cls_id metadata.
        parsing_blocks = res_data.get("parsing_res_list", [])
        layout_boxes = (res_data.get("layout_det_res") or {}).get("boxes", [])

        # Build layout score lookup by block_id
        layout_scores = {}
        layout_cls_ids = {}
        for box in layout_boxes:
            # Match by coordinate proximity since block_id may not align
            coord = box.get("coordinate", [])
            key = tuple(round(c) for c in coord) if coord else None
            if key:
                layout_scores[key] = float(box.get("score", 0.0))
                layout_cls_ids[key] = box.get("cls_id")

        for block in parsing_blocks:
            label = block.get("block_label", "text")
            element_type = LABEL_MAP.get(label, "printed_text")
            text_content = block.get("block_content", "")

            bbox_raw = block.get("block_bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, str):
                import ast
                bbox_raw = ast.literal_eval(bbox_raw)
            coord = [float(c) for c in bbox_raw]
            normalized_bbox = normalize_bbox(coord, "pixel_topleft")

            # Look up confidence from layout detection
            coord_key = tuple(round(c) for c in coord)
            score = layout_scores.get(coord_key, 0.85)

            metadata = {
                "coordinate_system": "pixel_300dpi_topleft",
                "paddleocr_label": label,
                "block_order": block.get("block_order"),
            }
            cls_id = layout_cls_ids.get(coord_key)
            if cls_id is not None:
                metadata["paddleocr_cls_id"] = cls_id

            table_structure = None
            if element_type == "table":
                table_structure = {"has_html": False}
                # block_content for tables may contain markdown/html
                if text_content and "|" in text_content:
                    metadata["table_markdown"] = text_content

            region = Region(
                id=f"r_{counter:03d}",
                element_type=element_type,
                bounding_box=normalized_bbox,
                confidence=score,
                text_content=text_content,
                table_structure=table_structure,
                metadata=metadata,
            )
            regions.append(region)
            counter += 1

    return EngineOutput(page=page_num, engine="paddleocr", regions=regions)


def recognize_handwriting(ocr_v5, region_image_bytes: bytes) -> tuple[str, float]:
    """Run PP-OCRv5 recognition-only on a pre-cropped handwriting region.

    Uses det=False to skip detection since regions are already localized
    by PaddleOCR's layout classifier. Per GitHub issue #15603, PP-OCRv5
    has a regression on tightly cropped images with detection enabled.

    Args:
        ocr_v5: PaddleOCR instance configured for PP-OCRv5 recognition.
        region_image_bytes: PNG bytes of the cropped handwriting region.

    Returns:
        (text, confidence) tuple. Empty string and 0.0 on failure.
    """
    import numpy as np

    img = Image.open(io.BytesIO(region_image_bytes)).convert("RGB")
    img_array = np.array(img)

    result = ocr_v5.ocr(img_array, cls=False, det=False, rec=True)

    if not result or not result[0]:
        return "", 0.0

    text_parts = []
    confidences = []
    for line in result[0]:
        text, conf = line
        text_parts.append(text)
        confidences.append(conf)

    if not text_parts:
        return "", 0.0

    return " ".join(text_parts), sum(confidences) / len(confidences)


@app.cls(
    gpu="A10G",
    image=paddleocr_image,
    volumes={MODELS_DIR: model_volume},
    timeout=300,
    startup_timeout=600,
    min_containers=0,
    max_containers=10,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
class PaddleOCREngine:
    @modal.enter()
    def load_model(self):
        """Load PP-StructureV3 and PP-OCRv5 on container start -- runs once per container lifecycle."""
        from paddleocr import PPStructureV3, PaddleOCR
        logger.info("Loading PP-StructureV3 model...")
        self.pipeline = PPStructureV3(device="gpu:0")
        logger.info("PP-StructureV3 model loaded.")
        logger.info("Loading PP-OCRv5 for handwriting recognition...")
        self.ocr_v5 = PaddleOCR(lang="en", ocr_version="PP-OCRv5")
        logger.info("PP-OCRv5 model loaded.")

    @modal.method()
    def run(self, page_image: bytes, page_num: int = 0) -> dict:
        """Process a single page image. Returns EngineOutput as dict."""
        result = extract_page(self.pipeline, page_image, page_num)
        return result.model_dump()

    @modal.method()
    def run_handwriting(self, region_image: bytes) -> dict:
        """Run PP-OCRv5 recognition on a handwriting region.

        Returns {"text": str, "confidence": float}.
        """
        text, confidence = recognize_handwriting(self.ocr_v5, region_image)
        return {"text": text, "confidence": confidence}

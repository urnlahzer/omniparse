"""Docling hierarchical structure extraction engine (GPU, L4).

Processes full PDFs for hierarchical structure, table recovery, and reading order.
Docling works best on the full PDF (not per-page images) because its reading order
analysis benefits from multi-page context.

Coordinate system: Docling uses BOTTOMLEFT origin (PDF convention, Y increases upward).
Bounding boxes are converted to 300 DPI pixel top-left coordinates via normalize_bbox.

GPU/CPU fallback: Tries CUDA first. If CUDA detection fails (known issues #2528, #2292),
falls back to CPU mode within the same GPU container image.

Pattern: pure function (extract_pages) + Modal class wrapper (DoclingEngine).
"""
import io
import logging

import modal

from omniparse.app import app, docling_image, model_volume, MODELS_DIR
from omniparse.models.region import Region, EngineOutput, VALID_ELEMENT_TYPES
from omniparse.normalization import normalize_bbox

logger = logging.getLogger(__name__)

# Docling label -> canonical element_type mapping.
# All values MUST be in VALID_ELEMENT_TYPES.
# Original Docling label preserved in metadata["docling_label"].
DOCLING_LABEL_MAP = {
    "text": "printed_text",
    "section_header": "header",
    "title": "header",
    "table": "table",
    "picture": "image",
    "formula": "formula",
    "list_item": "printed_text",
    "caption": "printed_text",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "footer",
}

# Fixed confidence for Docling regions (Docling does not expose per-region confidence).
DOCLING_CONFIDENCE = 0.95


def _make_source(pdf_bytes: bytes):
    """Create a DocumentStream from PDF bytes. Separated for testability."""
    from docling.datamodel.base_models import DocumentStream
    return DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))


def extract_pages(
    converter,
    pdf_bytes: bytes,
    page_heights: dict[int, float],
) -> dict[int, EngineOutput]:
    """Extract hierarchical structure from a PDF using Docling.

    Processes the full PDF at once (Docling works best this way for reading order).
    Returns a dict mapping zero-indexed page numbers to EngineOutput.

    Args:
        converter: Initialized DocumentConverter instance (loaded in @modal.enter).
        pdf_bytes: Raw PDF bytes.
        page_heights: Dict mapping zero-indexed page number to page height in pixels
                      (from PagePayload.height). Required for coordinate conversion.

    Returns:
        Dict[int, EngineOutput] mapping page numbers to their extracted output.
    """
    if not pdf_bytes:
        return {}

    source = _make_source(pdf_bytes)
    result = converter.convert(source)
    doc = result.document

    # Collect regions per page
    page_regions: dict[int, list[Region]] = {}
    page_counters: dict[int, int] = {}

    for item, level in doc.iterate_items():
        if not hasattr(item, "prov") or not item.prov:
            continue

        prov = item.prov[0]
        page_num = prov.page_no - 1  # Docling uses 1-indexed pages

        if page_num not in page_regions:
            page_regions[page_num] = []
            page_counters[page_num] = 1

        bbox = prov.bbox
        page_height = page_heights.get(page_num)

        if page_height is not None:
            bbox_native = [bbox.l, bbox.b, bbox.r, bbox.t]
            normalized = normalize_bbox(
                bbox_native,
                "docling_bottomleft",
                page_height=page_height,
            )
        else:
            logger.warning(
                "No page_height for page %d; using raw Docling coordinates",
                page_num,
            )
            normalized = [round(bbox.l, 2), round(bbox.t, 2),
                          round(bbox.r, 2), round(bbox.b, 2)]

        # Determine label and element_type
        label = str(getattr(item, "label", "text"))
        if "." in label:
            label = label.split(".")[-1].lower()
        element_type = DOCLING_LABEL_MAP.get(label, "printed_text")

        # Extract text content
        text_content = ""
        if hasattr(item, "text"):
            text_content = item.text or ""
        elif hasattr(item, "export_to_markdown"):
            try:
                text_content = item.export_to_markdown(doc) or ""
            except TypeError:
                text_content = item.export_to_markdown() or ""

        metadata = {
            "coordinate_system": "pixel_300dpi_topleft",
            "docling_label": label,
            "hierarchy_level": level,
            "docling_item_type": type(item).__name__,
        }

        # Table structure metadata
        table_structure = None
        if element_type == "table" and hasattr(item, "export_to_markdown"):
            try:
                table_md = item.export_to_markdown(doc)
            except TypeError:
                table_md = item.export_to_markdown()
            if table_md:
                metadata["table_markdown"] = table_md
                lines = [ln for ln in table_md.strip().split("\n") if ln.strip()]
                num_rows = max(0, len(lines) - 1)
                num_cols = lines[0].count("|") - 1 if lines else 0
                table_structure = {
                    "rows": num_rows,
                    "cols": max(0, num_cols),
                }

        counter = page_counters[page_num]
        region = Region(
            id=f"r_{counter:03d}",
            element_type=element_type,
            bounding_box=normalized,
            confidence=DOCLING_CONFIDENCE,
            text_content=text_content,
            table_structure=table_structure,
            metadata=metadata,
        )
        page_regions[page_num].append(region)
        page_counters[page_num] = counter + 1

    # Build EngineOutput per page
    outputs: dict[int, EngineOutput] = {}
    for page_num, regions in page_regions.items():
        outputs[page_num] = EngineOutput(
            page=page_num, engine="docling", regions=regions
        )

    return outputs


@app.cls(
    gpu="L4",
    image=docling_image,
    volumes={MODELS_DIR: model_volume},
    timeout=600,
    startup_timeout=600,
    min_containers=0,
    max_containers=10,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
class DoclingEngine:
    @modal.enter()
    def load_model(self):
        """Load Docling DocumentConverter on container start.

        Uses pypdfium2 backend instead of docling-parse to avoid known crashes
        on PDFs with NULL font references or missing page dimensions
        (docling-parse#160, docling-serve#421).

        OCR enabled with force_full_page_ocr=True to handle scanned PDFs.
        This bypasses get_bitmap_rects (which crashes on image-only pages)
        and instead OCRs entire pages (docling#2182, docling#2038).

        Tries GPU acceleration if available. Falls back to CPU mode.
        """
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        pipeline_options = PdfPipelineOptions(do_ocr=True)
        pipeline_options.ocr_options.force_full_page_ocr = True

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            ),
        }

        if has_cuda:
            try:
                from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
                accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
                self.converter = DocumentConverter(
                    accelerator_options=accel,
                    format_options=format_options,
                )
                self._device = "cuda"
                logger.info("Docling loaded: CUDA + pypdfium2 + OCR (force_full_page)")
                return
            except (TypeError, ImportError):
                pass

        self.converter = DocumentConverter(format_options=format_options)
        self._device = "cpu"
        logger.info("Docling loaded: CPU + pypdfium2 + OCR (force_full_page)")

    @modal.method()
    def run(self, pdf_bytes: bytes, page_heights: dict) -> dict:
        """Process a full PDF. Returns dict mapping page_num -> EngineOutput dict."""
        heights = {int(k): float(v) for k, v in page_heights.items()}
        results = extract_pages(self.converter, pdf_bytes, heights)
        return {page_num: eo.model_dump() for page_num, eo in results.items()}

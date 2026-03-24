"""Bounding box normalization -- converts engine-native coordinates to 300 DPI pixel top-left origin.

Common coordinate system: 300 DPI pixel coordinates, top-left origin.
- PaddleOCR: already outputs pixel coords at input image DPI (300 from preprocessor). Zero conversion.
- Docling: outputs l/t/r/b with BOTTOMLEFT origin. Needs Y-axis flip.
- pdfplumber: outputs PDF points (72 DPI). Needs scale by dpi/72.

Each engine calls normalize_bbox() on its regions before returning EngineOutput.
"""

SUPPORTED_SYSTEMS = {"pdf_points_topleft", "pixel_topleft", "docling_bottomleft"}


def normalize_bbox(
    bbox: list[float],
    source_system: str,
    dpi: int = 300,
    page_height: float | None = None,
) -> list[float]:
    """Convert engine-native bounding box to 300 DPI pixel coordinates, top-left origin.

    Args:
        bbox: [x1, y1, x2, y2] in engine-native coordinates.
        source_system: One of "pdf_points_topleft", "pixel_topleft", "docling_bottomleft".
        dpi: Target DPI (default 300, matching preprocessor output).
        page_height: Page height in target pixels. Required for docling_bottomleft.

    Returns:
        [x1, y1, x2, y2] in 300 DPI pixel coordinates, top-left origin.

    Raises:
        ValueError: If source_system is not recognized or page_height missing for docling.
    """
    if source_system not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"Unknown coordinate system: {source_system!r}. "
            f"Supported: {SUPPORTED_SYSTEMS}"
        )

    if source_system == "pdf_points_topleft":
        scale = dpi / 72.0
        return [round(v * scale, 2) for v in bbox]

    if source_system == "pixel_topleft":
        return [round(v, 2) for v in bbox]

    if source_system == "docling_bottomleft":
        if page_height is None:
            raise ValueError("page_height required for docling_bottomleft conversion")
        l, b, r, t = bbox
        # Docling outputs PDF points (72 DPI). Scale to target DPI first, then flip Y.
        scale = dpi / 72.0
        return [
            round(l * scale, 2),
            round(page_height - t * scale, 2),
            round(r * scale, 2),
            round(page_height - b * scale, 2),
        ]

    raise ValueError(f"Unhandled system: {source_system}")


def normalize_to_unit(
    bbox: list[float], page_width: float, page_height: float
) -> list[float]:
    """Convert 300 DPI pixel [x1,y1,x2,y2] to [0,1] normalized coordinates.

    No rounding -- float64 precision preserved for downstream IoU (per D-04).
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Page dimensions must be positive: {page_width}x{page_height}")
    return [
        bbox[0] / page_width,
        bbox[1] / page_height,
        bbox[2] / page_width,
        bbox[3] / page_height,
    ]


def unit_to_pixel(
    bbox_norm: list[float], page_width: float, page_height: float
) -> list[float]:
    """Convert [0,1] normalized coordinates back to pixel coordinates.

    Uses round() (banker's rounding) not int() (truncation).
    """
    return [
        round(bbox_norm[0] * page_width),
        round(bbox_norm[1] * page_height),
        round(bbox_norm[2] * page_width),
        round(bbox_norm[3] * page_height),
    ]

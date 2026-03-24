"""Preprocessing pipeline for document normalization.

Accepts PDF, PNG, JPG, JPEG, and TIFF files. Normalizes DPI, corrects skew,
rotates landscape pages, and chunks multi-page PDFs into per-page PagePayload
objects.

Processing order per page: render/open -> DPI normalize -> de-skew ->
landscape rotate -> encode to PNG -> create PagePayload.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes

from omniparse.models.page import PagePayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_DPI = 200
TARGET_DPI = 300
MAX_SKEW_DEGREES = 15
ACCEPTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(file_bytes: bytes, filename: str) -> list[PagePayload]:
    """Preprocess a document into normalized PagePayload objects.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (extension determines format handling).

    Returns:
        One PagePayload per page. Corrupted pages have error field set.

    Raises:
        ValueError: If the file extension is not in ACCEPTED_EXTENSIONS.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")

    if ext == ".pdf":
        return _preprocess_pdf(file_bytes)
    else:
        return _preprocess_image(file_bytes)


# ---------------------------------------------------------------------------
# PDF handling (PREP-04)
# ---------------------------------------------------------------------------

def _preprocess_pdf(file_bytes: bytes) -> list[PagePayload]:
    """Render PDF pages and process each one individually."""
    try:
        images = convert_from_bytes(
            file_bytes,
            dpi=TARGET_DPI,
            fmt="png",
            thread_count=4,
        )
    except Exception as e:
        logger.error("Failed to render PDF: %s", e)
        return [
            PagePayload(
                page_num=0,
                image_bytes=b"",
                pdf_bytes=file_bytes,
                dpi=0,
                width=0,
                height=0,
                error=f"PDF render failed: {e}",
            )
        ]

    results: list[PagePayload] = []
    for page_num, img in enumerate(images):
        try:
            payload = _process_single_page(
                img=img,
                page_num=page_num,
                source_dpi=TARGET_DPI,
                pdf_bytes=file_bytes,
            )
            results.append(payload)
        except Exception as e:
            logger.error("Failed to process PDF page %d: %s", page_num, e)
            results.append(
                PagePayload(
                    page_num=page_num,
                    image_bytes=b"",
                    pdf_bytes=file_bytes,
                    dpi=0,
                    width=0,
                    height=0,
                    error=f"Page {page_num} processing failed: {e}",
                )
            )
    return results


# ---------------------------------------------------------------------------
# Image handling (PREP-01)
# ---------------------------------------------------------------------------

def _preprocess_image(file_bytes: bytes) -> list[PagePayload]:
    """Process a single image file (PNG, JPG, JPEG, TIFF)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # Force load to catch corrupted images early
    except Exception as e:
        logger.error("Failed to open image: %s", e)
        return [
            PagePayload(
                page_num=0,
                image_bytes=b"",
                dpi=0,
                width=0,
                height=0,
                error=f"Image open failed: {e}",
            )
        ]

    # Extract DPI from metadata (default 72 if not present)
    dpi_info = img.info.get("dpi", (72, 72))
    source_dpi = int(round(dpi_info[0]))

    try:
        payload = _process_single_page(
            img=img,
            page_num=0,
            source_dpi=source_dpi,
            pdf_bytes=None,
        )
        return [payload]
    except Exception as e:
        logger.error("Failed to process image: %s", e)
        return [
            PagePayload(
                page_num=0,
                image_bytes=b"",
                dpi=0,
                width=0,
                height=0,
                error=f"Image processing failed: {e}",
            )
        ]


# ---------------------------------------------------------------------------
# Per-page processing pipeline
# ---------------------------------------------------------------------------

def _process_single_page(
    img: Image.Image,
    page_num: int,
    source_dpi: int,
    pdf_bytes: bytes | None,
) -> PagePayload:
    """Process a single page through the normalization pipeline.

    Pipeline: DPI normalize -> de-skew -> landscape rotate -> encode PNG.
    """
    # Ensure RGB mode for consistent processing
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Step 1: DPI normalization (PREP-02)
    img, current_dpi = _normalize_dpi(img, source_dpi)

    # Step 2: De-skew (PREP-03)
    img = _deskew(img)

    # Step 3: Landscape rotation (PREP-05)
    width, height = img.size
    was_rotated = False
    if width > height:
        img = img.rotate(90, expand=True)
        was_rotated = True

    # Encode to PNG
    width, height = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    return PagePayload(
        page_num=page_num,
        image_bytes=image_bytes,
        pdf_bytes=pdf_bytes,
        dpi=current_dpi,
        width=width,
        height=height,
        was_rotated=was_rotated,
    )


# ---------------------------------------------------------------------------
# DPI normalization (PREP-02)
# ---------------------------------------------------------------------------

def _normalize_dpi(img: Image.Image, source_dpi: int) -> tuple[Image.Image, int]:
    """Upscale image to TARGET_DPI if source DPI is below MIN_DPI.

    Returns:
        Tuple of (processed image, effective DPI).
    """
    if source_dpi >= MIN_DPI:
        return img, source_dpi

    scale = TARGET_DPI / source_dpi
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return img, TARGET_DPI


# ---------------------------------------------------------------------------
# De-skew (PREP-03)
# ---------------------------------------------------------------------------

def _detect_skew_angle(image_array: np.ndarray) -> float:
    """Detect skew angle using OpenCV minAreaRect on text contours.

    Returns 0.0 if skew is undetectable or beyond MAX_SKEW_DEGREES threshold.
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Threshold to get binary image (text regions become white)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Find non-zero coordinates (text pixels)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0  # Not enough content to determine skew

    # Get minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle to [-45, 45] range
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct within threshold
    if abs(angle) > MAX_SKEW_DEGREES:
        return 0.0  # Beyond threshold -- don't correct

    return angle


def _deskew(img: Image.Image) -> Image.Image:
    """Detect and correct skew in an image.

    Skew beyond MAX_SKEW_DEGREES or below 0.1 degrees is ignored.
    """
    image_array = np.array(img)
    angle = _detect_skew_angle(image_array)

    if abs(angle) < 0.1:
        return img  # No significant skew

    h, w = image_array.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_array,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)

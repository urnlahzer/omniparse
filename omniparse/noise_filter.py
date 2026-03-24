"""Pre-alignment noise filter — removes formatting artifacts before cross-engine matching.

Filters out regions that are layout noise rather than document content:
- Court line numbers: tiny digit-only regions in the left margin
- Page numbers: tiny regions at the bottom of the page
- Footer fragments: small bottom-of-page regions already typed as footer

These inflate the single-engine count without contributing useful content.
The Markdown compiler already skips footer/page_number types (SKIP_TYPES),
but Docling often mislabels line numbers as printed_text.

Does NOT filter:
- Small addresses, signatures, or annotations (larger than line numbers)
- Margin handwriting (different element_type)
- Any region with substantial text content
"""
import re

from omniparse.models.region import Region


def filter_noise_regions(regions: list[Region]) -> list[Region]:
    """Remove formatting noise from a list of regions.

    Requires bounding_box_norm to be set (called after [0,1] normalization).
    Regions without bounding_box_norm are kept (safe fallback).

    Args:
        regions: List of Region objects with bounding_box_norm populated.

    Returns:
        Filtered list with noise regions removed.
    """
    return [r for r in regions if not _is_noise(r)]


def _is_noise(region: Region) -> bool:
    """Classify a region as noise based on position, size, and content.

    Rules:
    1. Page-bottom noise: y > 93% of page AND normalized area < 0.003
       (catches page numbers and footer fragments at bottom of every page)
    2. Margin line numbers: x < 12% of page AND normalized area < 0.001
       AND text is just 1-2 digits
       (catches court line numbers like "7", "14", "28")

    Normalized area = width_norm * height_norm (in [0,1]^2 space).
    10K sq px on a 2550x3300 page ≈ 0.00119 normalized area.
    3K sq px ≈ 0.000357 normalized area.
    """
    bbox = region.bounding_box_norm
    if bbox is None:
        return False  # Can't classify without normalized coords

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h

    # Rule 1: Page-bottom noise (page numbers, footer fragments)
    if y1 > 0.93 and area < 0.003:
        return True

    # Rule 2: Margin line numbers (court formatting)
    if x1 < 0.12 and area < 0.001 and _is_line_number(region.text_content):
        return True

    return False


def _is_line_number(text: str) -> bool:
    """Check if text is just a line number (1-2 digits, optionally with period)."""
    stripped = text.strip()
    return bool(re.match(r"^\d{1,2}\.?$", stripped))

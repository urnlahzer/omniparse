"""Markdown compilation module -- transforms ConsensusResult to GFM Markdown.

Converts the consensus pipeline output into a single GFM Markdown document with:
- YAML frontmatter (title, pages, processed timestamp)
- Preserved header hierarchy (H1-H6) from Docling's hierarchy_level
- GFM pipe-tables for simple tables, HTML fallback for colspan/rowspan
- LaTeX formula rendering (inline $ and display $$)
- Inline SVG chart blocks (sanitized for XSS)
- Handwriting annotations (inline italic or margin notes)
- HITL review flags as HTML comments
- Page breaks as horizontal rules (---)

Reading order: Docling's hierarchical reading order is authoritative (from ConsensusResult.reading_order).
Fallback: spatial ordering (top-to-bottom, left-to-right) when no reading order available.

Pattern: pure functions, no Modal dependency.
"""
import logging
import re
from datetime import datetime, timezone

import yaml

from omniparse.models.consensus import AlignedRegion, ConsensusResult

logger = logging.getLogger(__name__)

# Element types to skip in Markdown output (not meaningful content)
SKIP_TYPES = {"footer", "page_number", "seal", "image"}

# Maximum header level (H6)
MAX_HEADER_LEVEL = 6

# Default page width at 300 DPI (letter size: 8.5" x 300 = 2550 pixels)
DEFAULT_PAGE_WIDTH = 2550.0


def classify_handwriting_position(
    region_bbox: list[float],
    page_width: float = DEFAULT_PAGE_WIDTH,
) -> str:
    """Classify a handwriting region as marginal or inline based on its position.

    Args:
        region_bbox: [x1, y1, x2, y2] bounding box coordinates.
        page_width: Page width in pixels (default 2550.0 for 300 DPI letter).

    Returns:
        "marginal" if in left/right margin area, "inline" otherwise.
    """
    x1, _, x2, _ = region_bbox
    left_margin = page_width * 0.15
    right_margin = page_width * 0.85

    if x1 < left_margin or x2 > right_margin:
        return "marginal"
    return "inline"


def sanitize_svg(svg_text: str) -> str:
    """Sanitize SVG content by removing script tags and event handlers.

    Defends against XSS in user-uploaded or engine-generated SVG content.

    Args:
        svg_text: Raw SVG string.

    Returns:
        Cleaned SVG string with scripts and event handlers removed.
    """
    # Remove <script>...</script> tags (case-insensitive, multiline)
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", svg_text, flags=re.DOTALL | re.IGNORECASE)

    # Remove on* event handler attributes (onclick, onload, onerror, etc.)
    cleaned = re.sub(r'\s+on\w+\s*=\s*"[^"]*"', "", cleaned)
    cleaned = re.sub(r"\s+on\w+\s*=\s*'[^']*'", "", cleaned)

    return cleaned


def hitl_comment(region: AlignedRegion, page_num: int) -> str:
    """Render an HITL review flag as an HTML comment.

    Format: <!-- REVIEW NEEDED: [x1,y1,x2,y2] page=N confidence=X.XX -->
    """
    bbox_str = ",".join(f"{v}" for v in region.bounding_box)
    return f"<!-- REVIEW NEEDED: [{bbox_str}] page={page_num} confidence={region.confidence:.2f} -->"


def region_to_gfm_table(region: AlignedRegion) -> str:
    """Convert a table region to GFM pipe-table or HTML fallback.

    Priority:
    1. Docling table_markdown metadata -> return directly
    2. PaddleOCR table_html with colspan/rowspan -> HTML passthrough
    3. consensus_text with tab-separated rows -> pipe-table
    4. Nothing usable -> empty string
    """
    metadata = region.metadata or {}

    # 1. Docling table_markdown (pre-formatted GFM)
    if "table_markdown" in metadata:
        return metadata["table_markdown"]

    # 2. PaddleOCR table_html with complex structure (colspan/rowspan)
    if "table_html" in metadata:
        html = metadata["table_html"]
        if "colspan" in html or "rowspan" in html:
            return html

    # 3. Build pipe-table from consensus_text
    if region.consensus_text:
        return _text_to_pipe_table(region.consensus_text)

    return ""


def _text_to_pipe_table(text: str) -> str:
    """Convert tab-separated text into a GFM pipe-table.

    Expected format: rows separated by newlines, columns by tabs.
    First row is treated as the header.
    """
    rows = [row.split("\t") for row in text.strip().split("\n") if row.strip()]
    if not rows:
        return ""

    # Normalize column count to the max across all rows
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    lines = []
    # Header row
    header = "| " + " | ".join(cell.strip() for cell in rows[0]) + " |"
    lines.append(header)
    # Separator row
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    lines.append(separator)
    # Data rows
    for row in rows[1:]:
        data = "| " + " | ".join(cell.strip() for cell in row) + " |"
        lines.append(data)

    return "\n".join(lines)


def region_to_markdown(region: AlignedRegion, page_num: int) -> str:
    """Convert a single AlignedRegion to Markdown text.

    Handles: headers (H1-H6), printed_text, tables, HITL flags.
    Skips: footer, page_number, seal, image.
    """
    # Skip non-content element types
    if region.element_type in SKIP_TYPES:
        return ""

    content = ""

    if region.element_type == "table":
        content = region_to_gfm_table(region)
    elif region.element_type == "header":
        level = _extract_header_level(region)
        text = region.consensus_text or ""
        content = f"{'#' * level} {text}"
    elif region.element_type == "formula":
        latex = region.consensus_text or ""
        if latex:
            if "\n" in latex or len(latex) > 80:
                content = f"$${latex}$$"  # display mode
            else:
                content = f"${latex}$"    # inline mode
        else:
            content = f"[Formula: unreadable, see page {page_num + 1}]"
    elif region.element_type == "chart":
        svg = (region.metadata or {}).get("svg_content", "")
        if svg:
            content = sanitize_svg(svg)  # inline SVG block
        else:
            content = f"[Chart: extraction failed, see page {page_num + 1}]"
    elif region.element_type == "handwriting":
        text = region.consensus_text or ""
        page_width = (region.metadata or {}).get("page_width", DEFAULT_PAGE_WIDTH)
        position = classify_handwriting_position(region.bounding_box, page_width)
        if position == "marginal":
            bbox_str = ",".join(f"{v}" for v in region.bounding_box)
            content = f"*[Margin note: {text}]* <!-- handwritten-margin [{bbox_str}] -->"
        else:
            content = f"*{text}* <!-- handwritten -->"
    else:
        # printed_text and any unknown type
        content = region.consensus_text or ""

    # HITL flag handling
    if region.hitl_flag:
        comment = hitl_comment(region, page_num)
        if content:
            return f"{content}\n{comment}"
        else:
            return comment

    return content


def _extract_header_level(region: AlignedRegion) -> int:
    """Extract and clamp header hierarchy level from region metadata."""
    metadata = region.metadata or {}
    level = metadata.get("hierarchy_level", 2)
    return max(1, min(level, MAX_HEADER_LEVEL))


def compile_page(result: ConsensusResult) -> str:
    """Compile a single page's ConsensusResult to Markdown.

    Iterates regions in reading_order. Skips region_ids not found in regions list.
    """
    # Build lookup by region_id
    region_map = {r.region_id: r for r in result.regions}

    parts = []
    for region_id in result.reading_order:
        region = region_map.get(region_id)
        if region is None:
            logger.warning("Region %s in reading_order not found, skipping", region_id)
            continue
        md = region_to_markdown(region, page_num=result.page)
        if md:
            parts.append(md)

    return "\n\n".join(parts)


def compile_document(pages: list[ConsensusResult], title: str | None = None) -> str:
    """Compile multiple pages into a single GFM Markdown document.

    Produces:
    - YAML frontmatter with title, pages count, processed timestamp
    - Page content separated by horizontal rules (---)
    """
    # Extract title from first header if not provided
    doc_title = title or _extract_title(pages) or "Untitled Document"

    # Build YAML frontmatter
    frontmatter = {
        "title": doc_title,
        "pages": len(pages),
        "processed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    header = f"---\n{fm_str}\n---\n"

    # Compile each page
    page_contents = [compile_page(page) for page in pages]

    # Join with page breaks
    body = "\n\n---\n\n".join(page_contents)

    return f"{header}\n{body}"


def _extract_title(pages: list[ConsensusResult]) -> str | None:
    """Extract title from the first header region on the first page."""
    if not pages:
        return None
    for region_id in pages[0].reading_order:
        for region in pages[0].regions:
            if region.region_id == region_id and region.element_type == "header":
                return region.consensus_text
    # Fallback: any header region on first page
    for region in pages[0].regions:
        if region.element_type == "header" and region.consensus_text:
            return region.consensus_text
    return None

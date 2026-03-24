"""pdfplumber text extraction engine (CPU).

Extracts text regions with character-level bounding boxes, font metadata,
and table structure from born-digital PDF pages. Outputs canonical
Region/EngineOutput conforming to the OmniParse data contract.

Coordinate system: top-left origin, PDF points (1/72 inch).
Uses pdfplumber's (x0, top, x1, bottom) fields -- NOT (x0, y0, x1, y1).
"""
import io
from typing import Optional

import pdfplumber

from omniparse.app import app, cpu_image
from omniparse.models.region import Region, EngineOutput
from omniparse.normalization import normalize_bbox


def _overlaps_table(line_top: float, line_bottom: float, line_x0: float, line_x1: float,
                    table_bboxes: list[tuple[float, float, float, float]]) -> bool:
    """Check if a text line overlaps with any detected table bounding box."""
    for tx0, ttop, tx1, tbottom in table_bboxes:
        # Vertical overlap: line's vertical range intersects table's vertical range
        v_overlap = line_top < tbottom and line_bottom > ttop
        # Horizontal overlap: line's horizontal range intersects table's horizontal range
        h_overlap = line_x0 < tx1 and line_x1 > tx0
        if v_overlap and h_overlap:
            return True
    return False


def _get_font_metadata(page, line_top: float, line_x0: float,
                       tolerance: float = 2.0) -> dict:
    """Get font metadata from the first character near the given position."""
    for char in page.chars:
        if (abs(char["top"] - line_top) <= tolerance
                and abs(char["x0"] - line_x0) <= tolerance
                and char["text"].strip()):
            fontname = char.get("fontname", "")
            return {
                "font_size": float(char["size"]),
                "bold": "Bold" in fontname or "bold" in fontname,
                "italic": "Italic" in fontname or "Oblique" in fontname,
                "fontname": fontname,
            }
    # Fallback: search for any char near this top position
    for char in page.chars:
        if abs(char["top"] - line_top) <= tolerance and char["text"].strip():
            fontname = char.get("fontname", "")
            return {
                "font_size": float(char["size"]),
                "bold": "Bold" in fontname or "bold" in fontname,
                "italic": "Italic" in fontname or "Oblique" in fontname,
                "fontname": fontname,
            }
    return {
        "font_size": 0.0,
        "bold": False,
        "italic": False,
        "fontname": "",
    }


def _group_words_into_lines(words: list[dict],
                            y_tolerance: float = 2.0) -> list[list[dict]]:
    """Group words into text lines based on vertical proximity.

    Words with similar 'top' values (within y_tolerance) are on the same line.
    """
    if not words:
        return []

    # Sort by top then x0
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]

    for word in sorted_words[1:]:
        if abs(word["top"] - current_line[0]["top"]) <= y_tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]

    if current_line:
        lines.append(current_line)

    return lines


def _format_table_text(data: list[list[Optional[str]]]) -> str:
    """Format table data as a pipe-delimited string."""
    rows = []
    for row in data:
        cells = [cell if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_page(pdf_bytes: bytes, page_num: int, is_pdf: bool = True) -> EngineOutput:
    """Extract text regions from a single PDF page via pdfplumber.

    Args:
        pdf_bytes: Raw bytes of the PDF (or image if is_pdf=False).
        page_num: Zero-indexed page number to extract.
        is_pdf: If False, input is an image (returns empty EngineOutput).

    Returns:
        EngineOutput with Region objects for text lines and tables.
    """
    # Early return for non-PDF inputs (pdfplumber cannot extract text from images)
    if not is_pdf:
        return EngineOutput(page=page_num, engine="pdfplumber", regions=[])

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # Early return for out-of-range page numbers
        if page_num >= len(pdf.pages):
            return EngineOutput(page=page_num, engine="pdfplumber", regions=[])

        page = pdf.pages[page_num]
        page_w = float(page.width)
        page_h = float(page.height)

        regions: list[Region] = []
        counter = 1

        # --- Table detection ---
        tables = page.find_tables()
        table_bboxes: list[tuple[float, float, float, float]] = []
        table_regions: list[Region] = []

        for table in tables:
            bbox = table.bbox  # (x0, top, x1, bottom) -- top-left origin
            table_bboxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
            data = table.extract()
            num_rows = len(data) if data else 0
            num_cols = len(data[0]) if data and data[0] else 0

            # Table region will be added after text regions, but we prepare it now
            table_regions.append({
                "bbox": [
                    round(float(bbox[0]), 2),
                    round(float(bbox[1]), 2),
                    round(float(bbox[2]), 2),
                    round(float(bbox[3]), 2),
                ],
                "text": _format_table_text(data) if data else "",
                "rows": num_rows,
                "cols": num_cols,
            })

        # --- Text extraction (text-line level) ---
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )

        lines = _group_words_into_lines(words, y_tolerance=2.0)

        for line_words in lines:
            # Compute line bounding box
            line_x0 = min(w["x0"] for w in line_words)
            line_top = min(w["top"] for w in line_words)
            line_x1 = max(w["x1"] for w in line_words)
            line_bottom = max(w["bottom"] for w in line_words)

            # Skip text lines that overlap with table bounding boxes
            if _overlaps_table(line_top, line_bottom, line_x0, line_x1, table_bboxes):
                continue

            # Get font metadata from the first character of the line
            font_meta = _get_font_metadata(page, line_top, line_x0)

            text_content = " ".join(w["text"] for w in line_words)

            raw_bbox = [
                float(line_x0), float(line_top),
                float(line_x1), float(line_bottom),
            ]
            region = Region(
                id=f"r_{counter:03d}",
                element_type="printed_text",
                bounding_box=normalize_bbox(raw_bbox, "pdf_points_topleft"),
                confidence=1.0,
                text_content=text_content,
                metadata={
                    "font_size": font_meta["font_size"],
                    "bold": font_meta["bold"],
                    "italic": font_meta["italic"],
                    "fontname": font_meta["fontname"],
                    "coordinate_system": "pixel_300dpi_topleft",
                    "page_width": page_w,
                    "page_height": page_h,
                },
            )
            regions.append(region)
            counter += 1

        # --- Add table regions after text regions ---
        for tr in table_regions:
            region = Region(
                id=f"r_{counter:03d}",
                element_type="table",
                bounding_box=normalize_bbox(tr["bbox"], "pdf_points_topleft"),
                confidence=1.0,
                text_content=tr["text"],
                table_structure={
                    "rows": tr["rows"],
                    "cols": tr["cols"],
                },
                metadata={
                    "coordinate_system": "pixel_300dpi_topleft",
                    "page_width": page_w,
                    "page_height": page_h,
                },
            )
            regions.append(region)
            counter += 1

        return EngineOutput(page=page_num, engine="pdfplumber", regions=regions)


@app.function(image=cpu_image, timeout=60, min_containers=0, max_containers=10)
def run_pdfplumber(pdf_bytes: bytes, page_num: int) -> dict:
    """Modal-wrapped pdfplumber extraction. Returns EngineOutput as dict."""
    result = extract_page(pdf_bytes, page_num)
    return result.model_dump()

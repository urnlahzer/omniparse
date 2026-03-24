"""Canonical Region and EngineOutput schemas -- the data contract between ALL engines."""
from pydantic import BaseModel, Field
from typing import Optional

VALID_ELEMENT_TYPES = {
    "printed_text", "table", "handwriting", "formula",
    "chart", "image", "header", "footer", "page_number", "seal",
}


class Region(BaseModel):
    """Canonical region schema -- the data contract between ALL engines."""
    id: str = Field(description="Unique region ID, e.g., 'r_001'")
    element_type: str = Field(description="Region type: printed_text, table, handwriting, formula, chart, image, header, footer, page_number, seal")
    bounding_box: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2] in engine-native coordinates")
    bounding_box_norm: list[float] | None = Field(
        default=None, min_length=4, max_length=4,
        description="[x1, y1, x2, y2] normalized to [0,1] range (page-relative)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score. pdfplumber always 1.0.")
    text_content: str = Field(description="Extracted text content")
    table_structure: Optional[dict] = Field(default=None, description="Table metadata: rows, cols, has_merged_cells")
    metadata: Optional[dict] = Field(default=None, description="Engine-specific metadata")


class EngineOutput(BaseModel):
    """Output from a single engine for a single page."""
    page: int = Field(ge=0, description="Zero-indexed page number")
    engine: str = Field(description="Engine name: pdfplumber, paddleocr, docling, trocr, dots")
    regions: list[Region] = Field(default_factory=list)

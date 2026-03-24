"""Page payload and result models for engine input."""
from pydantic import BaseModel, Field
from typing import Optional


class PagePayload(BaseModel):
    """Input to engine functions after preprocessing."""
    page_num: int = Field(ge=0, description="Zero-indexed page number")
    image_bytes: bytes = Field(description="PNG-encoded page image")
    pdf_bytes: Optional[bytes] = Field(default=None, description="Original PDF bytes (for pdfplumber)")
    dpi: int = Field(ge=0, description="Image DPI (0 when error)")
    width: int = Field(ge=0, description="Image width in pixels (0 when error)")
    height: int = Field(ge=0, description="Image height in pixels (0 when error)")
    was_rotated: bool = Field(default=False, description="Whether landscape rotation was applied")
    error: Optional[str] = Field(default=None, description="Preprocessing error for this page")

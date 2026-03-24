"""Tests for Docling pre-merge — conservative same-line word fragment merging.

Validates merge criteria: vertical overlap + horizontal gap < 0.05 + same element_type.
Covers D-01 through D-04 from Phase 10 context.
"""
import pytest

from omniparse.docling_premerge import premerge_docling_regions
from omniparse.models.region import Region


def _region(
    id: str,
    element_type: str,
    bbox_norm: list[float],
    text: str,
    confidence: float = 0.9,
    metadata: dict | None = None,
) -> Region:
    """Helper to build a Region with [0,1] normalized coords."""
    return Region(
        id=id,
        element_type=element_type,
        bounding_box=bbox_norm,  # pixel coords same as norm for test simplicity
        bounding_box_norm=bbox_norm,
        confidence=confidence,
        text_content=text,
        metadata=metadata,
    )


class TestPremergeDoclingRegions:
    """Tests for premerge_docling_regions function."""

    def test_empty_input_returns_empty(self):
        """Empty input returns empty list."""
        assert premerge_docling_regions([]) == []

    def test_single_region_unchanged(self):
        """Single region passes through unchanged."""
        r = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "WHEREAS")
        result = premerge_docling_regions([r])
        assert len(result) == 1
        assert result[0].id == "r1"
        assert result[0].text_content == "WHEREAS"

    def test_merge_same_line_words(self):
        """Two words on the same line (overlapping Y, gap < 0.05, same type) merge."""
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "WHEREAS", confidence=0.9)
        r2 = _region("r2", "printed_text", [0.23, 0.15, 0.27, 0.17], "the", confidence=0.85)
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 1
        merged = result[0]
        # D-04: text = left-to-right concatenation
        assert merged.text_content == "WHEREAS the"
        # D-04: id = leftmost constituent
        assert merged.id == "r1"
        # D-04: confidence = min of constituents
        assert merged.confidence == 0.85
        # D-04: bbox = union
        assert merged.bounding_box_norm == [0.12, 0.15, 0.27, 0.17]
        assert merged.bounding_box == [0.12, 0.15, 0.27, 0.17]
        # element_type preserved
        assert merged.element_type == "printed_text"

    def test_no_merge_different_lines(self):
        """Two regions on different lines (no Y overlap) stay separate."""
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "Line one")
        r2 = _region("r2", "printed_text", [0.12, 0.25, 0.22, 0.27], "Line two")
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 2

    def test_no_merge_large_gap(self):
        """Two regions with horizontal gap > 0.05 stay separate."""
        r1 = _region("r1", "printed_text", [0.10, 0.15, 0.20, 0.17], "Left")
        r2 = _region("r2", "printed_text", [0.30, 0.15, 0.40, 0.17], "Right")
        # gap = 0.30 - 0.20 = 0.10, which is > 0.05
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 2

    def test_no_merge_different_types(self):
        """Two regions with different element_types stay separate."""
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "Text")
        r2 = _region("r2", "header", [0.23, 0.15, 0.33, 0.17], "Header")
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 2

    def test_three_regions_transitive(self):
        """Three words on the same line all merge transitively (A+B, B+C -> A+B+C)."""
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "WHEREAS", confidence=0.9)
        r2 = _region("r2", "printed_text", [0.23, 0.15, 0.27, 0.17], "the", confidence=0.85)
        r3 = _region("r3", "printed_text", [0.28, 0.15, 0.36, 0.17], "parties", confidence=0.88)
        result = premerge_docling_regions([r1, r2, r3])
        assert len(result) == 1
        merged = result[0]
        assert merged.text_content == "WHEREAS the parties"
        assert merged.id == "r1"
        assert merged.confidence == 0.85
        assert merged.bounding_box_norm == [0.12, 0.15, 0.36, 0.17]

    def test_mixed_merge_and_separate(self):
        """Some regions merge, others stay separate (different lines or types)."""
        # Line 1: two words that should merge
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "WHEREAS")
        r2 = _region("r2", "printed_text", [0.23, 0.15, 0.27, 0.17], "the")
        # Line 2: separate region on different line
        r3 = _region("r3", "printed_text", [0.12, 0.30, 0.30, 0.32], "Another line")
        # Line 1 but different type: stays separate
        r4 = _region("r4", "header", [0.50, 0.15, 0.70, 0.17], "Section Title")

        result = premerge_docling_regions([r1, r2, r3, r4])
        assert len(result) == 3
        # Find the merged region
        texts = sorted([r.text_content for r in result])
        assert "Another line" in texts
        assert "Section Title" in texts
        assert "WHEREAS the" in texts

    def test_merged_region_metadata_from_leftmost(self):
        """Merged region metadata comes from leftmost constituent (D-04)."""
        r1 = _region(
            "r1", "printed_text", [0.12, 0.15, 0.22, 0.17],
            "WHEREAS", metadata={"source": "docling", "font": "Times"}
        )
        r2 = _region(
            "r2", "printed_text", [0.23, 0.15, 0.27, 0.17],
            "the", metadata={"source": "docling", "font": "Arial"}
        )
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 1
        assert result[0].metadata == {"source": "docling", "font": "Times"}

    def test_gap_at_boundary(self):
        """Gap just above 0.05 does NOT merge (must be strictly less than 0.05)."""
        r1 = _region("r1", "printed_text", [0.10, 0.15, 0.20, 0.17], "Left")
        r2 = _region("r2", "printed_text", [0.26, 0.15, 0.35, 0.17], "Right")
        # gap = 0.26 - 0.20 = 0.06 > 0.05
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 2

    def test_overlapping_x_regions_merge(self):
        """Regions that overlap horizontally (negative gap) should merge if same line/type."""
        r1 = _region("r1", "printed_text", [0.10, 0.15, 0.22, 0.17], "WHEREAS")
        r2 = _region("r2", "printed_text", [0.20, 0.15, 0.30, 0.17], "the parties")
        result = premerge_docling_regions([r1, r2])
        assert len(result) == 1
        assert result[0].bounding_box_norm == [0.10, 0.15, 0.30, 0.17]

    def test_unsorted_input_still_merges(self):
        """Regions provided in non-sorted order still merge correctly."""
        r3 = _region("r3", "printed_text", [0.28, 0.15, 0.36, 0.17], "parties")
        r1 = _region("r1", "printed_text", [0.12, 0.15, 0.22, 0.17], "WHEREAS")
        r2 = _region("r2", "printed_text", [0.23, 0.15, 0.27, 0.17], "the")
        result = premerge_docling_regions([r3, r1, r2])
        assert len(result) == 1
        # Text should be left-to-right by x1
        assert result[0].text_content == "WHEREAS the parties"
        assert result[0].id == "r1"

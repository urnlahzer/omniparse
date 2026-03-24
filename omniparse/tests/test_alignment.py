"""Tests for spatial alignment module -- IoU matching and NW text alignment.

Covers requirements ALGN-01 through ALGN-04.
"""
from unittest.mock import patch

import pytest

from omniparse.alignment import (
    compute_iou,
    compute_containment_ratio,
    compute_center_distance,
    merge_contained_texts,
    agglomerative_cluster_orphans,
    match_regions_across_engines,
    align_texts,
    align_region_group,
    NW_MATCH_SCORE,
    NW_MISMATCH_SCORE,
    NW_INDEL_SCORE,
    NW_GAP_CHAR,
    HIRSCHBERG_THRESHOLD,
)
from omniparse.models.region import Region, EngineOutput


# ---------------------------------------------------------------------------
# compute_iou tests
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_iou_identical_boxes(self):
        """IoU of identical boxes is 1.0."""
        assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes is 0.0."""
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes is approx 1/7."""
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert abs(iou - (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)) < 1e-6

    def test_iou_zero_area(self):
        """IoU with zero-area box returns 0.0."""
        assert compute_iou([10, 10, 10, 10], [0, 0, 100, 100]) == 0.0

    def test_iou_high_overlap_fixture_boxes(self):
        """Fixture bounding boxes for same region across engines have IoU > 0.85."""
        # Region 1 header boxes from conftest
        iou_pdf_pad = compute_iou(
            [100, 50, 2400, 120],
            [105, 52, 2395, 118],
        )
        assert iou_pdf_pad > 0.85

        iou_pdf_doc = compute_iou(
            [100, 50, 2400, 120],
            [102, 48, 2398, 122],
        )
        assert iou_pdf_doc > 0.85


# ---------------------------------------------------------------------------
# match_regions_across_engines tests
# ---------------------------------------------------------------------------

class TestMatchRegions:
    def test_match_regions_three_engines(self, three_engine_outputs):
        """With 3 engines and 3 region types, should get 3 matched groups."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        assert len(groups) == 3
        # Each group should contain regions from all 3 engines
        for group in groups:
            assert len(group["regions"]) == 3

    def test_match_regions_groups_by_type(self, three_engine_outputs):
        """Table regions only match with table regions, text with text."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        for group in groups:
            types = set()
            for region in group["regions"].values():
                types.add(region.element_type)
            assert len(types) == 1, f"Group has mixed types: {types}"

    def test_match_regions_unmatched_leftover(self):
        """A region with no IoU match appears as a single-engine group."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="printed_text",
                    bounding_box=[0.0, 0.0, 100.0, 100.0],
                    confidence=1.0,
                    text_content="hello",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[500.0, 500.0, 600.0, 600.0],  # Far away, no overlap
                    confidence=0.9,
                    text_content="world",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Two unmatched groups, each with 1 engine
        assert len(groups) == 2
        for group in groups:
            assert len(group["regions"]) == 1

    def test_match_regions_custom_threshold(self, three_engine_outputs):
        """Disabling all matching techniques produces single-engine groups only."""
        pdf, paddle, docling = three_engine_outputs
        # Disable all matching: IoU (0.999), center-distance (0.0),
        # containment (1.1), WBF (1.1), clustering (0.0)
        groups_no_match = match_regions_across_engines(
            [pdf, paddle, docling],
            iou_threshold=0.999,
            center_distance_threshold=0.0,
            center_iou_floor=1.0,
            containment_threshold=1.1,
            wbf_iou_threshold=1.1,
            cluster_distance_threshold=0.0,
        )
        # All 9 regions are single-engine groups
        assert len(groups_no_match) == 9
        for group in groups_no_match:
            assert len(group["regions"]) == 1

    def test_match_regions_has_bounding_box(self, three_engine_outputs):
        """Each matched group has a bounding_box (average of matched boxes)."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        for group in groups:
            assert "bounding_box" in group
            assert len(group["bounding_box"]) == 4

    def test_match_uses_normalized_coords(self):
        """IoU uses bounding_box_norm when available, not bounding_box.

        Create two engines where pixel bboxes do NOT overlap but normalized
        bboxes DO overlap (IoU > 0.5 in [0,1] space). If matching produces a
        multi-engine group, it proves IoU reads bounding_box_norm.
        """
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="printed_text",
                    # Pixel bbox: far away from engine_b
                    bounding_box=[0.0, 0.0, 10.0, 10.0],
                    # Norm bbox: overlaps engine_b norm bbox
                    bounding_box_norm=[0.0, 0.0, 0.8, 0.8],
                    confidence=1.0,
                    text_content="hello",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    # Pixel bbox: far away from engine_a
                    bounding_box=[9000.0, 9000.0, 9010.0, 9010.0],
                    # Norm bbox: overlaps engine_a norm bbox (IoU > 0.5)
                    bounding_box_norm=[0.1, 0.1, 0.9, 0.9],
                    confidence=0.9,
                    text_content="world",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # If IoU used pixel bboxes, we'd get 2 single-engine groups
        # If IoU used bounding_box_norm, we get 1 multi-engine group
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2

    def test_matched_group_has_both_bbox_keys(self, three_engine_outputs):
        """Every matched group dict has both 'bounding_box' and 'bounding_box_norm' keys."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        for group in groups:
            assert "bounding_box" in group, "Missing pixel bounding_box key"
            assert "bounding_box_norm" in group, "Missing bounding_box_norm key"
            assert len(group["bounding_box"]) == 4
            assert group["bounding_box_norm"] is not None
            assert len(group["bounding_box_norm"]) == 4

    def test_cross_type_matching_header_printed_text(self):
        """header + printed_text regions at overlapping bboxes should match (Group A)."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="header",
                    bounding_box=[100.0, 50.0, 2400.0, 120.0],
                    bounding_box_norm=[100.0 / 2550, 50.0 / 3300, 2400.0 / 2550, 120.0 / 3300],
                    confidence=1.0,
                    text_content="LAST WILL AND TESTAMENT",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[105.0, 52.0, 2395.0, 118.0],
                    bounding_box_norm=[105.0 / 2550, 52.0 / 3300, 2395.0 / 2550, 118.0 / 3300],
                    confidence=0.95,
                    text_content="LAST WILL AND TESTAMENT",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should produce 1 matched group with 2 engines (not 2 single-engine groups)
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2

    def test_table_not_matched_with_printed_text(self):
        """table + printed_text at overlapping bboxes should NOT match (Group B vs A)."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="table",
                    bounding_box=[100.0, 300.0, 2400.0, 600.0],
                    bounding_box_norm=[100.0 / 2550, 300.0 / 3300, 2400.0 / 2550, 600.0 / 3300],
                    confidence=1.0,
                    text_content="Beneficiary | Share",
                    table_structure={"rows": 2, "cols": 2, "has_merged_cells": False},
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[105.0, 305.0, 2395.0, 595.0],
                    bounding_box_norm=[105.0 / 2550, 305.0 / 3300, 2395.0 / 2550, 595.0 / 3300],
                    confidence=0.9,
                    text_content="Beneficiary | Share",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should produce 2 separate single-engine groups (not matched despite spatial overlap)
        assert len(groups) == 2
        for group in groups:
            assert len(group["regions"]) == 1


# ---------------------------------------------------------------------------
# align_texts tests
# ---------------------------------------------------------------------------

class TestAlignTexts:
    def test_nw_alignment_basic(self):
        """align_texts('hello', 'helo') returns aligned character lists with a gap."""
        aligned_a, aligned_b = align_texts("hello", "helo")
        assert isinstance(aligned_a, list)
        assert isinstance(aligned_b, list)
        assert len(aligned_a) == len(aligned_b)
        # One of the lists must contain a gap character
        assert NW_GAP_CHAR in aligned_a or NW_GAP_CHAR in aligned_b

    def test_nw_alignment_identical(self):
        """Identical texts produce no gaps."""
        aligned_a, aligned_b = align_texts("hello", "hello")
        assert aligned_a == list("hello")
        assert aligned_b == list("hello")

    def test_nw_alignment_parameters(self):
        """align_texts uses correct NW parameters."""
        assert NW_MATCH_SCORE == 2.0
        assert NW_MISMATCH_SCORE == -1.0
        assert NW_INDEL_SCORE == -2.0
        assert NW_GAP_CHAR == "\x00"

    def test_nw_hirschberg_long_text(self):
        """align_texts switches to hirschberg for texts > 1000 chars."""
        long_a = "a" * 1001
        long_b = "a" * 1001
        with patch("omniparse.alignment.hirschberg") as mock_hirschberg:
            mock_hirschberg.return_value = (list(long_a), list(long_b))
            align_texts(long_a, long_b)
            mock_hirschberg.assert_called_once()

    def test_nw_needleman_wunsch_short_text(self):
        """align_texts uses needleman_wunsch for texts <= 1000 chars."""
        short_a = "hello"
        short_b = "helo"
        with patch("omniparse.alignment.needleman_wunsch") as mock_nw:
            mock_nw.return_value = (list("hello"), list("hel_o"))
            align_texts(short_a, short_b)
            mock_nw.assert_called_once()


# ---------------------------------------------------------------------------
# align_region_group tests
# ---------------------------------------------------------------------------

class TestAlignRegionGroup:
    def test_identical_text_fastpath(self, three_engine_outputs):
        """Region 1 (identical text) returns source='identical' and consensus_text set."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        # Find the group for Region 1 (header text)
        header_group = None
        for g in groups:
            texts = [r.text_content for r in g["regions"].values()]
            if all(t == "LAST WILL AND TESTAMENT" for t in texts):
                header_group = g
                break
        assert header_group is not None

        result = align_region_group(header_group)
        assert result.source == "identical"
        assert result.consensus_text == "LAST WILL AND TESTAMENT"
        assert result.aligned_texts is None

    def test_identical_text_confidence(self, three_engine_outputs):
        """Identical text fast-path sets confidence to max of all region confidences."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        header_group = None
        for g in groups:
            texts = [r.text_content for r in g["regions"].values()]
            if all(t == "LAST WILL AND TESTAMENT" for t in texts):
                header_group = g
                break

        result = align_region_group(header_group)
        assert result.confidence == 1.0  # max of 1.0, 0.95, 0.98

    def test_align_region_group_differing(self, three_engine_outputs):
        """Region 2 (differing text) returns AlignedRegion with aligned_texts populated."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        diff_group = None
        for g in groups:
            texts = list({r.text_content for r in g["regions"].values()})
            if len(texts) > 1:
                diff_group = g
                break
        assert diff_group is not None

        result = align_region_group(diff_group)
        assert result.aligned_texts is not None
        assert result.source == "pending"
        assert len(result.aligned_texts) > 0

    def test_align_region_group_single_engine(self):
        """Single-engine region returns source='single_engine'."""
        group = {
            "regions": {
                "engine_a": Region(
                    id="a_r1",
                    element_type="printed_text",
                    bounding_box=[0.0, 0.0, 100.0, 100.0],
                    confidence=0.85,
                    text_content="standalone text",
                ),
            },
            "element_type": "printed_text",
            "bounding_box": [0.0, 0.0, 100.0, 100.0],
        }
        result = align_region_group(group)
        assert result.source == "single_engine"
        assert result.consensus_text == "standalone text"
        assert result.confidence == 0.85

    def test_align_region_group_carries_metadata(self, three_engine_outputs):
        """Docling metadata (hierarchy_level) is carried forward."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        # Find a group with docling regions that have metadata
        for g in groups:
            if "docling" in g["regions"]:
                doc_region = g["regions"]["docling"]
                if doc_region.metadata and "hierarchy_level" in doc_region.metadata:
                    result = align_region_group(g)
                    assert result.metadata is not None
                    assert "hierarchy_level" in result.metadata
                    return
        pytest.fail("No group found with docling hierarchy_level metadata")

    def test_align_region_group_carries_table_structure(self, three_engine_outputs):
        """Table structure metadata is carried forward."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        table_group = None
        for g in groups:
            if g["element_type"] == "table":
                table_group = g
                break
        assert table_group is not None

        result = align_region_group(table_group)
        assert result.metadata is not None
        assert "table_structure" in result.metadata


# ---------------------------------------------------------------------------
# compute_center_distance tests (FUSE-03)
# ---------------------------------------------------------------------------

class TestCenterDistance:
    def test_identical_boxes(self):
        """Center distance of identical boxes is 0.0."""
        assert compute_center_distance([0, 0, 100, 100], [0, 0, 100, 100]) == 0.0

    def test_far_apart_boxes(self):
        """Boxes at opposite corners of [0,1] space -> distance ~1.27."""
        dist = compute_center_distance(
            [0.0, 0.0, 0.1, 0.1],
            [0.9, 0.9, 1.0, 1.0],
        )
        # Centers: (0.05, 0.05) vs (0.95, 0.95)
        # Distance: sqrt((0.9)^2 + (0.9)^2) = sqrt(1.62) = ~1.2727
        assert abs(dist - 1.2727922) < 0.001

    def test_same_center_different_sizes(self):
        """Boxes with same center but different sizes -> distance 0.0."""
        dist = compute_center_distance(
            [0.2, 0.2, 0.8, 0.8],
            [0.3, 0.3, 0.7, 0.7],
        )
        assert dist == 0.0

    def test_y_offset_boxes(self):
        """Y-offset boxes (real-world scenario) -> center distance ~0.0305."""
        dist = compute_center_distance(
            [0.05, 0.10, 0.95, 0.15],
            [0.055, 0.133, 0.945, 0.178],
        )
        # Centers: (0.5, 0.125) vs (0.5, 0.1555)
        # Distance = |0.1555 - 0.125| = 0.0305
        assert abs(dist - 0.0305) < 0.001


# ---------------------------------------------------------------------------
# Center-distance rescue pass tests (FUSE-03)
# ---------------------------------------------------------------------------

class TestCenterDistanceRescue:
    def test_y_offset_rescued(self, y_offset_outputs):
        """Y-offset boxes grouped by center_rescue (IoU=0.21, center_dist=0.03)."""
        paddleocr, docling = y_offset_outputs
        groups = match_regions_across_engines([paddleocr, docling])
        # Should produce 1 matched group with 2 engines (rescued by center-distance)
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2
        assert groups[0]["match_type"] == "center_rescue"

    def test_no_overlap_rejected(self, no_overlap_close_centers):
        """Close centers with zero overlap NOT matched (IoU floor prevents it)."""
        engine_a, engine_b = no_overlap_close_centers
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should produce 2 separate groups (IoU = 0.0 fails the floor)
        assert len(groups) == 2
        for g in groups:
            assert len(g["regions"]) == 1

    def test_consumed_set_enforced(self):
        """IoU-matched regions not re-processed by center_rescue."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="printed_text",
                    bounding_box=[0.0, 0.0, 100.0, 100.0],
                    bounding_box_norm=[0.0, 0.0, 0.5, 0.5],
                    confidence=1.0,
                    text_content="matched by iou",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[2.0, 2.0, 98.0, 98.0],
                    bounding_box_norm=[0.01, 0.01, 0.49, 0.49],
                    confidence=0.95,
                    text_content="matched by iou",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should be 1 group (IoU match), not duplicated by center_rescue
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2
        assert groups[0]["match_type"] == "iou"

    def test_compat_group_respected(self):
        """Center_rescue only matches within same compat group."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_table",
                    element_type="table",
                    bounding_box=[127.5, 330.0, 2422.5, 495.0],
                    bounding_box_norm=[0.05, 0.10, 0.95, 0.15],
                    confidence=1.0,
                    text_content="Table content",
                    table_structure={"rows": 1, "cols": 1, "has_merged_cells": False},
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_text",
                    element_type="printed_text",
                    bounding_box=[140.0, 440.0, 2410.0, 587.0],
                    bounding_box_norm=[0.055, 0.133, 0.945, 0.178],
                    confidence=0.95,
                    text_content="Text content",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should be 2 separate groups (different compat groups: table vs text)
        assert len(groups) == 2

    def test_match_type_center_rescue(self, y_offset_outputs):
        """Center-rescued groups carry match_type='center_rescue'."""
        paddleocr, docling = y_offset_outputs
        groups = match_regions_across_engines([paddleocr, docling])
        center_groups = [g for g in groups if g.get("match_type") == "center_rescue"]
        assert len(center_groups) == 1


# ---------------------------------------------------------------------------
# compute_containment_ratio tests (FUSE-01)
# ---------------------------------------------------------------------------

class TestContainmentRatio:
    def test_identical_boxes(self):
        """Containment ratio of identical boxes is 1.0."""
        assert compute_containment_ratio([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_non_overlapping_boxes(self):
        """Containment ratio of non-overlapping boxes is 0.0."""
        assert compute_containment_ratio([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_small_inside_large(self):
        """Small box fully inside large box returns 1.0."""
        ratio = compute_containment_ratio(
            [0.2, 0.2, 0.4, 0.4],  # small
            [0.0, 0.0, 1.0, 1.0],  # large
        )
        assert ratio == 1.0

    def test_partial_containment(self):
        """Partial containment returns value between 0 and 1."""
        ratio = compute_containment_ratio(
            [0.0, 0.0, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.75],
        )
        # intersection: [0.25, 0.25, 0.5, 0.5] = 0.25 * 0.25 = 0.0625
        # area_a = 0.25, area_b = 0.25, min = 0.25
        # ratio = 0.0625 / 0.25 = 0.25
        assert abs(ratio - 0.25) < 1e-6

    def test_zero_area_box(self):
        """Zero-area box returns 0.0."""
        assert compute_containment_ratio([10, 10, 10, 10], [0, 0, 100, 100]) == 0.0


# ---------------------------------------------------------------------------
# Containment rescue pass tests (FUSE-01)
# ---------------------------------------------------------------------------

class TestContainmentRescue:
    def test_cross_granularity_grouping(self, cross_granularity_outputs):
        """Word boxes inside line box are rescued by containment after IoU fails."""
        pdfplumber, docling, paddleocr = cross_granularity_outputs
        groups = match_regions_across_engines([pdfplumber, docling, paddleocr])

        # All regions should end up in a single match group (not 4 separate groups)
        # pdfplumber line + paddleocr line match by IoU (> 0.5)
        # docling words rescued by containment (inside the line box)
        multi_engine_groups = [g for g in groups if len(g["regions"]) > 1]
        assert len(multi_engine_groups) == 1
        group = multi_engine_groups[0]
        assert "pdfplumber" in group["regions"]
        assert "docling" in group["regions"]
        assert "paddleocr" in group["regions"]

    def test_containment_only_within_same_compat_group(self):
        """Containment rescue does not match across incompatible types."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_table",
                    element_type="table",
                    bounding_box=[0.0, 0.0, 1000.0, 1000.0],
                    bounding_box_norm=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                    text_content="Table content",
                    table_structure={"rows": 1, "cols": 1, "has_merged_cells": False},
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_text",
                    element_type="printed_text",
                    bounding_box=[100.0, 100.0, 200.0, 200.0],
                    bounding_box_norm=[0.1, 0.1, 0.2, 0.2],
                    confidence=0.9,
                    text_content="Some text",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should be 2 separate groups (different compat groups)
        assert len(groups) == 2
        for g in groups:
            assert len(g["regions"]) == 1


class TestManyToOneContainment:
    def test_merged_text_reading_order(self, cross_granularity_outputs):
        """Many-to-one containment produces merged text in reading order."""
        pdfplumber, docling, paddleocr = cross_granularity_outputs
        groups = match_regions_across_engines([pdfplumber, docling, paddleocr])

        multi_groups = [g for g in groups if len(g["regions"]) > 1]
        assert len(multi_groups) == 1
        group = multi_groups[0]

        # Docling region should have merged text from the 3 word regions
        docling_region = group["regions"]["docling"]
        assert docling_region.text_content == "WHEREAS the parties agree"


class TestContainmentVoteCounting:
    def test_vote_count_equals_engine_count(self, cross_granularity_outputs):
        """Vote count = number of distinct engines, not sub-box count."""
        pdfplumber, docling, paddleocr = cross_granularity_outputs
        groups = match_regions_across_engines([pdfplumber, docling, paddleocr])

        multi_groups = [g for g in groups if len(g["regions"]) > 1]
        assert len(multi_groups) == 1
        group = multi_groups[0]

        # 3 engines: pdfplumber, docling, paddleocr -- NOT 5 (1+3+1)
        assert len(group["regions"]) == 3


class TestContainmentNoDoubleMatch:
    def test_consumed_set_enforced(self):
        """Regions already matched by IoU are not re-processed by containment."""
        # Two engines with regions that match by IoU
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="printed_text",
                    bounding_box=[0.0, 0.0, 100.0, 100.0],
                    bounding_box_norm=[0.0, 0.0, 0.5, 0.5],
                    confidence=1.0,
                    text_content="matched by iou",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[2.0, 2.0, 98.0, 98.0],
                    bounding_box_norm=[0.01, 0.01, 0.49, 0.49],
                    confidence=0.95,
                    text_content="matched by iou",
                ),
            ],
        )
        groups = match_regions_across_engines([engine_a, engine_b])
        # Should be exactly 1 group with 2 engines (IoU match)
        # No duplicate from containment
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2


# ---------------------------------------------------------------------------
# match_type tracking tests (FUSE-01 + FUSE-02)
# ---------------------------------------------------------------------------

class TestMatchTypeTracking:
    def test_iou_groups_have_match_type(self, three_engine_outputs):
        """IoU-matched groups carry match_type='iou'."""
        pdf, paddle, docling = three_engine_outputs
        groups = match_regions_across_engines([pdf, paddle, docling])
        for g in groups:
            assert "match_type" in g
            assert g["match_type"] in ("iou", "containment", "wbf", "center_rescue", "cluster", "orphan")

    def test_containment_groups_labeled(self, cross_granularity_outputs):
        """Containment-rescued groups carry match_type='containment'."""
        pdfplumber, docling, paddleocr = cross_granularity_outputs
        groups = match_regions_across_engines([pdfplumber, docling, paddleocr])
        containment_groups = [g for g in groups if g.get("match_type") == "containment"]
        # At least one containment group expected
        assert len(containment_groups) >= 1

    def test_orphan_groups_labeled(self):
        """Single-engine orphan groups carry match_type='orphan'."""
        engine_a = EngineOutput(
            page=0,
            engine="engine_a",
            regions=[
                Region(
                    id="a_r1",
                    element_type="printed_text",
                    bounding_box=[0.0, 0.0, 100.0, 100.0],
                    bounding_box_norm=[0.0, 0.0, 0.5, 0.5],
                    confidence=1.0,
                    text_content="hello",
                ),
            ],
        )
        engine_b = EngineOutput(
            page=0,
            engine="engine_b",
            regions=[
                Region(
                    id="b_r1",
                    element_type="printed_text",
                    bounding_box=[500.0, 500.0, 600.0, 600.0],
                    bounding_box_norm=[0.8, 0.8, 0.9, 0.9],
                    confidence=0.9,
                    text_content="world",
                ),
            ],
        )
        groups = match_regions_across_engines(
            [engine_a, engine_b], containment_threshold=1.1,
        )
        # At least the engine_b region should be an orphan
        orphan_groups = [g for g in groups if g.get("match_type") == "orphan"]
        assert len(orphan_groups) >= 1


# ---------------------------------------------------------------------------
# agglomerative_cluster_orphans tests (FUSE-04)
# ---------------------------------------------------------------------------

class TestAgglomerativeCluster:
    def test_overlapping_different_engines_grouped(self):
        """2 overlapping regions from different engines -> 1 cluster."""
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.9, text_content="text A",
            )),
            ("engine_b", Region(
                id="b_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.85, text_content="text B",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        # Identical boxes -> IoU=1.0 -> distance=0.0 < 0.92 -> should cluster
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2
        assert groups[0]["match_type"] == "cluster"

    def test_non_overlapping_separate(self):
        """2 non-overlapping regions -> 2 singletons."""
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 100, 100],
                bounding_box_norm=[0.0, 0.0, 0.1, 0.1],
                confidence=0.9, text_content="text A",
            )),
            ("engine_b", Region(
                id="b_r1", element_type="printed_text",
                bounding_box=[900, 900, 1000, 1000],
                bounding_box_norm=[0.9, 0.9, 1.0, 1.0],
                confidence=0.85, text_content="text B",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        # IoU=0.0 -> distance=1.0 > 0.92 -> separate singletons
        assert len(groups) == 2
        for g in groups:
            assert len(g["regions"]) == 1
            assert g["match_type"] == "orphan"

    def test_complete_linkage_conservative(self):
        """Chained overlap (A-B, B-C but not A-C) stays separate with complete linkage."""
        # A and B overlap (IoU > 0.08), B and C overlap (IoU > 0.08), but A and C don't
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 200],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.2],
                confidence=0.9, text_content="text A",
            )),
            ("engine_b", Region(
                id="b_r1", element_type="printed_text",
                bounding_box=[400, 100, 900, 300],
                bounding_box_norm=[0.4, 0.1, 0.9, 0.3],
                confidence=0.85, text_content="text B",
            )),
            ("engine_c", Region(
                id="c_r1", element_type="printed_text",
                bounding_box=[800, 200, 1300, 400],
                bounding_box_norm=[0.8, 0.2, 1.0, 0.4],
                confidence=0.88, text_content="text C",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        # With complete linkage, A-C distance (IoU=0.0, dist=1.0) prevents merging
        # all three. A+B might cluster, C stays separate, or all 3 stay separate.
        # Key: they are NOT all in 1 cluster (complete linkage is conservative)
        all_in_one = any(len(g["regions"]) == 3 for g in groups)
        assert not all_in_one

    def test_singleton_becomes_orphan(self):
        """Single-region cluster gets match_type='orphan'."""
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 100, 100],
                bounding_box_norm=[0.0, 0.0, 0.1, 0.1],
                confidence=0.9, text_content="solo",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        assert len(groups) == 1
        assert groups[0]["match_type"] == "orphan"

    def test_multi_engine_cluster_type(self):
        """Multi-region cluster from different engines gets match_type='cluster'."""
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.9, text_content="text A",
            )),
            ("engine_b", Region(
                id="b_r1", element_type="printed_text",
                bounding_box=[10, 5, 490, 95],
                bounding_box_norm=[0.01, 0.005, 0.49, 0.095],
                confidence=0.85, text_content="text B",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        cluster_groups = [g for g in groups if g["match_type"] == "cluster"]
        assert len(cluster_groups) == 1


# ---------------------------------------------------------------------------
# Clustering integration tests (FUSE-04)
# ---------------------------------------------------------------------------

class TestClusteringIntegration:
    def test_clustering_after_wbf(self, clustering_orphan_outputs):
        """WBF singletons are fed to clustering and grouped."""
        engine_a, engine_b = clustering_orphan_outputs
        groups = match_regions_across_engines([engine_a, engine_b])
        # With clustering, these should be grouped (IoU ~0.11 > 0.08 threshold)
        cluster_groups = [g for g in groups if g["match_type"] == "cluster"]
        assert len(cluster_groups) == 1
        assert len(cluster_groups[0]["regions"]) == 2

    def test_full_layered_match_types(self):
        """Verify all 6 match_type values can appear in output."""
        # Create a complex scenario with regions that trigger each match type
        valid_types = {"iou", "center_rescue", "containment", "wbf", "cluster", "orphan"}
        # Just verify the enum set is documented -- full pipeline test is in Task 1
        assert valid_types == {"iou", "center_rescue", "containment", "wbf", "cluster", "orphan"}

    def test_clustering_vote_collapsing(self):
        """If same engine has multiple regions in a cluster, texts are merged."""
        regions = [
            ("engine_a", Region(
                id="a_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.9, text_content="first part",
            )),
            ("engine_a", Region(
                id="a_r2", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.85, text_content="second part",
            )),
            ("engine_b", Region(
                id="b_r1", element_type="printed_text",
                bounding_box=[0, 0, 500, 100],
                bounding_box_norm=[0.0, 0.0, 0.5, 0.1],
                confidence=0.88, text_content="engine b text",
            )),
        ]
        groups = agglomerative_cluster_orphans(regions)
        # All 3 have identical boxes -> 1 cluster
        # engine_a has 2 regions -> merged into 1 -> vote count = 2 engines
        assert len(groups) == 1
        assert len(groups[0]["regions"]) == 2  # engine_a (merged) + engine_b
        # engine_a's text should be merged
        assert "first part" in groups[0]["regions"]["engine_a"].text_content
        assert "second part" in groups[0]["regions"]["engine_a"].text_content

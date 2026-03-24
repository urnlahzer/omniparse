"""Tests for vendored WBF (Weighted Boxes Fusion) with provenance tracking.

Covers requirement FUSE-02.
"""
import numpy as np
import pytest

from omniparse.wbf import (
    _prefilter_boxes,
    _get_weighted_box,
    _find_matching_box_fast,
    weighted_boxes_fusion_with_provenance,
)
from omniparse.alignment import (
    _wbf_group_regions,
    ENGINE_WEIGHTS,
)
from omniparse.models.region import Region, EngineOutput


# ---------------------------------------------------------------------------
# _prefilter_boxes tests
# ---------------------------------------------------------------------------

class TestPrefilterBoxes:
    def test_builds_8col_array(self):
        """prefilter_boxes produces an 8-column array."""
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5]],
        ]
        scores_list = [[0.9]]
        labels_list = [[0]]
        weights = [1.0]
        filtered, orig_indices = _prefilter_boxes(
            boxes_list, scores_list, labels_list, weights, skip_box_thr=0.0,
        )
        assert filtered.shape[1] == 8
        assert len(orig_indices) == 1

    def test_clamps_coords(self):
        """Coordinates outside [0,1] are clamped."""
        boxes_list = [
            [[-0.1, -0.2, 1.3, 1.5]],
        ]
        scores_list = [[0.8]]
        labels_list = [[0]]
        weights = [1.0]
        filtered, _ = _prefilter_boxes(
            boxes_list, scores_list, labels_list, weights, skip_box_thr=0.0,
        )
        assert filtered[0, 4] >= 0.0  # x1
        assert filtered[0, 5] >= 0.0  # y1
        assert filtered[0, 6] <= 1.0  # x2
        assert filtered[0, 7] <= 1.0  # y2

    def test_filters_zero_area(self):
        """Zero-area boxes are filtered out."""
        boxes_list = [
            [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.5, 0.5]],  # first is zero-area
        ]
        scores_list = [[0.9, 0.8]]
        labels_list = [[0, 0]]
        weights = [1.0]
        filtered, orig_indices = _prefilter_boxes(
            boxes_list, scores_list, labels_list, weights, skip_box_thr=0.0,
        )
        assert len(filtered) == 1
        assert orig_indices[0] == (0, 1)  # only second box survived


# ---------------------------------------------------------------------------
# _find_matching_box_fast tests
# ---------------------------------------------------------------------------

class TestFindMatchingBoxFast:
    def test_finds_matching_box(self):
        """Finds best IoU match against existing cluster centroids."""
        # Create a cluster centroid at [0, 0.1, 0.1, 0.5, 0.5] (label=0)
        # new_box at similar position should match
        boxes = np.array([[0, 0.9, 1.0, 0, 0.1, 0.1, 0.5, 0.5]])
        new_box = np.array([0, 0.8, 1.0, 1, 0.12, 0.12, 0.48, 0.48])
        idx = _find_matching_box_fast(boxes, new_box, match_iou=0.3)
        assert idx == 0

    def test_no_match_below_threshold(self):
        """Returns -1 when no box exceeds IoU threshold."""
        boxes = np.array([[0, 0.9, 1.0, 0, 0.0, 0.0, 0.1, 0.1]])
        new_box = np.array([0, 0.8, 1.0, 1, 0.8, 0.8, 1.0, 1.0])
        idx = _find_matching_box_fast(boxes, new_box, match_iou=0.3)
        assert idx == -1

    def test_label_mismatch(self):
        """Same-position boxes with different labels do not match."""
        boxes = np.array([[0, 0.9, 1.0, 0, 0.1, 0.1, 0.5, 0.5]])  # label=0
        new_box = np.array([1, 0.8, 1.0, 1, 0.1, 0.1, 0.5, 0.5])  # label=1
        idx = _find_matching_box_fast(boxes, new_box, match_iou=0.3)
        assert idx == -1


# ---------------------------------------------------------------------------
# WBF clustering tests
# ---------------------------------------------------------------------------

class TestWBFClustering:
    def test_overlapping_boxes_cluster(self):
        """Two overlapping boxes from different engines form 1 cluster."""
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5]],  # engine 0
            [[0.12, 0.12, 0.48, 0.48]],  # engine 1 -- overlaps
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1.0, 1.0]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=0.3,
        )
        assert len(boxes) == 1
        assert len(clusters) == 1
        assert len(clusters[0]) == 2  # 2 contributing boxes

    def test_non_overlapping_stay_separate(self):
        """Two non-overlapping boxes remain as 2 separate clusters."""
        boxes_list = [
            [[0.0, 0.0, 0.1, 0.1]],  # engine 0 -- top-left
            [[0.8, 0.8, 1.0, 1.0]],  # engine 1 -- bottom-right
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1.0, 1.0]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=0.3,
        )
        assert len(boxes) == 2
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# WBF provenance tests
# ---------------------------------------------------------------------------

class TestWBFProvenance:
    def test_cluster_maps_to_source(self):
        """Provenance maps each cluster back to (model_idx, box_idx)."""
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5], [0.7, 0.7, 0.9, 0.9]],  # engine 0: 2 boxes
            [[0.12, 0.12, 0.48, 0.48]],  # engine 1: 1 box (overlaps first of engine 0)
        ]
        scores_list = [[0.9, 0.7], [0.8]]
        labels_list = [[0, 0], [0]]
        weights = [1.0, 1.0]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=0.3,
        )
        # Should have 2 clusters: one with 2 boxes (overlapping), one with 1 box
        assert len(clusters) == 2

        # Find the cluster with 2 members
        multi_cluster = [c for c in clusters if len(c) == 2]
        assert len(multi_cluster) == 1
        # Each entry is (model_idx, original_box_idx)
        model_indices = {m for m, _ in multi_cluster[0]}
        assert model_indices == {0, 1}  # From both engines


# ---------------------------------------------------------------------------
# WBF weights tests
# ---------------------------------------------------------------------------

class TestWBFWeights:
    def test_fixed_weights_applied(self):
        """Fixed engine weights affect fused confidence scores."""
        # Two boxes at same location, one from high-weight engine
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5]],
            [[0.12, 0.12, 0.48, 0.48]],
        ]
        scores_list = [[0.5], [0.5]]
        labels_list = [[0], [0]]

        # Without weighting
        boxes1, scores1, _, _ = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[1.0, 1.0], iou_thr=0.3,
        )

        # With weighting (engine 0 weighted 3x)
        boxes2, scores2, _, _ = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[3.0, 1.0], iou_thr=0.3,
        )

        # Higher weight shifts fused box toward engine 0's coords
        assert not np.allclose(boxes1[0], boxes2[0])

    def test_engine_weights_constant(self):
        """ENGINE_WEIGHTS has correct fixed values per D-10."""
        assert ENGINE_WEIGHTS["pdfplumber"] == 3.0
        assert ENGINE_WEIGHTS["paddleocr"] == 1.0
        assert ENGINE_WEIGHTS["docling"] == 1.0
        assert ENGINE_WEIGHTS["trocr"] == 2.0


# ---------------------------------------------------------------------------
# WBF edge cases
# ---------------------------------------------------------------------------

class TestWBFEdgeCases:
    def test_single_box_input(self):
        """Single box produces single cluster."""
        boxes_list = [[[0.1, 0.1, 0.5, 0.5]]]
        scores_list = [[0.9]]
        labels_list = [[0]]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[1.0], iou_thr=0.3,
        )
        assert len(boxes) == 1
        assert len(clusters) == 1
        assert clusters[0] == [(0, 0)]

    def test_all_same_model(self):
        """Multiple boxes from same model still cluster."""
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5], [0.12, 0.12, 0.48, 0.48]],
        ]
        scores_list = [[0.9, 0.8]]
        labels_list = [[0, 0]]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[1.0], iou_thr=0.3,
        )
        # Same-model overlapping boxes cluster together
        assert len(boxes) == 1

    def test_skip_box_thr_zero(self):
        """skip_box_thr=0.0 means no boxes are discarded."""
        boxes_list = [
            [[0.1, 0.1, 0.5, 0.5]],
            [[0.6, 0.6, 0.9, 0.9]],
        ]
        scores_list = [[0.01], [0.01]]  # Very low confidence
        labels_list = [[0], [0]]
        boxes, scores, labels, clusters = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[1.0, 1.0], iou_thr=0.3, skip_box_thr=0.0,
        )
        assert len(boxes) == 2  # Neither discarded

    def test_wbf_uses_unit_coords(self):
        """WBF operates on [0,1] coordinates per D-11."""
        # Coords within [0,1] -- should work fine
        boxes_list = [[[0.0, 0.0, 0.5, 0.5]]]
        scores_list = [[0.9]]
        labels_list = [[0]]
        boxes, _, _, _ = weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=[1.0], iou_thr=0.3,
        )
        # All coords should be in [0,1]
        assert np.all(boxes >= 0.0) and np.all(boxes <= 1.0)


# ---------------------------------------------------------------------------
# WBF integration with alignment wrapper
# ---------------------------------------------------------------------------

class TestWBFIntegration:
    def test_wbf_group_regions_basic(self):
        """_wbf_group_regions groups overlapping regions from different engines."""
        unmatched = [
            ("pdfplumber", Region(
                id="pdf_r1", element_type="printed_text",
                bounding_box=[255.0, 330.0, 1275.0, 1650.0],
                bounding_box_norm=[0.1, 0.1, 0.5, 0.5],
                confidence=1.0, text_content="Hello",
            )),
            ("paddleocr", Region(
                id="pad_r1", element_type="printed_text",
                bounding_box=[306.0, 396.0, 1224.0, 1584.0],
                bounding_box_norm=[0.12, 0.12, 0.48, 0.48],
                confidence=0.9, text_content="Hello",
            )),
        ]
        groups = _wbf_group_regions(unmatched)
        # Should produce 1 group with both engines
        multi_groups = [g for g in groups if len(g["regions"]) > 1]
        assert len(multi_groups) == 1
        assert "pdfplumber" in multi_groups[0]["regions"]
        assert "paddleocr" in multi_groups[0]["regions"]
        assert multi_groups[0]["match_type"] == "wbf"

    def test_wbf_preserves_original_text(self):
        """WBF groups preserve original engine text (grouping only per D-09)."""
        unmatched = [
            ("pdfplumber", Region(
                id="pdf_r1", element_type="printed_text",
                bounding_box=[255.0, 330.0, 1275.0, 1650.0],
                bounding_box_norm=[0.1, 0.1, 0.5, 0.5],
                confidence=1.0, text_content="Original pdfplumber text",
            )),
            ("paddleocr", Region(
                id="pad_r1", element_type="printed_text",
                bounding_box=[306.0, 396.0, 1224.0, 1584.0],
                bounding_box_norm=[0.12, 0.12, 0.48, 0.48],
                confidence=0.9, text_content="Original paddleocr text",
            )),
        ]
        groups = _wbf_group_regions(unmatched)
        multi_groups = [g for g in groups if len(g["regions"]) > 1]
        assert len(multi_groups) == 1
        assert multi_groups[0]["regions"]["pdfplumber"].text_content == "Original pdfplumber text"
        assert multi_groups[0]["regions"]["paddleocr"].text_content == "Original paddleocr text"

    def test_wbf_non_overlapping_separate(self):
        """Non-overlapping regions stay as separate single-engine groups."""
        unmatched = [
            ("pdfplumber", Region(
                id="pdf_r1", element_type="printed_text",
                bounding_box=[0.0, 0.0, 255.0, 330.0],
                bounding_box_norm=[0.0, 0.0, 0.1, 0.1],
                confidence=1.0, text_content="Far away",
            )),
            ("paddleocr", Region(
                id="pad_r1", element_type="printed_text",
                bounding_box=[2040.0, 2640.0, 2550.0, 3300.0],
                bounding_box_norm=[0.8, 0.8, 1.0, 1.0],
                confidence=0.9, text_content="Also far away",
            )),
        ]
        groups = _wbf_group_regions(unmatched)
        # Should produce 2 single-engine groups
        assert len(groups) == 2
        for g in groups:
            assert len(g["regions"]) == 1

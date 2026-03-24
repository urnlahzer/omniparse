# Vendored from ensemble-boxes v1.0.9 (MIT License)
# Original author: ZFTurbo (Roman Solovyev)
# Source: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
# Paper: Solovyev et al., "Weighted Boxes Fusion" (arXiv:1910.13302)
#
# Modifications:
# - Extracted WBF-only subset (removed NMS, Soft-NMS, NMW functions)
# - Added provenance tracking: returns cluster membership mapping
# - Removed print()+exit() error handling, replaced with ValueError
"""Vendored Weighted Boxes Fusion with provenance tracking.

Clusters overlapping bounding boxes from multiple models using confidence-weighted
averaging. Returns cluster membership mapping for provenance tracing back to
source engine Regions.

All coordinates must be in [0,1] normalized space.
"""
from __future__ import annotations

import numpy as np


def _prefilter_boxes(
    boxes_list: list[list[list[float]]],
    scores_list: list[list[float]],
    labels_list: list[list[int]],
    weights: list[float],
    skip_box_thr: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Validate, clamp, filter, and build 8-column array for WBF.

    8-column format: [label, score*weight, weight, model_index, x1, y1, x2, y2]

    Returns:
        filtered: ndarray of shape (N, 8) sorted by weighted score descending.
        orig_indices: list of (model_idx, box_idx_in_model) for provenance.
    """
    all_boxes = []
    orig_indices: list[tuple[int, int]] = []

    for model_idx, (boxes, scores, labels) in enumerate(
        zip(boxes_list, scores_list, labels_list)
    ):
        weight = weights[model_idx] if model_idx < len(weights) else 1.0

        for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score * weight < skip_box_thr:
                continue

            # Clamp coords to [0, 1]
            x1 = max(0.0, min(1.0, box[0]))
            y1 = max(0.0, min(1.0, box[1]))
            x2 = max(0.0, min(1.0, box[2]))
            y2 = max(0.0, min(1.0, box[3]))

            # Skip zero-area boxes
            if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
                continue

            all_boxes.append([
                float(label),
                score * weight,
                weight,
                float(model_idx),
                x1, y1, x2, y2,
            ])
            orig_indices.append((model_idx, box_idx))

    if len(all_boxes) == 0:
        return np.zeros((0, 8)), []

    arr = np.array(all_boxes, dtype=np.float64)

    # Sort by weighted score descending
    sort_order = np.argsort(-arr[:, 1])
    arr = arr[sort_order]
    orig_indices = [orig_indices[i] for i in sort_order]

    return arr, orig_indices


def _get_weighted_box(boxes: np.ndarray) -> np.ndarray:
    """Compute weighted average of boxes in a cluster.

    Args:
        boxes: ndarray of shape (K, 8) -- all boxes in this cluster.

    Returns:
        Weighted average box of shape (8,).
    """
    box = np.zeros(8, dtype=np.float64)
    total_weight = 0.0

    for b in boxes:
        weight = b[2]  # column 2 is weight
        box[4:] += b[4:] * weight  # weighted coords
        total_weight += weight

    if total_weight > 0:
        box[4:] /= total_weight

    # Label from highest-scoring box
    box[0] = boxes[0][0]
    # Weighted score: average of all weighted scores
    box[1] = np.mean(boxes[:, 1])
    # Average weight
    box[2] = total_weight / len(boxes)
    # Model index from first box (not meaningful for cluster centroid)
    box[3] = boxes[0][3]

    return box


def _find_matching_box_fast(
    boxes: np.ndarray,
    new_box: np.ndarray,
    match_iou: float,
) -> int:
    """Find the best IoU match for new_box against existing cluster centroids.

    Only matches boxes with the same label. Returns best match index or -1.

    Args:
        boxes: ndarray of shape (M, 8) -- existing cluster centroids.
        new_box: ndarray of shape (8,) -- candidate box.
        match_iou: IoU threshold for matching.

    Returns:
        Index of best match, or -1 if none exceeds threshold.
    """
    if len(boxes) == 0:
        return -1

    # Same-label filter
    label_mask = boxes[:, 0] == new_box[0]
    if not np.any(label_mask):
        return -1

    # Vectorized IoU computation against all centroids
    x1 = np.maximum(boxes[:, 4], new_box[4])
    y1 = np.maximum(boxes[:, 5], new_box[5])
    x2 = np.minimum(boxes[:, 6], new_box[6])
    y2 = np.minimum(boxes[:, 7], new_box[7])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes[:, 6] - boxes[:, 4]) * (boxes[:, 7] - boxes[:, 5])
    area_b = (new_box[6] - new_box[4]) * (new_box[7] - new_box[5])
    union = area_a + area_b - intersection

    iou = np.where(union > 0, intersection / union, 0.0)

    # Mask out different labels
    iou = np.where(label_mask, iou, 0.0)

    best_idx = int(np.argmax(iou))
    if iou[best_idx] >= match_iou:
        return best_idx

    return -1


def weighted_boxes_fusion_with_provenance(
    boxes_list: list[list[list[float]]],
    scores_list: list[list[float]],
    labels_list: list[list[int]],
    weights: list[float] | None = None,
    iou_thr: float = 0.3,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    """Weighted Boxes Fusion with provenance tracking.

    Clusters overlapping bounding boxes from multiple models using confidence-weighted
    averaging. Returns fused boxes plus cluster membership mapping.

    Args:
        boxes_list: Per-model list of boxes, each box [x1, y1, x2, y2] in [0,1].
        scores_list: Per-model list of confidence scores.
        labels_list: Per-model list of integer labels.
        weights: Per-model weights (default: all 1.0).
        iou_thr: IoU threshold for clustering (default 0.3).
        skip_box_thr: Minimum weighted score to keep a box (default 0.0 = keep all).
        conf_type: Confidence fusion type: "avg", "max", or "box_and_model_avg".

    Returns:
        boxes: ndarray of shape (N, 4) -- fused box coordinates in [0,1].
        scores: ndarray of shape (N,) -- fused confidence scores.
        labels: ndarray of shape (N,) -- fused labels (int).
        clusters: list of N lists, each containing (model_idx, original_box_idx)
                  tuples identifying contributing input boxes.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models

    # Prefilter and build 8-col array
    filtered, orig_indices = _prefilter_boxes(
        boxes_list, scores_list, labels_list, weights, skip_box_thr,
    )

    if len(filtered) == 0:
        return (
            np.zeros((0, 4)),
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            [],
        )

    # Cluster centroids and membership
    overall_boxes: list[np.ndarray] = []  # weighted avg centroids
    cluster_members: list[list[int]] = []  # indices into `filtered`

    for i, box in enumerate(filtered):
        match_idx = _find_matching_box_fast(
            np.array(overall_boxes) if overall_boxes else np.zeros((0, 8)),
            box,
            match_iou=iou_thr,
        )

        if match_idx >= 0:
            # Add to existing cluster
            cluster_members[match_idx].append(i)
            # Recompute weighted average
            member_boxes = filtered[cluster_members[match_idx]]
            overall_boxes[match_idx] = _get_weighted_box(member_boxes)
        else:
            # Create new cluster
            overall_boxes.append(box.copy())
            cluster_members.append([i])

    # Build output arrays
    n_clusters = len(overall_boxes)
    out_boxes = np.zeros((n_clusters, 4), dtype=np.float64)
    out_scores = np.zeros(n_clusters, dtype=np.float64)
    out_labels = np.zeros(n_clusters, dtype=np.int64)
    out_clusters: list[list[tuple[int, int]]] = []

    for c_idx in range(n_clusters):
        centroid = overall_boxes[c_idx]
        members = cluster_members[c_idx]

        out_boxes[c_idx] = centroid[4:]  # x1, y1, x2, y2
        out_labels[c_idx] = int(centroid[0])

        # Confidence rescaling
        member_boxes = filtered[members]
        if conf_type == "avg":
            out_scores[c_idx] = np.mean(member_boxes[:, 1])
        elif conf_type == "max":
            out_scores[c_idx] = np.max(member_boxes[:, 1])
        elif conf_type == "box_and_model_avg":
            # Average per model, then average across models
            model_scores: dict[int, list[float]] = {}
            for m_idx in members:
                model_id = int(filtered[m_idx, 3])
                if model_id not in model_scores:
                    model_scores[model_id] = []
                model_scores[model_id].append(filtered[m_idx, 1])
            model_avgs = [np.mean(s) for s in model_scores.values()]
            out_scores[c_idx] = np.mean(model_avgs)
        else:
            out_scores[c_idx] = np.mean(member_boxes[:, 1])

        # Provenance: map back to original (model_idx, box_idx)
        cluster_provenance = [orig_indices[m] for m in members]
        out_clusters.append(cluster_provenance)

    return out_boxes, out_scores, out_labels, out_clusters

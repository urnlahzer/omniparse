"""Spatial alignment module -- IoU bounding box matching and NW text alignment.

Matches regions across engine outputs by element_type + IoU, then aligns
differing text via Needleman-Wunsch using the sequence-align library.

Pattern: pure functions, no Modal dependency.
"""
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sequence_align.pairwise import needleman_wunsch, hirschberg

from omniparse.models.region import Region, EngineOutput
from omniparse.models.consensus import AlignedRegion
from omniparse.type_compatibility import TYPE_COMPAT_GROUPS
from omniparse.wbf import weighted_boxes_fusion_with_provenance

logger = logging.getLogger(__name__)

# NW parameters per ALGN-04.
# KNOWN LIMITATION: ALGN-04 specifies visually similar OCR pairs (0/O, l/1, rn/m)
# scored at -0.5 mismatch instead of -1.0. The sequence-align library accepts only
# a single scalar mismatch_score, not a substitution matrix. Implementing per-pair
# scoring would require a pre-alignment character normalization step (e.g., mapping
# 0->O, l->1 before alignment). Deferred to v2 -- uniform -1.0 mismatch is used
# for all character pairs. This may cause slightly suboptimal alignments for
# visually similar OCR confusions, but CE + voting still resolves most cases.
NW_MATCH_SCORE = 2.0
NW_MISMATCH_SCORE = -1.0  # Uniform; visually similar pair scoring deferred to v2
NW_INDEL_SCORE = -2.0
NW_GAP_CHAR = "\x00"  # Null byte — won't appear in OCR text (underscore did)

# Text length threshold for switching from NW to Hirschberg (O(min(n,m)) space)
HIRSCHBERG_THRESHOLD = 1000


def compute_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute Intersection over Union for two [x1, y1, x2, y2] bounding boxes.

    Returns 0.0 for non-overlapping or zero-area boxes.
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def compute_containment_ratio(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute containment ratio: intersection / min(area_A, area_B).

    Returns 1.0 when smaller box is fully inside larger box.
    Returns 0.0 for non-overlapping or zero-area boxes.
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    min_area = min(area_a, area_b)

    if min_area <= 0.0:
        return 0.0

    return intersection / min_area


def compute_center_distance(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute Euclidean distance between bbox centers in coordinate space.

    For [0,1] normalized coordinates, result is in [0, sqrt(2)] range.
    Returns 0.0 for identical centers.
    """
    cx_a = (bbox_a[0] + bbox_a[2]) / 2.0
    cy_a = (bbox_a[1] + bbox_a[3]) / 2.0
    cx_b = (bbox_b[0] + bbox_b[2]) / 2.0
    cy_b = (bbox_b[1] + bbox_b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def merge_contained_texts(regions: list[Region]) -> str:
    """Concatenate region texts in reading order (y1 then x1).

    Uses bounding_box_norm if available, falls back to bounding_box.
    """
    sorted_regions = sorted(
        regions,
        key=lambda r: (
            (r.bounding_box_norm or r.bounding_box)[1],  # y1
            (r.bounding_box_norm or r.bounding_box)[0],  # x1
        ),
    )
    return " ".join(r.text_content for r in sorted_regions)


# Engine name -> WBF weight per D-10
ENGINE_WEIGHTS: dict[str, float] = {
    "pdfplumber": 3.0,
    "paddleocr": 1.0,
    "docling": 1.0,
    "trocr": 2.0,
}


def _wbf_group_regions(
    unmatched_regions: list[tuple[str, Region]],
    iou_thr: float = 0.3,
) -> list[dict]:
    """Group unmatched regions via WBF. Returns match groups with provenance.

    Converts Region objects to WBF input format, runs weighted_boxes_fusion_with_provenance,
    and converts WBF clusters back to match group dicts (same format as IoU groups).

    Args:
        unmatched_regions: List of (engine_name, Region) tuples.
        iou_thr: IoU threshold for WBF clustering.

    Returns:
        List of match group dicts with 'regions', 'element_type', 'bounding_box',
        'bounding_box_norm', and 'match_type' keys.
    """
    if not unmatched_regions:
        return []

    # Build per-engine index for WBF input
    engine_names_seen: list[str] = []
    engine_idx_map: dict[str, int] = {}
    engine_region_lists: dict[str, list[Region]] = {}

    for eng_name, region in unmatched_regions:
        if eng_name not in engine_idx_map:
            engine_idx_map[eng_name] = len(engine_names_seen)
            engine_names_seen.append(eng_name)
            engine_region_lists[eng_name] = []
        engine_region_lists[eng_name].append(region)

    # Build WBF inputs per model
    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []
    weights: list[float] = []

    for eng_name in engine_names_seen:
        regions = engine_region_lists[eng_name]
        boxes = []
        scores = []
        labels = []
        for r in regions:
            bbox = r.bounding_box_norm or r.bounding_box
            boxes.append(bbox)
            scores.append(r.confidence)
            labels.append(0)  # All same label (within same compat bucket)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        weights.append(ENGINE_WEIGHTS.get(eng_name, 1.0))

    # Run WBF
    fused_boxes, fused_scores, fused_labels, clusters = (
        weighted_boxes_fusion_with_provenance(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=0.0,
        )
    )

    # Convert clusters back to match groups
    result_groups: list[dict] = []
    for c_idx, cluster in enumerate(clusters):
        group_regions: dict[str, Region] = {}
        for model_idx, box_idx in cluster:
            eng_name = engine_names_seen[model_idx]
            region = engine_region_lists[eng_name][box_idx]
            if eng_name in group_regions:
                # Multiple boxes from same engine in same cluster -- merge
                existing = group_regions[eng_name]
                merged_text = existing.text_content + " " + region.text_content
                group_regions[eng_name] = Region(
                    id=existing.id,
                    element_type=existing.element_type,
                    bounding_box=existing.bounding_box,
                    bounding_box_norm=existing.bounding_box_norm,
                    confidence=max(existing.confidence, region.confidence),
                    text_content=merged_text,
                    metadata=existing.metadata,
                )
            else:
                group_regions[eng_name] = region

        # Determine element_type from first region
        first_region = next(iter(group_regions.values()))
        element_type = first_region.element_type

        avg_bbox = _average_bboxes([r.bounding_box for r in group_regions.values()])
        avg_bbox_norm = _average_bboxes([
            r.bounding_box_norm or r.bounding_box for r in group_regions.values()
        ])

        match_type = "wbf" if len(group_regions) > 1 else "orphan"

        result_groups.append({
            "regions": group_regions,
            "element_type": element_type,
            "bounding_box": avg_bbox,
            "bounding_box_norm": avg_bbox_norm,
            "match_type": match_type,
        })

    return result_groups


def agglomerative_cluster_orphans(
    orphan_regions: list[tuple[str, Region]],
    distance_threshold: float = 0.92,
) -> list[dict]:
    """Group orphan regions via agglomerative clustering (complete linkage).

    Uses 1-IoU as distance metric. Regions closer than distance_threshold
    (i.e., IoU > 1-distance_threshold = 0.08) are grouped together.

    Args:
        orphan_regions: List of (engine_name, Region) tuples.
        distance_threshold: Maximum distance (1-IoU) for cluster membership.
            Default 0.92 means IoU > 0.08 required (per D-05).

    Returns:
        List of match group dicts (same format as IoU groups).
    """
    if not orphan_regions:
        return []

    # Single region -> orphan
    if len(orphan_regions) == 1:
        eng_name, region = orphan_regions[0]
        return [{
            "regions": {eng_name: region},
            "element_type": region.element_type,
            "bounding_box": region.bounding_box,
            "bounding_box_norm": region.bounding_box_norm,
            "match_type": "orphan",
        }]

    n = len(orphan_regions)

    # Build pairwise distance matrix (1-IoU)
    dist_matrix = []
    for i in range(n):
        for j in range(i + 1, n):
            bbox_i = orphan_regions[i][1].bounding_box_norm or orphan_regions[i][1].bounding_box
            bbox_j = orphan_regions[j][1].bounding_box_norm or orphan_regions[j][1].bounding_box
            iou = compute_iou(bbox_i, bbox_j)
            dist_matrix.append(1.0 - iou)

    # Run agglomerative clustering (complete linkage = most conservative)
    if len(dist_matrix) > 0:
        Z = linkage(dist_matrix, method="complete")
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        labels = [1]

    # Group regions by cluster label
    cluster_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)

    # Build match groups from clusters
    result_groups: list[dict] = []
    for cluster_label, member_indices in cluster_map.items():
        group_regions: dict[str, Region] = {}

        for idx in member_indices:
            eng_name, region = orphan_regions[idx]
            if eng_name in group_regions:
                # Same engine, multiple regions in cluster -> merge texts
                existing = group_regions[eng_name]
                merged_text = merge_contained_texts([existing, region])
                group_regions[eng_name] = Region(
                    id=existing.id,
                    element_type=existing.element_type,
                    bounding_box=existing.bounding_box,
                    bounding_box_norm=existing.bounding_box_norm,
                    confidence=max(existing.confidence, region.confidence),
                    text_content=merged_text,
                    metadata=existing.metadata,
                )
            else:
                group_regions[eng_name] = region

        # Determine match_type
        match_type = "cluster" if len(group_regions) > 1 else "orphan"

        first_region = next(iter(group_regions.values()))
        avg_bbox = _average_bboxes([r.bounding_box for r in group_regions.values()])
        avg_bbox_norm = _average_bboxes([
            r.bounding_box_norm or r.bounding_box for r in group_regions.values()
        ])

        result_groups.append({
            "regions": group_regions,
            "element_type": first_region.element_type,
            "bounding_box": avg_bbox,
            "bounding_box_norm": avg_bbox_norm,
            "match_type": match_type,
        })

    return result_groups


def match_regions_across_engines(
    engine_outputs: list[EngineOutput],
    iou_threshold: float = 0.5,
    center_distance_threshold: float = 0.05,
    center_iou_floor: float = 0.05,
    containment_threshold: float = 0.6,
    wbf_iou_threshold: float = 0.3,
    cluster_distance_threshold: float = 0.92,
) -> list[dict]:
    """Match regions across engine outputs using layered strategy.

    Layered order: IoU -> center_rescue -> containment -> WBF -> clustering -> orphans.

    Algorithm:
    1. Group all regions across all engines by compatibility group.
    2. Greedy IoU matching (highest IoU above threshold).
    2a. Center-distance rescue (center_distance <= threshold AND IoU > floor).
    2b. Containment rescue (intersection/min_area >= containment_threshold).
    2c. WBF clustering on remaining unmatched regions.
    2d. Agglomerative clustering on WBF orphans.
    3. True orphans become single-engine groups.

    Args:
        engine_outputs: List of EngineOutput objects to align.
        iou_threshold: IoU threshold for greedy matching (default 0.5).
        center_distance_threshold: Max center distance for rescue (default 0.05).
        center_iou_floor: Min IoU required for center rescue (default 0.05).
        containment_threshold: Containment ratio threshold for rescue pass (default 0.6).
        wbf_iou_threshold: IoU threshold for WBF clustering (default 0.3).
        cluster_distance_threshold: Max 1-IoU distance for agglomerative clustering (default 0.92).

    Returns:
        List of dicts with keys:
          - regions: dict[engine_name -> Region]
          - element_type: str
          - bounding_box: list[float] (average of matched boxes)
          - bounding_box_norm: list[float] | None
          - match_type: str ("iou", "center_rescue", "containment", "wbf", "cluster", "orphan")
    """
    if not engine_outputs:
        return []

    # Step 1: Group regions by compatibility group (not exact type) per D-06
    compat_buckets: dict[str, dict[str, list[Region]]] = {}
    for eo in engine_outputs:
        for region in eo.regions:
            group = TYPE_COMPAT_GROUPS.get(region.element_type, region.element_type)
            if group not in compat_buckets:
                compat_buckets[group] = {}
            if eo.engine not in compat_buckets[group]:
                compat_buckets[group][eo.engine] = []
            compat_buckets[group][eo.engine].append(region)

    matched_groups: list[dict] = []

    # Step 2: Within each type group, do greedy IoU matching
    for element_type, engine_regions in compat_buckets.items():
        engine_names = list(engine_regions.keys())
        if len(engine_names) == 0:
            continue

        # Track which regions have been matched (consumed)
        consumed: dict[str, set[int]] = {en: set() for en in engine_names}

        # Use the first engine as the anchor
        anchor_engine = engine_names[0]
        other_engines = engine_names[1:]

        for anchor_idx, anchor_region in enumerate(engine_regions[anchor_engine]):
            if anchor_idx in consumed[anchor_engine]:
                continue

            group_regions: dict[str, Region] = {anchor_engine: anchor_region}

            for other_engine in other_engines:
                best_iou = 0.0
                best_idx = -1

                for other_idx, other_region in enumerate(engine_regions[other_engine]):
                    if other_idx in consumed[other_engine]:
                        continue
                    bbox_a = anchor_region.bounding_box_norm or anchor_region.bounding_box
                    bbox_b = other_region.bounding_box_norm or other_region.bounding_box
                    iou = compute_iou(bbox_a, bbox_b)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = other_idx

                if best_iou >= iou_threshold and best_idx >= 0:
                    group_regions[other_engine] = engine_regions[other_engine][best_idx]
                    consumed[other_engine].add(best_idx)

            # Only consume anchor and create group if multi-engine match found
            if len(group_regions) > 1:
                consumed[anchor_engine].add(anchor_idx)

                # Compute average bounding box (pixel + normalized)
                avg_bbox = _average_bboxes([r.bounding_box for r in group_regions.values()])
                avg_bbox_norm = _average_bboxes([
                    r.bounding_box_norm or r.bounding_box for r in group_regions.values()
                ])

                matched_groups.append({
                    "regions": group_regions,
                    "element_type": element_type,
                    "bounding_box": avg_bbox,
                    "bounding_box_norm": avg_bbox_norm,
                    "match_type": "iou",
                })

        # Step 2a: Center-distance rescue pass (D-02, D-07)
        # For unconsumed regions, check if centers are close AND there is SOME overlap.
        # This catches Y-offset boxes that IoU misses (IoU 0.08-0.32, center_dist < 0.05).
        # The IoU floor prevents false matches on stacked text with no overlap.
        for anchor_idx, anchor_region in enumerate(engine_regions[anchor_engine]):
            if anchor_idx in consumed[anchor_engine]:
                continue

            anchor_bbox = anchor_region.bounding_box_norm or anchor_region.bounding_box

            for other_engine in other_engines:
                best_dist = float("inf")
                best_idx = -1
                best_iou = 0.0

                for other_idx, other_region in enumerate(engine_regions[other_engine]):
                    if other_idx in consumed[other_engine]:
                        continue
                    other_bbox = other_region.bounding_box_norm or other_region.bounding_box
                    dist = compute_center_distance(anchor_bbox, other_bbox)
                    iou = compute_iou(anchor_bbox, other_bbox)
                    if (
                        dist <= center_distance_threshold
                        and iou > center_iou_floor
                        and dist < best_dist
                    ):
                        best_dist = dist
                        best_idx = other_idx
                        best_iou = iou

                if best_idx >= 0:
                    # Form a center-rescue match group
                    group_regions: dict[str, Region] = {
                        anchor_engine: anchor_region,
                        other_engine: engine_regions[other_engine][best_idx],
                    }
                    consumed[anchor_engine].add(anchor_idx)
                    consumed[other_engine].add(best_idx)

                    avg_bbox = _average_bboxes([r.bounding_box for r in group_regions.values()])
                    avg_bbox_norm = _average_bboxes([
                        r.bounding_box_norm or r.bounding_box for r in group_regions.values()
                    ])

                    matched_groups.append({
                        "regions": group_regions,
                        "element_type": element_type,
                        "bounding_box": avg_bbox,
                        "bounding_box_norm": avg_bbox_norm,
                        "match_type": "center_rescue",
                    })
                    break  # Anchor consumed, move to next anchor

        # Step 2b: Containment rescue pass (D-03, D-04, D-05)
        # For each unconsumed region, check containment against existing matched
        # groups AND other unconsumed regions (to handle the case where a large
        # unconsumed box contains multiple small unconsumed boxes from another engine).

        # First: try to attach unconsumed regions to existing IoU-matched groups
        for group in list(matched_groups):
            if group["element_type"] != element_type:
                continue
            # Get the representative bbox for this group (use first region's norm bbox)
            for _en, _reg in group["regions"].items():
                group_bbox = _reg.bounding_box_norm or _reg.bounding_box
                break

            for eng_name in engine_names:
                if eng_name in group["regions"]:
                    continue  # This engine already has a region in this group
                contained_regions: list[tuple[int, Region]] = []
                for idx, region in enumerate(engine_regions[eng_name]):
                    if idx in consumed[eng_name]:
                        continue
                    region_bbox = region.bounding_box_norm or region.bounding_box
                    cr = compute_containment_ratio(region_bbox, group_bbox)
                    if cr >= containment_threshold:
                        contained_regions.append((idx, region))

                if contained_regions:
                    # Many-to-one: merge texts in reading order (D-06)
                    for idx, _ in contained_regions:
                        consumed[eng_name].add(idx)
                    if len(contained_regions) == 1:
                        group["regions"][eng_name] = contained_regions[0][1]
                    else:
                        regions_list = [r for _, r in contained_regions]
                        merged_text = merge_contained_texts(regions_list)
                        # Create a merged Region entry (D-06, D-07)
                        base = contained_regions[0][1]
                        merged_region = Region(
                            id=base.id,
                            element_type=base.element_type,
                            bounding_box=base.bounding_box,
                            bounding_box_norm=base.bounding_box_norm,
                            confidence=max(r.confidence for _, r in contained_regions),
                            text_content=merged_text,
                            metadata=base.metadata,
                        )
                        group["regions"][eng_name] = merged_region
                    group["match_type"] = "containment"

        # Second: form NEW groups from unconsumed regions where a large box from
        # one engine contains small boxes from another engine
        # Collect all unconsumed regions with engine identity
        unconsumed_by_engine: list[tuple[str, int, Region]] = []
        for eng_name in engine_names:
            for idx, region in enumerate(engine_regions[eng_name]):
                if idx not in consumed[eng_name]:
                    unconsumed_by_engine.append((eng_name, idx, region))

        # For each unconsumed region, check if it contains other unconsumed regions
        # from different engines
        newly_consumed: set[tuple[str, int]] = set()
        for eng_a, idx_a, region_a in unconsumed_by_engine:
            if (eng_a, idx_a) in newly_consumed:
                continue
            bbox_a = region_a.bounding_box_norm or region_a.bounding_box
            area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])

            # Find all unconsumed regions from OTHER engines that are contained
            contained_by_engine: dict[str, list[tuple[int, Region]]] = {}
            for eng_b, idx_b, region_b in unconsumed_by_engine:
                if eng_b == eng_a or (eng_b, idx_b) in newly_consumed:
                    continue
                bbox_b = region_b.bounding_box_norm or region_b.bounding_box
                area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
                cr = compute_containment_ratio(bbox_a, bbox_b)
                if cr >= containment_threshold:
                    # Only consider cases where region_b is smaller (contained)
                    # or region_a is smaller (contained in region_b)
                    if eng_b not in contained_by_engine:
                        contained_by_engine[eng_b] = []
                    contained_by_engine[eng_b].append((idx_b, region_b))

            if contained_by_engine:
                # Form a new match group
                new_group_regions: dict[str, Region] = {eng_a: region_a}
                newly_consumed.add((eng_a, idx_a))
                consumed[eng_a].add(idx_a)

                for eng_b, regions_b in contained_by_engine.items():
                    for idx_b, _ in regions_b:
                        newly_consumed.add((eng_b, idx_b))
                        consumed[eng_b].add(idx_b)

                    if len(regions_b) == 1:
                        new_group_regions[eng_b] = regions_b[0][1]
                    else:
                        regions_list = [r for _, r in regions_b]
                        merged_text = merge_contained_texts(regions_list)
                        base = regions_b[0][1]
                        merged_region = Region(
                            id=base.id,
                            element_type=base.element_type,
                            bounding_box=base.bounding_box,
                            bounding_box_norm=base.bounding_box_norm,
                            confidence=max(r.confidence for _, r in regions_b),
                            text_content=merged_text,
                            metadata=base.metadata,
                        )
                        new_group_regions[eng_b] = merged_region

                avg_bbox = _average_bboxes([r.bounding_box for r in new_group_regions.values()])
                avg_bbox_norm = _average_bboxes([
                    r.bounding_box_norm or r.bounding_box for r in new_group_regions.values()
                ])
                matched_groups.append({
                    "regions": new_group_regions,
                    "element_type": element_type,
                    "bounding_box": avg_bbox,
                    "bounding_box_norm": avg_bbox_norm,
                    "match_type": "containment",
                })

        # Step 2c: WBF pass on remaining unmatched regions (D-04, D-05)
        # WBF requires [0,1] normalized coords. Regions without bounding_box_norm
        # go directly to orphan status.
        remaining_for_wbf: list[tuple[str, Region]] = []
        remaining_orphans: list[tuple[str, Region]] = []
        for eng_name in engine_names:
            for idx, region in enumerate(engine_regions[eng_name]):
                if idx not in consumed[eng_name]:
                    if region.bounding_box_norm is not None:
                        remaining_for_wbf.append((eng_name, region))
                    else:
                        remaining_orphans.append((eng_name, region))

        if remaining_for_wbf:
            wbf_groups = _wbf_group_regions(remaining_for_wbf, iou_thr=wbf_iou_threshold)
        else:
            wbf_groups = []

        # Step 2d: Agglomerative clustering on WBF orphans (D-04, D-05, D-06)
        # Collect WBF orphans (single-engine WBF clusters) and no-norm regions
        wbf_orphan_regions: list[tuple[str, Region]] = []
        non_orphan_wbf_groups: list[dict] = []

        for group in wbf_groups:
            if group["match_type"] == "orphan":
                # Extract the single-engine region back for clustering
                for eng_name, region in group["regions"].items():
                    wbf_orphan_regions.append((eng_name, region))
            else:
                non_orphan_wbf_groups.append(group)

        matched_groups.extend(non_orphan_wbf_groups)

        # Add regions that skipped WBF (no bounding_box_norm)
        for eng_name, region in remaining_orphans:
            wbf_orphan_regions.append((eng_name, region))

        # Run agglomerative clustering on all orphans
        if wbf_orphan_regions:
            cluster_groups = agglomerative_cluster_orphans(
                wbf_orphan_regions,
                distance_threshold=cluster_distance_threshold,
            )
            matched_groups.extend(cluster_groups)

    return matched_groups


def _average_bboxes(bboxes: list[list[float]]) -> list[float]:
    """Compute element-wise average of a list of [x1, y1, x2, y2] bounding boxes."""
    n = len(bboxes)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        sum(b[i] for b in bboxes) / n
        for i in range(4)
    ]


def align_texts(text_a: str, text_b: str) -> tuple[list[str], list[str]]:
    """Align two texts character-by-character via Needleman-Wunsch.

    Uses hirschberg() for texts exceeding HIRSCHBERG_THRESHOLD characters
    (O(min(n,m)) space vs O(n*m) for standard NW).

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Tuple of (aligned_a, aligned_b) as character lists with gap characters.
    """
    # Strip gap char from input — sequence_align rejects it in input sequences
    clean_a = text_a.replace(NW_GAP_CHAR, "")
    clean_b = text_b.replace(NW_GAP_CHAR, "")
    seq_a = list(clean_a)
    seq_b = list(clean_b)

    if len(clean_a) > HIRSCHBERG_THRESHOLD or len(clean_b) > HIRSCHBERG_THRESHOLD:
        aligned_a, aligned_b = hirschberg(
            seq_a, seq_b,
            match_score=NW_MATCH_SCORE,
            mismatch_score=NW_MISMATCH_SCORE,
            indel_score=NW_INDEL_SCORE,
            gap=NW_GAP_CHAR,
        )
    else:
        aligned_a, aligned_b = needleman_wunsch(
            seq_a, seq_b,
            match_score=NW_MATCH_SCORE,
            mismatch_score=NW_MISMATCH_SCORE,
            indel_score=NW_INDEL_SCORE,
            gap=NW_GAP_CHAR,
        )

    return aligned_a, aligned_b


def align_region_group(matched_group: dict) -> AlignedRegion:
    """Produce an AlignedRegion from a matched group of cross-engine regions.

    Logic:
    - All texts identical -> source="identical", consensus set, skip NW.
    - Single engine -> source="single_engine", consensus set.
    - Texts differ -> pairwise NW alignment, source="pending" (consensus module resolves).

    Args:
        matched_group: Dict with keys "regions" (engine->Region), "element_type", "bounding_box".

    Returns:
        AlignedRegion with alignment results.
    """
    regions: dict[str, Region] = matched_group["regions"]
    element_type: str = matched_group["element_type"]
    bounding_box: list[float] = matched_group["bounding_box"]

    engine_texts = {engine: region.text_content for engine, region in regions.items()}

    # Determine the primary region ID (use first engine's region ID)
    first_engine = next(iter(regions))
    region_id = regions[first_engine].id

    # Collect metadata from Docling (hierarchy_level) and table regions (table_structure)
    metadata = _collect_metadata(regions)

    # Single engine case
    if len(regions) == 1:
        region = next(iter(regions.values()))
        return AlignedRegion(
            region_id=region_id,
            element_type=element_type,
            bounding_box=bounding_box,
            engine_texts=engine_texts,
            consensus_text=region.text_content,
            confidence=region.confidence,
            source="single_engine",
            metadata=metadata,
        )

    # Check if all texts are identical (fast path)
    unique_texts = set(engine_texts.values())
    if len(unique_texts) == 1:
        max_confidence = max(r.confidence for r in regions.values())
        return AlignedRegion(
            region_id=region_id,
            element_type=element_type,
            bounding_box=bounding_box,
            engine_texts=engine_texts,
            consensus_text=next(iter(unique_texts)),
            confidence=max_confidence,
            source="identical",
            metadata=metadata,
        )

    # Texts differ -- pairwise NW alignment against the first engine's text
    anchor_engine = first_engine
    anchor_text = engine_texts[anchor_engine]
    aligned_texts: dict[str, list[str]] = {}

    for engine, text in engine_texts.items():
        if engine == anchor_engine:
            # Will be set after first alignment (anchor gets aligned too)
            continue
        aligned_anchor, aligned_other = align_texts(anchor_text, text)
        # Store the anchor alignment from first pair only
        if anchor_engine not in aligned_texts:
            aligned_texts[anchor_engine] = aligned_anchor
        aligned_texts[engine] = aligned_other

    # If only 2 engines, the anchor was already set above
    # If anchor wasn't set (shouldn't happen, but safety), set it from the text
    if anchor_engine not in aligned_texts:
        aligned_texts[anchor_engine] = list(anchor_text)

    return AlignedRegion(
        region_id=region_id,
        element_type=element_type,
        bounding_box=bounding_box,
        engine_texts=engine_texts,
        aligned_texts=aligned_texts,
        source="pending",
        metadata=metadata,
    )


def _collect_metadata(regions: dict[str, Region]) -> dict | None:
    """Collect metadata from matched regions.

    Carries forward:
    - hierarchy_level from Docling region metadata
    - table_structure from any region that has it
    """
    metadata: dict = {}

    for engine, region in regions.items():
        # Docling hierarchy_level
        if engine == "docling" and region.metadata and "hierarchy_level" in region.metadata:
            metadata["hierarchy_level"] = region.metadata["hierarchy_level"]

        # Table structure from any engine
        if region.table_structure is not None:
            metadata["table_structure"] = region.table_structure

    return metadata if metadata else None

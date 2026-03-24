"""Conservative Docling pre-merge — join word-level fragments on the same text line.

Docling produces word-level regions (e.g., "WHEREAS", "the", "parties agree") where
PaddleOCR detects a single line. This creates orphans during cross-engine alignment
because the granularity mismatch prevents IoU/containment matching.

Conservative pre-merge joins Docling fragments that are on the same text line:
- Vertical overlap in Y range (actual overlap, not just adjacent)
- Horizontal gap < 0.05 in [0,1] space (~one word gap)
- Same element_type

Runs AFTER [0,1] normalization and noise filtering, BEFORE alignment.
Only applies to Docling engine output (PaddleOCR already detects at line level).

Implements D-01 through D-04 from Phase 10 context.
"""
from omniparse.models.region import Region


# Maximum horizontal gap (in [0,1] normalized space) for merging.
# ~5% of page width, roughly one word gap on a legal document.
MAX_HORIZONTAL_GAP = 0.05


def premerge_docling_regions(regions: list[Region]) -> list[Region]:
    """Merge word-level Docling fragments that belong to the same text line.

    Uses union-find for transitive merging: if A merges with B and B merges with C,
    all three end up in one merged region.

    Args:
        regions: Docling regions with bounding_box_norm populated (post-normalization).

    Returns:
        Merged regions list. Non-mergeable regions pass through unchanged.
    """
    if not regions:
        return []

    n = len(regions)
    if n == 1:
        return list(regions)

    # Sort by (y_center, x1) for deterministic processing
    sorted_regions = sorted(
        regions,
        key=lambda r: (
            ((r.bounding_box_norm or r.bounding_box)[1] + (r.bounding_box_norm or r.bounding_box)[3]) / 2,
            (r.bounding_box_norm or r.bounding_box)[0],
        ),
    )

    # Union-find for transitive merging
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Check all pairs for merge eligibility
    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(sorted_regions[i], sorted_regions[j]):
                union(i, j)

    # Group regions by their root in the union-find
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build merged regions
    result: list[Region] = []
    for indices in groups.values():
        if len(indices) == 1:
            result.append(sorted_regions[indices[0]])
        else:
            result.append(_merge_group([sorted_regions[i] for i in indices]))

    return result


def _should_merge(a: Region, b: Region) -> bool:
    """Check if two regions should be merged (D-01, D-03).

    Criteria (ALL must be true):
    1. Same element_type
    2. Vertical overlap in Y range (actual overlap: y1_A < y2_B AND y1_B < y2_A)
    3. Horizontal gap < MAX_HORIZONTAL_GAP in [0,1] space
    """
    if a.element_type != b.element_type:
        return False

    a_bbox = a.bounding_box_norm or a.bounding_box
    b_bbox = b.bounding_box_norm or b.bounding_box

    a_x1, a_y1, a_x2, a_y2 = a_bbox
    b_x1, b_y1, b_x2, b_y2 = b_bbox

    # Vertical overlap check (actual overlap, not just adjacent)
    if not (a_y1 < b_y2 and b_y1 < a_y2):
        return False

    # Horizontal gap check
    # Sort by x1 to determine left/right
    left_x2 = min(a_x2, b_x2) if a_x1 > b_x1 else a_x2 if a_x1 <= b_x1 else b_x2
    right_x1 = max(a_x1, b_x1) if a_x1 > b_x1 else b_x1 if a_x1 <= b_x1 else a_x1

    # For overlapping regions, gap is negative (which is < threshold, so they merge)
    if a_x1 <= b_x1:
        gap = b_x1 - a_x2
    else:
        gap = a_x1 - b_x2

    # gap < 0 means overlap (should merge)
    # gap >= 0 means actual gap (must be < threshold)
    if gap >= MAX_HORIZONTAL_GAP:
        return False

    return True


def _merge_group(regions: list[Region]) -> Region:
    """Merge a group of regions into one (D-04).

    - bounding_box/bounding_box_norm: union of all constituent bboxes
    - text_content: concatenated left-to-right by x1 position, space-separated
    - confidence: min of all constituents
    - id: from the leftmost constituent (smallest x1)
    - element_type: preserved (all same by merge criteria)
    - metadata: from leftmost constituent
    """
    # Sort by x1 for left-to-right ordering
    by_x = sorted(
        regions,
        key=lambda r: (r.bounding_box_norm or r.bounding_box)[0],
    )

    leftmost = by_x[0]

    # Union of bounding boxes (pixel coords)
    all_bbox = [r.bounding_box for r in regions]
    merged_bbox = [
        min(b[0] for b in all_bbox),
        min(b[1] for b in all_bbox),
        max(b[2] for b in all_bbox),
        max(b[3] for b in all_bbox),
    ]

    # Union of normalized bounding boxes
    all_norm = [r.bounding_box_norm for r in regions if r.bounding_box_norm is not None]
    merged_norm = None
    if all_norm:
        merged_norm = [
            min(b[0] for b in all_norm),
            min(b[1] for b in all_norm),
            max(b[2] for b in all_norm),
            max(b[3] for b in all_norm),
        ]

    return Region(
        id=leftmost.id,
        element_type=leftmost.element_type,
        bounding_box=merged_bbox,
        bounding_box_norm=merged_norm,
        confidence=min(r.confidence for r in regions),
        text_content=" ".join(r.text_content for r in by_x),
        table_structure=leftmost.table_structure,
        metadata=leftmost.metadata,
    )

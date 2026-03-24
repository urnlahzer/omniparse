"""Consensus decision module -- CE computation, majority voting, and page-level orchestration.

Decision pipeline per region:
1. source="identical" -> pass through (fast path from alignment)
2. source="single_engine" -> pass through
3. Compute CE from aligned_texts
4. CE <= CE_THRESHOLD -> weighted majority vote -> source="voting"
5. CE > CE_THRESHOLD -> weighted majority vote -> source="voting_fallback" (best-effort),
   then set needs_arbitration=True (LLM arbiter will attempt to improve).
   Per CONTEXT.md: "LLM rejection fallback: HITL flag + majority vote fallback."
   The voting_fallback consensus_text persists if the LLM output is later rejected.

Orchestration layer (arbitrate_page):
6. For each region with needs_arbitration=True: call LLM arbiter.
7. Accepted -> overwrite with source="arbitration".
8. Rejected or exception -> keep voting_fallback, set source="hitl_fallback", hitl_flag=True.

Pattern: pure functions, no Modal dependency.
"""
import io
import math
import logging
import string
from collections import Counter

from PIL import Image

from omniparse.models.consensus import AlignedRegion, ConsensusResult
from omniparse.alignment import NW_GAP_CHAR

logger = logging.getLogger(__name__)

# CE threshold: below this, voting resolves; above, LLM arbitration
# Range for 3 engines: 0.0 (perfect) to 1.585 (total disagreement)
# 0.4 chosen as starting point per research recommendation
CE_THRESHOLD = 0.4

# Default engine voting weights (pdfplumber gets 2x when ground truth)
DEFAULT_WEIGHTS = {"pdfplumber": 1.0, "paddleocr": 1.0, "docling": 1.0}
GROUND_TRUTH_WEIGHTS = {"pdfplumber": 2.0, "paddleocr": 1.0, "docling": 1.0}

# Handwriting specialist weights -- TrOCR weighted 2x per SPEC-04
HANDWRITING_WEIGHTS = {"trocr": 2.0, "paddleocr": 1.0}


def compute_position_entropy(chars: list[str]) -> float:
    """Shannon entropy for a list of characters at a single alignment position.

    Args:
        chars: Characters from each engine at one aligned position.

    Returns:
        Shannon entropy in bits. 0.0 = perfect agreement, log2(n) = total disagreement.
    """
    if not chars:
        return 0.0

    n = len(chars)
    counts = Counter(chars)

    # All identical -> entropy 0
    if len(counts) == 1:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return entropy


def compute_region_ce(
    aligned_texts: dict[str, list[str]],
    gap_char: str = NW_GAP_CHAR,
) -> float:
    """Compute Consensus Entropy for a region as average per-position Shannon entropy.

    Skips positions where ALL engines have the gap character (insertion artifacts).

    Args:
        aligned_texts: Engine name -> NW-aligned character list. All lists must be same length.
        gap_char: The gap character used in NW alignment.

    Returns:
        Average Shannon entropy across valid (non-all-gap) positions.
    """
    if not aligned_texts:
        return 0.0

    engines = list(aligned_texts.keys())
    # Use min length — pairwise NW alignment may produce different lengths per engine
    num_positions = min(len(aligned_texts[e]) for e in engines)

    if num_positions == 0:
        return 0.0

    total_entropy = 0.0
    valid_positions = 0

    for pos in range(num_positions):
        chars_at_pos = [aligned_texts[engine][pos] for engine in engines]

        # Skip positions where ALL engines have the gap character
        if all(c == gap_char for c in chars_at_pos):
            continue

        total_entropy += compute_position_entropy(chars_at_pos)
        valid_positions += 1

    if valid_positions == 0:
        return 0.0

    return total_entropy / valid_positions


def weighted_majority_vote(
    aligned_texts: dict[str, list[str]],
    weights: dict[str, float],
    gap_char: str = NW_GAP_CHAR,
) -> str:
    """Per-position weighted vote across aligned engine texts.

    For each position, sums weights for each non-gap character. The character
    with the highest total weight wins. Gap-only positions produce no output.

    Args:
        aligned_texts: Engine name -> NW-aligned character list.
        weights: Engine name -> voting weight.
        gap_char: The gap character used in NW alignment.

    Returns:
        Consensus string (gap characters excluded from output).
    """
    if not aligned_texts:
        return ""

    engines = list(aligned_texts.keys())
    # Use min length — pairwise NW alignment may produce different lengths per engine
    num_positions = min(len(aligned_texts[e]) for e in engines)
    result_chars: list[str] = []

    for pos in range(num_positions):
        # Collect weighted votes for non-gap characters
        char_weights: dict[str, float] = {}

        for engine in engines:
            char = aligned_texts[engine][pos]
            if char == gap_char:
                continue
            weight = weights.get(engine, 1.0)
            char_weights[char] = char_weights.get(char, 0.0) + weight

        # Gap-only position -> no output
        if not char_weights:
            continue

        # Winner is the character with highest total weight
        winner = max(char_weights, key=lambda c: char_weights[c])
        result_chars.append(winner)

    return "".join(result_chars)


def resolve_region(
    region: AlignedRegion,
    weights: dict[str, float] | None = None,
    ce_threshold: float = CE_THRESHOLD,
) -> AlignedRegion:
    """Resolve a single region through the consensus decision pipeline.

    Pipeline:
    1. source="identical" or "single_engine" -> pass through unchanged.
    2. No aligned_texts -> pass through unchanged.
    3. Compute CE from aligned_texts.
    4. CE <= threshold -> weighted majority vote -> source="voting".
    5. CE > threshold -> weighted majority vote (fallback) -> source="voting_fallback",
       needs_arbitration=True.

    Args:
        region: AlignedRegion to resolve.
        weights: Engine voting weights. Defaults to DEFAULT_WEIGHTS.
        ce_threshold: CE threshold separating voting from arbitration.

    Returns:
        Updated AlignedRegion with consensus_text and source set.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Handwriting regions use specialist weights (TrOCR 2x) per SPEC-04
    if region.element_type == "handwriting":
        weights = HANDWRITING_WEIGHTS

    # Pass through already-resolved regions
    if region.source in ("identical", "single_engine"):
        return region

    # Nothing to resolve without aligned texts
    if region.aligned_texts is None:
        return region

    # Compute Consensus Entropy
    ce = compute_region_ce(region.aligned_texts)
    num_engines = len(region.aligned_texts)
    max_entropy = math.log2(num_engines) if num_engines > 1 else 1.0

    # Confidence: 1.0 - (ce / max_possible_entropy)
    confidence = max(0.0, min(1.0, 1.0 - (ce / max_entropy)))

    if ce <= ce_threshold:
        # Low CE -> voting resolves
        consensus_text = weighted_majority_vote(region.aligned_texts, weights)
        return region.model_copy(update={
            "consensus_text": consensus_text,
            "source": "voting",
            "confidence": confidence,
            "needs_arbitration": False,
        })
    else:
        # High CE -> store voting fallback, flag for arbitration
        consensus_text = weighted_majority_vote(region.aligned_texts, weights)
        logger.info(
            "Region %s: high CE (%.3f), voting fallback stored, needs arbitration",
            region.region_id, ce,
        )
        return region.model_copy(update={
            "consensus_text": consensus_text,
            "source": "voting_fallback",
            "confidence": confidence,
            "needs_arbitration": True,
        })


def resolve_page(
    regions: list[AlignedRegion],
    page_num: int,
    is_ground_truth: bool = False,
    ce_threshold: float = CE_THRESHOLD,
) -> ConsensusResult:
    """Resolve all regions on a page and build reading order.

    Args:
        regions: List of AlignedRegion objects for this page.
        page_num: Zero-indexed page number.
        is_ground_truth: If True, use GROUND_TRUTH_WEIGHTS (pdfplumber 2x).
        ce_threshold: CE threshold separating voting from arbitration.

    Returns:
        ConsensusResult with resolved regions and reading order.
    """
    weights = GROUND_TRUTH_WEIGHTS if is_ground_truth else DEFAULT_WEIGHTS

    # Resolve each region
    resolved = [resolve_region(r, weights=weights, ce_threshold=ce_threshold) for r in regions]

    # Build reading order: hierarchy_level (if present) then spatial (y1, x1)
    def sort_key(region: AlignedRegion) -> tuple:
        hierarchy = float("inf")
        if region.metadata and "hierarchy_level" in region.metadata:
            hierarchy = region.metadata["hierarchy_level"]

        y1 = region.bounding_box[1]
        x1 = region.bounding_box[0]
        return (hierarchy, y1, x1)

    sorted_regions = sorted(resolved, key=sort_key)
    reading_order = [r.region_id for r in sorted_regions]

    return ConsensusResult(
        page=page_num,
        regions=resolved,
        reading_order=reading_order,
        page_metadata={"is_ground_truth": is_ground_truth},
    )


# ---------------------------------------------------------------------------
# LLM arbitration orchestration (Plan 03-05)
# ---------------------------------------------------------------------------

# Anonymous label sequence for LLM candidates
CANDIDATE_LABELS = list(string.ascii_uppercase)  # ["A", "B", "C", ...]


def crop_region_image(page_image_bytes: bytes, bounding_box: list[float]) -> bytes:
    """Crop a region from a full page image using its bounding box.

    Args:
        page_image_bytes: Full page PNG image as bytes.
        bounding_box: [x1, y1, x2, y2] in 300 DPI pixel coordinates.

    Returns:
        Cropped region PNG image as bytes.
    """
    page_img = Image.open(io.BytesIO(page_image_bytes))
    x1, y1, x2, y2 = bounding_box
    cropped = page_img.crop((int(x1), int(y1), int(x2), int(y2)))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def build_anonymous_candidates(engine_texts: dict[str, str]) -> dict[str, str]:
    """Map engine texts to anonymous labels for unbiased LLM evaluation.

    Uses sorted engine keys for deterministic ordering: first engine -> "A",
    second -> "B", etc.

    Args:
        engine_texts: Engine name -> raw text content.

    Returns:
        Dict like {"A": "text from engine 1", "B": "text from engine 2"}.
    """
    result: dict[str, str] = {}
    for i, engine_name in enumerate(sorted(engine_texts.keys())):
        label = CANDIDATE_LABELS[i]
        result[label] = engine_texts[engine_name]
    return result


def arbitrate_page(
    result: ConsensusResult,
    arbiter,
    page_image_bytes: bytes,
) -> ConsensusResult:
    """Wire LLM arbitration to high-CE regions in a ConsensusResult.

    Iterates over regions. For needs_arbitration=True regions:
    - Crops region image from the full page.
    - Builds anonymous candidates (A/B/C) from engine_texts.
    - Calls arbiter.run(image_bytes, candidates).
    - Accepted output -> overwrite consensus_text, source="arbitration", confidence=0.95.
    - Rejected output -> keep voting_fallback, source="hitl_fallback", hitl_flag=True.
    - Exception -> keep voting_fallback, source="hitl_fallback", hitl_flag=True.

    After this function, no region has source="voting_fallback" or "pending".

    Args:
        result: ConsensusResult from resolve_page (may have voting_fallback regions).
        arbiter: Duck-typed object with .run(image_bytes=bytes, candidates=dict) -> dict.
        page_image_bytes: Full page PNG image (300 DPI) as bytes.

    Returns:
        Updated ConsensusResult with all regions definitively resolved.
    """
    updated_regions: list[AlignedRegion] = []

    for region in result.regions:
        if not region.needs_arbitration:
            updated_regions.append(region)
            continue

        # Crop region from full page image
        cropped_bytes = crop_region_image(page_image_bytes, region.bounding_box)

        # Build anonymous candidates
        anonymous_candidates = build_anonymous_candidates(region.engine_texts)

        try:
            # Handwriting regions use the relaxed handwriting arbiter (LLM-07)
            # Per CONTEXT.md: when both engines produce low confidence (<0.40),
            # use modified prompt with relaxed edit distance (3->5).
            # For simplicity: any handwriting region needing arbitration uses
            # the handwriting variant (high-CE implies engine disagreement).
            if region.element_type == "handwriting" and hasattr(arbiter, "run_handwriting"):
                llm_result = arbiter.run_handwriting(
                    image_bytes=cropped_bytes,
                    candidates=anonymous_candidates,
                )
            else:
                llm_result = arbiter.run(
                    image_bytes=cropped_bytes,
                    candidates=anonymous_candidates,
                )

            if not llm_result.get("rejected", False):
                # Accepted: overwrite voting_fallback with LLM output
                updated = region.model_copy(update={
                    "consensus_text": llm_result["text"],
                    "source": "arbitration",
                    "confidence": 0.95,
                    "hitl_flag": llm_result.get("hitl_flag", False),
                    "needs_arbitration": False,
                })
            else:
                # Rejected: keep voting_fallback, flag for HITL
                logger.warning(
                    "Region %s: LLM output rejected, using voting fallback",
                    region.region_id,
                )
                updated = region.model_copy(update={
                    "source": "hitl_fallback",
                    "hitl_flag": True,
                    "needs_arbitration": False,
                })

        except Exception as e:
            # Exception: keep voting_fallback, flag for HITL
            logger.warning(
                "Region %s: LLM arbitration failed (%s), using voting fallback",
                region.region_id, e,
            )
            updated = region.model_copy(update={
                "source": "hitl_fallback",
                "hitl_flag": True,
                "needs_arbitration": False,
            })

        updated_regions.append(updated)

    return result.model_copy(update={"regions": updated_regions})

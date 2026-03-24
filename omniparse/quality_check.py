"""Text layer quality check -- determines if pdfplumber is ground truth for a page.

Compares concatenated text from pdfplumber and PaddleOCR using character-level
Levenshtein ratio (via rapidfuzz). When agreement exceeds 90%, pdfplumber is
treated as ground truth for that page.

Per CONTEXT.md decisions:
- Granularity: per-page, not per-region
- Metric: 1 - (edit_distance / max_length) via rapidfuzz.fuzz.ratio
- Threshold: >90% agreement
- Skip when pdfplumber has no text (scanned/image input)
"""
from rapidfuzz import fuzz

from omniparse.models.region import EngineOutput


def check_text_quality(
    pdfplumber_output: EngineOutput,
    paddleocr_output: EngineOutput,
    threshold: float = 0.90,
) -> dict:
    """Per-page text quality check between pdfplumber and PaddleOCR.

    Concatenates all region text from each engine and compares via
    character-level Levenshtein ratio.

    Args:
        pdfplumber_output: EngineOutput from pdfplumber for this page.
        paddleocr_output: EngineOutput from PaddleOCR for this page.
        threshold: Similarity threshold for ground truth (default 0.90).

    Returns:
        Dict with keys:
          - is_ground_truth (bool): True if pdfplumber is ground truth
          - similarity (float): Levenshtein ratio 0.0-1.0
          - reason (str): "agreement", "disagreement", or "no_pdfplumber_text"
    """
    plumber_text = " ".join(
        r.text_content for r in pdfplumber_output.regions
        if r.text_content.strip()
    )
    paddle_text = " ".join(
        r.text_content for r in paddleocr_output.regions
        if r.text_content.strip()
    )

    if not plumber_text.strip():
        return {
            "is_ground_truth": False,
            "similarity": 0.0,
            "reason": "no_pdfplumber_text",
        }

    # rapidfuzz.fuzz.ratio returns 0.0-100.0; normalize to 0.0-1.0
    similarity = fuzz.ratio(plumber_text, paddle_text) / 100.0

    return {
        "is_ground_truth": similarity > threshold,
        "similarity": round(similarity, 4),
        "reason": "agreement" if similarity > threshold else "disagreement",
    }

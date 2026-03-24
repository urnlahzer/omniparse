"""Smart dispatch module -- routes PaddleOCR-classified specialist regions to engines.

PaddleOCR is the layout authority. When it classifies a region as handwriting, formula,
or chart, that region is dispatched to the appropriate specialist engine:
- handwriting -> TrOCR
- formula -> Dots.ocr (formula mode)
- chart -> Dots.ocr (chart mode)

No confidence threshold -- if PaddleOCR labels a region as a specialist type, it is
always dispatched regardless of confidence score (per CONTEXT.md decision).

Pattern: pure functions, no Modal dependency. Dispatch results are used by the
orchestrator to invoke the appropriate Modal classes.
"""
from omniparse.models.region import Region, EngineOutput

# Element types that trigger specialist dispatch
HANDWRITING_TYPES = {"handwriting"}
FORMULA_TYPES = {"formula"}
CHART_TYPES = {"chart"}


def classify_dispatch(paddleocr_output: EngineOutput) -> dict[str, list[Region]]:
    """Classify regions for specialist dispatch based on element_type.

    Per CONTEXT.md: No confidence threshold -- if PaddleOCR labels a region
    as handwriting/formula/chart, always dispatch regardless of confidence.

    Args:
        paddleocr_output: EngineOutput from PaddleOCR with layout-classified regions.

    Returns:
        Dict with keys: "trocr", "dots_formula", "dots_chart".
        Each value is a list of Region objects to send to that engine.
    """
    dispatch: dict[str, list[Region]] = {
        "trocr": [],
        "dots_formula": [],
        "dots_chart": [],
    }

    for region in paddleocr_output.regions:
        if region.element_type in HANDWRITING_TYPES:
            dispatch["trocr"].append(region)
        elif region.element_type in FORMULA_TYPES:
            dispatch["dots_formula"].append(region)
        elif region.element_type in CHART_TYPES:
            dispatch["dots_chart"].append(region)
        # All other element_types (printed_text, table, header, etc.)
        # are handled by the always-run 3-engine consensus pipeline

    return dispatch

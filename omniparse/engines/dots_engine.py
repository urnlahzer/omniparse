"""Dots.ocr formula/chart engine -- vLLM-based VLM inference for LaTeX and SVG extraction.

Dots.ocr-1.5 converts formula regions to LaTeX text and chart regions to SVG markup.
Uses the same vLLM offline inference pattern as the LLM arbiter (llm_arbiter.py).

Pattern: pure functions (testable without GPU) + Modal class wrapper.

Requirements: SPEC-05 (formula-to-LaTeX), SPEC-06 (chart-to-SVG batch dispatch).
"""
import base64
import io
import logging

import modal
from omniparse.app import app, dots_image, model_volume, MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMULA_PROMPT = "Extract the text content from this image."
SVG_PROMPT_TEMPLATE = 'Please generate the SVG code based on the image.viewBox="0 0 {width} {height}"'

# Model path on Modal Volume (periods replaced with underscores per research pitfall #4)
DOTS_MODEL_PATH = "DotsOCR_1_5"

# ---------------------------------------------------------------------------
# Pure functions -- testable without GPU
# ---------------------------------------------------------------------------


def extract_formula(llm, image_bytes: bytes) -> dict:
    """Extract LaTeX from a formula region image via vLLM multimodal inference.

    Args:
        llm: vLLM LLM instance (loaded in Modal class).
        image_bytes: Cropped PNG of the formula region.

    Returns:
        Dict with keys: latex, success, confidence, and optionally error.
    """
    from vllm import SamplingParams

    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": FORMULA_PROMPT},
            ],
        }]

        params = SamplingParams(temperature=0.1, max_tokens=4096)
        outputs = llm.chat(messages, sampling_params=params)
        result_text = outputs[0].outputs[0].text.strip()

        success = bool(result_text.strip())
        return {
            "latex": result_text,
            "success": success,
            "confidence": 0.85 if success else 0.0,
        }
    except Exception as e:
        logger.error("extract_formula failed: %s", e)
        return {
            "latex": "",
            "success": False,
            "confidence": 0.0,
            "error": str(e),
        }


def extract_chart_svg(llm, image_bytes: bytes, width: int = 800, height: int = 600) -> dict:
    """Extract SVG from a chart region image via vLLM multimodal inference.

    Args:
        llm: vLLM LLM instance (loaded in Modal class).
        image_bytes: Cropped PNG of the chart region.
        width: SVG viewBox width.
        height: SVG viewBox height.

    Returns:
        Dict with keys: svg, success, confidence, and optionally error.
    """
    from vllm import SamplingParams

    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = SVG_PROMPT_TEMPLATE.format(width=width, height=height)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }]

        params = SamplingParams(temperature=0.1, max_tokens=24000)
        outputs = llm.chat(messages, sampling_params=params)
        result_text = outputs[0].outputs[0].text.strip()

        has_svg = "<svg" in result_text.lower()
        return {
            "svg": result_text,
            "success": has_svg,
            "confidence": 0.85 if has_svg else 0.0,
        }
    except Exception as e:
        logger.error("extract_chart_svg failed: %s", e)
        return {
            "svg": "",
            "success": False,
            "confidence": 0.0,
            "error": str(e),
        }


def validate_latex(latex: str) -> bool:
    """Validate LaTeX string for balanced delimiters and non-empty content.

    Checks:
    - Non-empty after stripping whitespace
    - Balanced $$ pairs (count must be even)
    - Balanced $ pairs (single-dollar, excluding $$)

    Args:
        latex: LaTeX string to validate.

    Returns:
        True if valid, False otherwise.
    """
    stripped = latex.strip()
    if not stripped:
        return False

    # Check balanced $$ pairs (count must be even)
    double_dollar_count = stripped.count("$$")
    if double_dollar_count % 2 != 0:
        return False

    # Check balanced single $ pairs (excluding $$)
    # Replace $$ with placeholder to count only single $
    temp = stripped.replace("$$", "")
    single_dollar_count = temp.count("$")
    if single_dollar_count % 2 != 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Modal class wrapper
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A10G",
    image=dots_image,
    volumes={MODELS_DIR: model_volume},
    timeout=120,
    min_containers=0,
    max_containers=10,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
class DotsEngine:
    """Modal class for Dots.ocr formula/chart extraction via vLLM.

    Model loaded on container start via @modal.enter. Scale-to-zero (min_containers=0)
    because formula/chart regions are a minority of document content.
    """

    @modal.enter()
    def load_model(self):
        """Load Dots.ocr-1.5 via vLLM on container start."""
        from vllm import LLM
        logger.info("Loading Dots.ocr-1.5 via vLLM...")
        self.llm = LLM(
            model=f"{MODELS_DIR}/{DOTS_MODEL_PATH}",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            enforce_eager=False,
        )
        logger.info("Dots.ocr-1.5 loaded.")

    @modal.method()
    def run_formula(self, region_image: bytes) -> dict:
        """Extract LaTeX from a formula region."""
        return extract_formula(self.llm, region_image)

    @modal.method()
    def run_chart(self, region_image: bytes, width: int = 800, height: int = 600) -> dict:
        """Extract SVG from a chart region."""
        return extract_chart_svg(self.llm, region_image, width, height)

    @modal.method()
    def run_batch(self, regions: list[dict]) -> list[dict]:
        """Process a batch of formula/chart regions.

        Each dict has: {"image_bytes": bytes, "task": "formula"|"chart", "region_id": str,
                        "width": int (chart only), "height": int (chart only)}

        This method enables Modal .map() to dispatch multiple regions to a single
        container for batch processing (SPEC-06).
        """
        results = []
        for item in regions:
            task = item.get("task", "formula")
            image_bytes = item["image_bytes"]
            if task == "chart":
                result = extract_chart_svg(
                    self.llm, image_bytes,
                    width=item.get("width", 800),
                    height=item.get("height", 600),
                )
            else:
                result = extract_formula(self.llm, image_bytes)
            result["region_id"] = item.get("region_id", "unknown")
            results.append(result)
        return results

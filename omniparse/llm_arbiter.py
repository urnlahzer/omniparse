"""LLM arbitration module -- Qwen3-VL-8B-Instruct-FP8 via vLLM for disputed regions.

The LLM arbiter is the last resort for regions where engines disagree significantly.
It receives ONLY a cropped image patch and anonymous text candidates (A/B/C).
Three-layer hallucination defense: edit distance, consecutive insertion, regex post-validation.

Pattern: pure functions (testable without GPU) + Modal class wrapper.
"""
import base64
import io
import logging
import re
from collections import OrderedDict

import modal
from rapidfuzz.distance import Levenshtein

from omniparse.app import app, llm_arbiter_image, model_volume, MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (LLM-01, LLM-02)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a deterministic OCR verifier. Examine the image patch. "
    "The candidate texts are provided below. Output ONLY the exact text visible in the "
    "image. Do not correct spelling. Do not substitute characters based on context or "
    "probability. Do not add explanations. If the patch is unreadable, output exactly: <UNREADABLE>"
)

# ---------------------------------------------------------------------------
# Hallucination guard thresholds (LLM-04, LLM-05)
# ---------------------------------------------------------------------------

MAX_EDIT_DISTANCE = 3
MAX_CONSECUTIVE_INSERTION = 5

# ---------------------------------------------------------------------------
# Handwriting-specific prompt and thresholds (LLM-07)
# ---------------------------------------------------------------------------

HANDWRITING_SYSTEM_PROMPT = (
    "You are a handwriting recognition verifier. Examine the image patch containing "
    "handwritten text. The candidate texts are provided below. Output the text that best "
    "matches the handwriting visible in the image. You may use slightly higher semantic "
    "inference than for printed text, but do not guess beyond what is visually present. "
    "Do not add explanations. If the patch is unreadable, output exactly: <UNREADABLE>"
)

HANDWRITING_MAX_EDIT_DISTANCE = 5  # Relaxed from 3 for handwriting per CONTEXT.md

# ---------------------------------------------------------------------------
# Legal field regex patterns (LLM-06)
# ---------------------------------------------------------------------------

LEGAL_FIELD_PATTERNS = OrderedDict({
    "dollar_amount": re.compile(r"\$[\d,]+(?:\.\d{2})?"),
    "date_mdy_slash": re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    "date_written": re.compile(
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    ),
    "case_citation": re.compile(r"\d+\s+[A-Z]\.\w+\s+\d+"),
    "statute_ref": re.compile(r"\d+\s+(?:U\.?S\.?C\.?|USC)\s*(?:\u00a7|SS)\s*\d+"),
    "percentage": re.compile(r"\d+(?:\.\d+)?%"),
})


# ---------------------------------------------------------------------------
# Pure functions -- testable without GPU
# ---------------------------------------------------------------------------


def encode_image(image) -> str:
    """Convert PIL Image to base64 PNG string for vLLM multimodal input.

    Args:
        image: PIL.Image.Image object.

    Returns:
        Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_edit_distance(
    llm_output: str,
    candidates: dict[str, str],
    max_distance: int = MAX_EDIT_DISTANCE,
) -> bool:
    """Check if LLM output is within edit distance threshold of ANY candidate.

    Args:
        llm_output: Text produced by the LLM.
        candidates: Label -> candidate text mapping.
        max_distance: Maximum Levenshtein edit distance to accept.

    Returns:
        True if within threshold of at least one candidate, False otherwise.
    """
    for candidate_text in candidates.values():
        dist = Levenshtein.distance(llm_output, candidate_text)
        if dist <= max_distance:
            return True
    return False


def check_consecutive_insertions(
    llm_output: str,
    candidate_texts: list[str],
    max_consecutive: int = MAX_CONSECUTIVE_INSERTION,
) -> bool:
    """Detect hallucinated text: consecutive chars not in any candidate.

    Slides a window of size (max_consecutive + 1) across llm_output. If any
    window substring does not appear in any candidate, hallucination is flagged.

    Args:
        llm_output: Text produced by the LLM.
        candidate_texts: List of original candidate texts to check against.
        max_consecutive: Maximum allowed consecutive insertion length.

    Returns:
        True if hallucination detected (consecutive insertion > threshold),
        False otherwise.
    """
    window_size = max_consecutive + 1
    if len(llm_output) < window_size:
        return False

    for i in range(len(llm_output) - window_size + 1):
        window = llm_output[i : i + window_size]
        found_in_candidate = False
        for candidate in candidate_texts:
            if window in candidate:
                found_in_candidate = True
                break
        if not found_in_candidate:
            return True
    return False


def validate_legal_fields(candidate_texts: list[str], llm_output: str) -> list[str]:
    """Regex post-validation for legal-critical fields.

    For each legal field pattern, extracts matches from llm_output and from all
    candidates. If the LLM output contains a match not found in any candidate,
    a warning is generated.

    Args:
        candidate_texts: List of original candidate texts.
        llm_output: Text produced by the LLM.

    Returns:
        List of warning strings (empty = all fields match).
    """
    warnings: list[str] = []

    for field_name, pattern in LEGAL_FIELD_PATTERNS.items():
        llm_matches = set(pattern.findall(llm_output))
        if not llm_matches:
            continue

        # Collect all matches from all candidates
        candidate_matches: set[str] = set()
        for candidate in candidate_texts:
            candidate_matches.update(pattern.findall(candidate))

        # Any LLM match not found in candidates is novel
        novel = llm_matches - candidate_matches
        for value in sorted(novel):
            warnings.append(
                f"{field_name}: LLM output contains '{value}' not found in any candidate"
            )

    return warnings


def validate_llm_output(
    llm_output: str,
    candidates: dict[str, str],
    max_edit_distance: int | None = None,
) -> dict:
    """Orchestrate all three validation layers on LLM output.

    Layer 1: Edit distance (LLM-04) -- reject if too far from all candidates.
    Layer 2: Consecutive insertion (LLM-05) -- reject if hallucinated text detected.
    Layer 3: Legal field regex (LLM-06) -- warn but do NOT reject.

    Args:
        llm_output: Text produced by the LLM.
        candidates: Label -> candidate text mapping.
        max_edit_distance: Override edit distance threshold (default: MAX_EDIT_DISTANCE=3).
            For handwriting, use HANDWRITING_MAX_EDIT_DISTANCE=5.

    Returns:
        Dict with keys: text, source, rejected, hitl_flag, warnings.
    """
    if max_edit_distance is None:
        max_edit_distance = MAX_EDIT_DISTANCE

    # Handle <UNREADABLE> special case
    if llm_output.strip() == "<UNREADABLE>":
        return {
            "text": "",
            "source": "llm_unreadable",
            "rejected": False,
            "hitl_flag": True,
            "warnings": [],
        }

    # Layer 1: Edit distance
    if not check_edit_distance(llm_output, candidates, max_distance=max_edit_distance):
        return {
            "text": llm_output,
            "source": "arbitration",
            "rejected": True,
            "hitl_flag": True,
            "warnings": ["edit_distance: output exceeds threshold from all candidates"],
        }

    # Layer 2: Consecutive insertions
    candidate_texts = list(candidates.values())
    if check_consecutive_insertions(llm_output, candidate_texts):
        return {
            "text": llm_output,
            "source": "arbitration",
            "rejected": True,
            "hitl_flag": True,
            "warnings": ["consecutive_insertion: hallucinated text detected"],
        }

    # Layer 3: Legal field regex (advisory only)
    candidate_texts_list = list(candidates.values())
    warnings = validate_legal_fields(candidate_texts_list, llm_output)

    return {
        "text": llm_output,
        "source": "arbitration",
        "rejected": False,
        "hitl_flag": len(warnings) > 0,
        "warnings": warnings,
    }


def arbitrate_region(llm, image_bytes: bytes, candidates: dict[str, str]) -> dict:
    """Arbitrate a disputed region using vLLM multimodal inference.

    Builds anonymous candidates (A/B/C), sends image + candidates to the LLM,
    and validates the output through three hallucination defense layers.

    Args:
        llm: vLLM LLM instance (loaded in Modal class).
        image_bytes: Cropped 300 DPI PNG of the disputed region.
        candidates: Anonymous label -> candidate text (e.g., {"A": "...", "B": "..."}).

    Returns:
        Dict with keys: text, source, rejected, hitl_flag, warnings.
    """
    from PIL import Image as PILImage
    from vllm import SamplingParams

    # Decode image bytes to PIL Image for base64 encoding
    image = PILImage.open(io.BytesIO(image_bytes))
    b64_image = encode_image(image)

    # Build candidate text block with anonymous labels
    candidate_lines = []
    for label in sorted(candidates.keys()):
        candidate_lines.append(f"Candidate {label}: {candidates[label]}")
    candidate_block = "\n".join(candidate_lines)

    # Construct vLLM chat messages (multimodal)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
                {
                    "type": "text",
                    "text": f"Which candidate text matches the image?\n\n{candidate_block}",
                },
            ],
        },
    ]

    # Call LLM with deterministic parameters (LLM-03)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    outputs = llm.chat(messages, sampling_params=params)
    result_text = outputs[0].outputs[0].text.strip()

    logger.info("LLM arbiter result: %r (candidates: %s)", result_text, list(candidates.keys()))

    return validate_llm_output(result_text, candidates)


def arbitrate_region_handwriting(llm, image_bytes: bytes, candidates: dict[str, str]) -> dict:
    """Arbitrate a disputed handwriting region using HANDWRITING_SYSTEM_PROMPT.

    Uses relaxed edit distance (HANDWRITING_MAX_EDIT_DISTANCE=5) and
    handwriting-specific system prompt that allows slightly higher semantic inference.

    Args:
        llm: vLLM LLM instance (loaded in Modal class).
        image_bytes: Cropped 300 DPI PNG of the disputed handwriting region.
        candidates: Anonymous label -> candidate text (e.g., {"A": "...", "B": "..."}).

    Returns:
        Dict with keys: text, source, rejected, hitl_flag, warnings.
    """
    from PIL import Image as PILImage
    from vllm import SamplingParams

    # Decode image bytes to PIL Image for base64 encoding
    image = PILImage.open(io.BytesIO(image_bytes))
    b64_image = encode_image(image)

    # Build candidate text block with anonymous labels
    candidate_lines = []
    for label in sorted(candidates.keys()):
        candidate_lines.append(f"Candidate {label}: {candidates[label]}")
    candidate_block = "\n".join(candidate_lines)

    # Construct vLLM chat messages (multimodal) with HANDWRITING prompt
    messages = [
        {"role": "system", "content": HANDWRITING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
                {
                    "type": "text",
                    "text": f"Which candidate text matches the handwriting in the image?\n\n{candidate_block}",
                },
            ],
        },
    ]

    # Call LLM with deterministic parameters (LLM-03)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    outputs = llm.chat(messages, sampling_params=params)
    result_text = outputs[0].outputs[0].text.strip()

    logger.info("LLM handwriting arbiter result: %r (candidates: %s)", result_text, list(candidates.keys()))

    return validate_llm_output(result_text, candidates, max_edit_distance=HANDWRITING_MAX_EDIT_DISTANCE)


# ---------------------------------------------------------------------------
# Modal class wrapper
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A10G",
    image=llm_arbiter_image,
    volumes={MODELS_DIR: model_volume},
    timeout=120,
    startup_timeout=600,
    min_containers=0,
    max_containers=5,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
class LLMArbiter:
    """Modal class for LLM arbitration using Qwen3-VL-8B-Instruct-FP8.

    Model loaded on container start via @modal.enter. Scale-to-zero (min_containers=0)
    because LLM arbitration is only triggered for high-CE regions (<5% of born-digital).
    """

    @modal.enter()
    def load_model(self):
        """Load Qwen3-VL-8B-Instruct-FP8 via vLLM on container start."""
        from vllm import LLM
        logger.info("Loading Qwen3-VL-8B-Instruct-FP8 via vLLM...")
        self.llm = LLM(
            model=f"{MODELS_DIR}/Qwen3-VL-8B-Instruct-FP8",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enforce_eager=False,
        )
        logger.info("Qwen3-VL-8B-Instruct-FP8 loaded.")

    @modal.method()
    def run(self, image_bytes: bytes, candidates: dict) -> dict:
        """Arbitrate a single disputed region.

        Args:
            image_bytes: Cropped 300 DPI PNG of the disputed region.
            candidates: Anonymous label -> candidate text (e.g., {"A": "...", "B": "..."}).

        Returns:
            Dict with keys: text, source, rejected, hitl_flag, warnings.
        """
        return arbitrate_region(self.llm, image_bytes, candidates)

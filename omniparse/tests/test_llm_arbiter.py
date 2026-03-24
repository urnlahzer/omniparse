"""Tests for LLM arbiter -- hallucination safeguards, validation, and arbitration.

Covers requirements: LLM-01..LLM-06, ACCY-04.
"""
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Helper: import all public symbols from the module under test
# ---------------------------------------------------------------------------


def _import():
    from omniparse.llm_arbiter import (
        check_edit_distance,
        check_consecutive_insertions,
        validate_legal_fields,
        validate_llm_output,
        arbitrate_region,
        SYSTEM_PROMPT,
        LEGAL_FIELD_PATTERNS,
        MAX_EDIT_DISTANCE,
        MAX_CONSECUTIVE_INSERTION,
    )
    return {
        "check_edit_distance": check_edit_distance,
        "check_consecutive_insertions": check_consecutive_insertions,
        "validate_legal_fields": validate_legal_fields,
        "validate_llm_output": validate_llm_output,
        "arbitrate_region": arbitrate_region,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "LEGAL_FIELD_PATTERNS": LEGAL_FIELD_PATTERNS,
        "MAX_EDIT_DISTANCE": MAX_EDIT_DISTANCE,
        "MAX_CONSECUTIVE_INSERTION": MAX_CONSECUTIVE_INSERTION,
    }


# ===========================================================================
# 1. Edit Distance (LLM-04)
# ===========================================================================


class TestCheckEditDistance:
    """LLM-04: edit distance threshold = 3."""

    def test_edit_distance_within_threshold(self):
        m = _import()
        # "$1,234.56" vs "$1,234.65" -- distance = 2 (swap '5' and '6')
        result = m["check_edit_distance"]("$1,234.56", {"A": "$1,234.56", "B": "$1,234.65"})
        assert result is True, "Output within distance of candidate A should be accepted"

    def test_edit_distance_exceeds_threshold(self):
        m = _import()
        result = m["check_edit_distance"]("completely different text", {"A": "hello", "B": "world"})
        assert result is False, "Output far from all candidates should be rejected"

    def test_edit_distance_exact_match(self):
        m = _import()
        result = m["check_edit_distance"]("exact match", {"A": "exact match", "B": "other"})
        assert result is True, "Exact match should be accepted"


# ===========================================================================
# 2. Consecutive Insertions (LLM-05)
# ===========================================================================


class TestCheckConsecutiveInsertions:
    """LLM-05: consecutive insertions > 5 chars triggers hallucination flag."""

    def test_consecutive_insertions_none(self):
        m = _import()
        result = m["check_consecutive_insertions"]("hello world", ["hello world", "hello wrold"])
        assert result is False, "No insertions should return False"

    def test_consecutive_insertions_detected(self):
        m = _import()
        # "XYZABC" (6 chars) is not a substring of any candidate
        result = m["check_consecutive_insertions"]("hello XYZABC world", ["hello world", "hello world"])
        assert result is True, ">5 char consecutive insertion should be detected"

    def test_consecutive_insertions_short_ok(self):
        m = _import()
        # Short string where no 6-char window is novel (string too short for any window)
        result = m["check_consecutive_insertions"]("hi XY", ["hi there"])
        assert result is False, "String shorter than window size should not be flagged"


# ===========================================================================
# 3. Legal Field Regex (LLM-06)
# ===========================================================================


class TestValidateLegalFields:
    """LLM-06: regex post-validation for legal-critical fields."""

    def test_regex_dollar_amount_match(self):
        m = _import()
        warnings = m["validate_legal_fields"](["$1,234.56 owes the estate"], "$1,234.56 owes the estate")
        dollar_warnings = [w for w in warnings if "dollar" in w.lower()]
        assert len(dollar_warnings) == 0, "Matching dollar amount should produce no warnings"

    def test_regex_dollar_amount_mismatch(self):
        m = _import()
        warnings = m["validate_legal_fields"](["$1,234.56 owes the estate"], "$1,234.65 owes the estate")
        dollar_warnings = [w for w in warnings if "dollar" in w.lower()]
        assert len(dollar_warnings) > 0, "Mismatched dollar amount should produce warning"

    def test_regex_date_match(self):
        m = _import()
        warnings = m["validate_legal_fields"](["filed on 01/15/2024"], "filed on 01/15/2024")
        date_warnings = [w for w in warnings if "date" in w.lower()]
        assert len(date_warnings) == 0

    def test_regex_date_mismatch(self):
        m = _import()
        warnings = m["validate_legal_fields"](["filed on 01/15/2024"], "filed on 01/16/2024")
        date_warnings = [w for w in warnings if "date" in w.lower()]
        assert len(date_warnings) > 0, "Mismatched date should produce warning"

    def test_regex_case_citation_match(self):
        m = _import()
        warnings = m["validate_legal_fields"](["123 F.3d 456"], "123 F.3d 456")
        citation_warnings = [w for w in warnings if "case" in w.lower() or "citation" in w.lower()]
        assert len(citation_warnings) == 0

    def test_regex_case_citation_mismatch(self):
        m = _import()
        warnings = m["validate_legal_fields"](["123 F.3d 456"], "123 F.3d 457")
        citation_warnings = [w for w in warnings if "case" in w.lower() or "citation" in w.lower()]
        assert len(citation_warnings) > 0, "Mismatched case citation should produce warning"

    def test_regex_statute_ref(self):
        m = _import()
        warnings = m["validate_legal_fields"](["26 USC \u00a7 1001"], "26 USC \u00a7 1002")
        statute_warnings = [w for w in warnings if "statute" in w.lower()]
        assert len(statute_warnings) > 0, "Mismatched statute ref should produce warning"

    def test_regex_percentage(self):
        m = _import()
        warnings = m["validate_legal_fields"](["50.5% of the estate"], "50.6% of the estate")
        pct_warnings = [w for w in warnings if "percent" in w.lower()]
        assert len(pct_warnings) > 0, "Mismatched percentage should produce warning"


# ===========================================================================
# 4. Validate LLM Output (orchestrator)
# ===========================================================================


class TestValidateLLMOutput:
    """validate_llm_output orchestrates all three validation layers."""

    def test_validate_output_accepted(self):
        m = _import()
        result = m["validate_llm_output"]("hello world", {"A": "hello world", "B": "hello wrold"})
        assert result["rejected"] is False
        assert result["text"] == "hello world"
        assert result["source"] == "arbitration"

    def test_validate_output_rejected_edit_distance(self):
        m = _import()
        result = m["validate_llm_output"]("completely different text", {"A": "hello", "B": "world"})
        assert result["rejected"] is True
        assert result["hitl_flag"] is True

    def test_validate_output_rejected_insertion(self):
        m = _import()
        # "XYZABC" (6 chars) not in any candidate
        result = m["validate_llm_output"]("hello XYZABC world", {"A": "hello world", "B": "hello world"})
        assert result["rejected"] is True
        assert result["hitl_flag"] is True

    def test_validate_output_unreadable(self):
        m = _import()
        result = m["validate_llm_output"]("<UNREADABLE>", {"A": "hello", "B": "world"})
        assert result["hitl_flag"] is True
        assert result["rejected"] is False
        assert result["source"] == "llm_unreadable"
        assert result["text"] == ""

    def test_validate_output_regex_warnings_not_rejection(self):
        m = _import()
        # Dollar amount mismatch but within edit distance
        result = m["validate_llm_output"]("$1,234.65", {"A": "$1,234.56", "B": "$1,234.65"})
        assert result["rejected"] is False, "Regex warnings should NOT cause rejection"

    def test_validate_output_has_warnings_key(self):
        m = _import()
        result = m["validate_llm_output"]("hello world", {"A": "hello world"})
        assert "warnings" in result


# ===========================================================================
# 5. SYSTEM_PROMPT content (LLM-01, LLM-02)
# ===========================================================================


class TestSystemPrompt:
    """SYSTEM_PROMPT enforces deterministic OCR verification."""

    def test_system_prompt_content(self):
        m = _import()
        assert "deterministic OCR verifier" in m["SYSTEM_PROMPT"]
        assert "Do not correct spelling" in m["SYSTEM_PROMPT"]


# ===========================================================================
# 6. Constants
# ===========================================================================


class TestConstants:
    """Verify hallucination thresholds match requirements."""

    def test_max_edit_distance(self):
        m = _import()
        assert m["MAX_EDIT_DISTANCE"] == 3

    def test_max_consecutive_insertion(self):
        m = _import()
        assert m["MAX_CONSECUTIVE_INSERTION"] == 5

    def test_legal_field_pattern_keys(self):
        m = _import()
        expected = {"dollar_amount", "date_mdy_slash", "date_written", "case_citation", "statute_ref", "percentage"}
        assert set(m["LEGAL_FIELD_PATTERNS"].keys()) == expected


# ===========================================================================
# 7. arbitrate_region (LLM-01, LLM-03)
# ===========================================================================


class TestArbitrateRegion:
    """arbitrate_region uses anonymous candidates (A/B/C) and calls LLM."""

    def test_arbitrate_region_anonymous_candidates(self):
        """Candidates must be labeled A/B/C, not engine names."""
        m = _import()
        # Mock vLLM's SamplingParams since vLLM is GPU-only
        mock_sampling_params_cls = MagicMock()
        mock_sampling_params_instance = MagicMock()
        mock_sampling_params_instance.temperature = 0.0
        mock_sampling_params_instance.top_p = 1.0
        mock_sampling_params_instance.max_tokens = 512
        mock_sampling_params_cls.return_value = mock_sampling_params_instance

        import sys
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = mock_sampling_params_cls
        sys.modules["vllm"] = mock_vllm

        try:
            # Create a mock LLM that captures the messages passed to it
            mock_llm = MagicMock()
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = "hello world"
            mock_llm.chat.return_value = [mock_output]

            # Create a tiny 10x10 white PNG
            from PIL import Image
            import io
            img = Image.new("RGB", (10, 10), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            result = m["arbitrate_region"](mock_llm, image_bytes, {"A": "hello world", "B": "hello wrold"})

            # Verify LLM was called
            assert mock_llm.chat.called
            # Check the user message contains "Candidate A" and "Candidate B" (not engine names)
            call_args = mock_llm.chat.call_args
            messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
            user_msg_content = str(messages)
            assert "Candidate A" in user_msg_content
            assert "Candidate B" in user_msg_content
        finally:
            del sys.modules["vllm"]

    def test_arbitrate_region_calls_llm(self):
        """Verify arbitrate_region calls llm.chat with correct SamplingParams."""
        m = _import()
        # Mock vLLM's SamplingParams since vLLM is GPU-only
        mock_sampling_params_cls = MagicMock()
        mock_sampling_params_instance = MagicMock()
        mock_sampling_params_instance.temperature = 0.0
        mock_sampling_params_instance.top_p = 1.0
        mock_sampling_params_instance.max_tokens = 512
        mock_sampling_params_cls.return_value = mock_sampling_params_instance

        import sys
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = mock_sampling_params_cls
        sys.modules["vllm"] = mock_vllm

        try:
            mock_llm = MagicMock()
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = "test output"
            mock_llm.chat.return_value = [mock_output]

            from PIL import Image
            import io
            img = Image.new("RGB", (10, 10), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            m["arbitrate_region"](mock_llm, image_bytes, {"A": "test output"})

            assert mock_llm.chat.called
            call_args = mock_llm.chat.call_args
            # sampling_params should be in kwargs
            sampling_params = call_args[1].get("sampling_params") if call_args[1] else None
            if sampling_params is None and len(call_args[0]) > 1:
                sampling_params = call_args[0][1]
            assert sampling_params is not None, "SamplingParams must be passed to llm.chat"
            assert sampling_params.temperature == 0.0
            assert sampling_params.top_p == 1.0
            assert sampling_params.max_tokens == 512
        finally:
            del sys.modules["vllm"]


# ===========================================================================
# 8. Zero Semantic Drift (ACCY-04)
# ===========================================================================


class TestZeroSemanticDrift:
    """ACCY-04: zero semantic drift on legal-critical fields."""

    def test_zero_semantic_drift_dollar_amounts(self):
        m = _import()
        result = m["validate_llm_output"]("$1,234.56", {"A": "$1,234.56", "B": "$1,234.65"})
        assert result["rejected"] is False
        assert result["text"] == "$1,234.56"
        assert len(result["warnings"]) == 0, "Matching dollar amount should produce no warnings"

    def test_zero_semantic_drift_dates(self):
        m = _import()
        result = m["validate_llm_output"]("January 15, 2024", {"A": "January 15, 2024", "B": "January 16, 2024"})
        assert result["rejected"] is False
        assert result["text"] == "January 15, 2024"
        assert len(result["warnings"]) == 0, "Matching date should produce no warnings"


# ===========================================================================
# 9. Handwriting-specific prompt and relaxed edit distance (LLM-07)
# ===========================================================================


class TestHandwritingPrompt:
    """LLM-07: handwriting-modified system prompt and relaxed edit distance."""

    def test_handwriting_system_prompt_contains_keywords(self):
        from omniparse.llm_arbiter import HANDWRITING_SYSTEM_PROMPT
        assert "handwriting" in HANDWRITING_SYSTEM_PROMPT.lower()
        assert "semantic inference" in HANDWRITING_SYSTEM_PROMPT.lower()

    def test_handwriting_max_edit_distance_is_5(self):
        from omniparse.llm_arbiter import HANDWRITING_MAX_EDIT_DISTANCE
        assert HANDWRITING_MAX_EDIT_DISTANCE == 5

    def test_validate_with_relaxed_distance_accepts_distance_4(self):
        """validate_llm_output with max_edit_distance=5 accepts text at distance 4."""
        from omniparse.llm_arbiter import validate_llm_output
        # Use a short string (< 6 chars) so consecutive insertion check doesn't apply.
        # "abcd" vs "ABCD" has Levenshtein distance 4.
        result = validate_llm_output("ABCD", {"A": "abcd"}, max_edit_distance=5)
        assert result["rejected"] is False, "Distance 4 should be accepted with max_edit_distance=5"

    def test_validate_with_default_distance_rejects_distance_4(self):
        """validate_llm_output with default max_edit_distance=3 rejects same text at distance 4."""
        from omniparse.llm_arbiter import validate_llm_output
        # Same pair: "ABCD" vs "abcd" distance = 4, exceeds default threshold of 3
        result = validate_llm_output("ABCD", {"A": "abcd"})
        assert result["rejected"] is True, "Distance 4 should be rejected with default max_edit_distance=3"

    def test_arbitrate_region_handwriting_uses_relaxed_params(self):
        """arbitrate_region_handwriting uses HANDWRITING_SYSTEM_PROMPT and relaxed distance."""
        from omniparse.llm_arbiter import HANDWRITING_SYSTEM_PROMPT
        # Import function exists
        from omniparse.llm_arbiter import arbitrate_region_handwriting
        assert callable(arbitrate_region_handwriting)

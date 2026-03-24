"""Tests for consensus decision module -- CE computation, majority voting, page orchestration.

Covers requirements: ALGN-05 (CE), ALGN-06 (majority voting), ACCY-01 (born-digital CER).
"""
import math

import pytest

from omniparse.models.consensus import AlignedRegion, ConsensusResult
from omniparse.consensus import (
    CE_THRESHOLD,
    DEFAULT_WEIGHTS,
    GROUND_TRUTH_WEIGHTS,
    NW_GAP_CHAR,
    compute_position_entropy,
    compute_region_ce,
    weighted_majority_vote,
    resolve_region,
    resolve_page,
)


# ---------------------------------------------------------------------------
# compute_position_entropy
# ---------------------------------------------------------------------------


class TestPositionEntropy:
    def test_perfect_agreement(self):
        """All engines agree -> entropy 0.0."""
        assert compute_position_entropy(["a", "a", "a"]) == 0.0

    def test_total_disagreement(self):
        """All engines differ -> entropy log2(3) ~ 1.585."""
        result = compute_position_entropy(["a", "b", "c"])
        assert result == pytest.approx(math.log2(3), abs=0.001)

    def test_two_thirds_agreement(self):
        """2/3 agreement -> entropy ~ 0.918."""
        result = compute_position_entropy(["a", "a", "b"])
        # -((2/3)*log2(2/3) + (1/3)*log2(1/3)) = 0.9183
        assert result == pytest.approx(0.9183, abs=0.001)


# ---------------------------------------------------------------------------
# compute_region_ce
# ---------------------------------------------------------------------------


class TestRegionCE:
    def test_identical_texts(self):
        """All engines have identical aligned text -> CE 0.0."""
        aligned = {
            "pdfplumber": list("hello"),
            "paddleocr": list("hello"),
            "docling": list("hello"),
        }
        assert compute_region_ce(aligned) == 0.0

    def test_partial_disagreement(self):
        """Some positions disagree -> CE between 0 and log2(3)."""
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],  # l -> 1
            "docling":    ["h", "e", "l", "l", "o"],
        }
        ce = compute_region_ce(aligned)
        assert 0.0 < ce < math.log2(3)

    def test_skips_all_gap_positions(self):
        """Positions where ALL engines have gap char are excluded from CE."""
        gap = NW_GAP_CHAR
        aligned = {
            "pdfplumber": ["a", gap, "b"],
            "paddleocr":  ["a", gap, "b"],
            "docling":    ["a", gap, "b"],
        }
        # Only positions 0 and 2 count; both are identical -> CE = 0.0
        assert compute_region_ce(aligned) == 0.0

    def test_partial_gaps_included(self):
        """Positions where SOME (not all) engines have gaps ARE included."""
        gap = NW_GAP_CHAR
        aligned = {
            "pdfplumber": ["a", gap, "b"],
            "paddleocr":  ["a", "x", "b"],
            "docling":    ["a", gap, "b"],
        }
        ce = compute_region_ce(aligned)
        # Position 1 has gap,x,gap -> not all gaps, so included; entropy > 0
        assert ce > 0.0


# ---------------------------------------------------------------------------
# weighted_majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_unanimous(self):
        """All 3 engines agree at every position -> exact text."""
        aligned = {
            "pdfplumber": list("hello"),
            "paddleocr": list("hello"),
            "docling": list("hello"),
        }
        result = weighted_majority_vote(aligned, DEFAULT_WEIGHTS)
        assert result == "hello"

    def test_two_vs_one(self):
        """pdfplumber + docling agree vs paddleocr -> winner matches majority."""
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],
            "docling":    ["h", "e", "l", "l", "o"],
        }
        result = weighted_majority_vote(aligned, DEFAULT_WEIGHTS)
        assert result == "hello"

    def test_pdfplumber_weight_tips_scale(self):
        """pdfplumber (weight=2.0) + docling (1.0) vs paddleocr (1.0).

        At position 3: pdfplumber='l' (weight 2.0), docling='l' (weight 1.0) = 3.0
                        paddleocr='1' (weight 1.0) = 1.0
        pdfplumber+docling win. This is the normal case.
        """
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],
            "docling":    ["h", "e", "l", "l", "o"],
        }
        result = weighted_majority_vote(aligned, GROUND_TRUTH_WEIGHTS)
        assert result == "hello"

    def test_pdfplumber_weight_minority_wins(self):
        """pdfplumber (weight=2.0) disagrees with paddleocr (1.0) + docling (1.0).

        At position 3: pdfplumber='l' (weight 2.0) vs paddleocr='1' (1.0) + docling='1' (1.0) = 2.0
        It's a tie (2.0 vs 2.0). In a tie, the first char encountered wins (deterministic).
        """
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],
            "docling":    ["h", "e", "l", "1", "o"],
        }
        # With ground truth weights: pdfplumber=2.0, paddle=1.0+docling=1.0 = tie
        # In a tie, the first character encountered with max weight wins
        result = weighted_majority_vote(aligned, GROUND_TRUTH_WEIGHTS)
        # Either 'l' or '1' is acceptable in a tie; we just verify it resolves
        assert result[3] in ("l", "1")
        assert len(result) == 5

    def test_gap_chars_skipped(self):
        """Gap characters don't appear in output."""
        gap = NW_GAP_CHAR
        aligned = {
            "pdfplumber": ["h", gap, "i"],
            "paddleocr":  ["h", "e", "i"],
            "docling":    ["h", gap, "i"],
        }
        result = weighted_majority_vote(aligned, DEFAULT_WEIGHTS)
        # Position 1: gap has 2 votes, 'e' has 1 vote. Gap wins by count BUT
        # gap-only positions produce no output. Since gap doesn't win unanimously
        # (one engine has 'e'), the non-gap char should be considered.
        # Actually per spec: gap chars are skipped in output, so 'e' is the only
        # non-gap vote. The result depends on implementation detail, but should
        # not contain any gap characters.
        assert gap not in result

    def test_all_gaps_position_produces_nothing(self):
        """A position where all engines have gaps produces no output character."""
        gap = NW_GAP_CHAR
        aligned = {
            "pdfplumber": ["h", gap, "i"],
            "paddleocr":  ["h", gap, "i"],
            "docling":    ["h", gap, "i"],
        }
        result = weighted_majority_vote(aligned, DEFAULT_WEIGHTS)
        assert result == "hi"


# ---------------------------------------------------------------------------
# resolve_region
# ---------------------------------------------------------------------------


class TestResolveRegion:
    def _make_region(self, **kwargs) -> AlignedRegion:
        """Helper to build AlignedRegion with sensible defaults."""
        defaults = {
            "region_id": "r_001",
            "element_type": "printed_text",
            "bounding_box": [100.0, 50.0, 2400.0, 120.0],
            "engine_texts": {"pdfplumber": "hello", "paddleocr": "hello", "docling": "hello"},
            "aligned_texts": None,
            "consensus_text": None,
            "confidence": 0.0,
            "source": "pending",
            "needs_arbitration": False,
            "hitl_flag": False,
            "metadata": None,
        }
        defaults.update(kwargs)
        return AlignedRegion(**defaults)

    def test_identical_passthrough(self):
        """source='identical' region passes through unchanged."""
        region = self._make_region(
            source="identical",
            consensus_text="hello",
            confidence=1.0,
        )
        result = resolve_region(region)
        assert result.source == "identical"
        assert result.consensus_text == "hello"
        assert result.needs_arbitration is False

    def test_single_engine_passthrough(self):
        """source='single_engine' region passes through unchanged."""
        region = self._make_region(
            source="single_engine",
            consensus_text="hello",
            confidence=0.9,
            engine_texts={"pdfplumber": "hello"},
        )
        result = resolve_region(region)
        assert result.source == "single_engine"
        assert result.consensus_text == "hello"

    def test_low_ce_voting(self):
        """CE <= 0.4 resolves via voting, source='voting'."""
        # Two engines agree, one slightly different -- low CE
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],  # one char diff
            "docling":    ["h", "e", "l", "l", "o"],
        }
        region = self._make_region(
            aligned_texts=aligned,
            engine_texts={
                "pdfplumber": "hello",
                "paddleocr": "hel1o",
                "docling": "hello",
            },
        )
        result = resolve_region(region)
        assert result.source == "voting"
        assert result.consensus_text == "hello"
        assert result.needs_arbitration is False
        assert result.confidence > 0.0

    def test_high_ce_has_voting_fallback(self):
        """CE > 0.4 sets needs_arbitration=True AND consensus_text from voting AND source='voting_fallback'.

        This is the critical test for the LLM rejection fallback per CONTEXT.md decision:
        "LLM rejection fallback: HITL flag + majority vote fallback"
        """
        # All three engines disagree significantly -> high CE
        aligned = {
            "pdfplumber": list("abcde"),
            "paddleocr":  list("vwxyz"),
            "docling":    list("12345"),
        }
        region = self._make_region(
            aligned_texts=aligned,
            engine_texts={
                "pdfplumber": "abcde",
                "paddleocr": "vwxyz",
                "docling": "12345",
            },
        )
        result = resolve_region(region)
        assert result.source == "voting_fallback"
        assert result.needs_arbitration is True
        assert result.consensus_text is not None
        assert len(result.consensus_text) > 0

    def test_high_ce_region_consensus_text_not_none(self):
        """Explicit test: high-CE region's consensus_text is NEVER None after resolve_region.

        Build an AlignedRegion with 3 very different engine texts (high CE),
        call resolve_region, assert consensus_text is a non-empty string.
        """
        # Construct maximally different texts to guarantee high CE
        aligned = {
            "pdfplumber": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "paddleocr":  ["1", "2", "3", "4", "5", "6", "7", "8"],
            "docling":    ["x", "y", "z", "w", "v", "u", "t", "s"],
        }
        region = self._make_region(
            aligned_texts=aligned,
            engine_texts={
                "pdfplumber": "ABCDEFGH",
                "paddleocr": "12345678",
                "docling": "xyzwvuts",
            },
        )
        result = resolve_region(region)
        assert result.consensus_text is not None, "High-CE region must have voting fallback text"
        assert isinstance(result.consensus_text, str)
        assert len(result.consensus_text) > 0
        assert result.source == "voting_fallback"
        assert result.needs_arbitration is True

    def test_no_aligned_texts_passthrough(self):
        """Region with aligned_texts=None passes through unchanged."""
        region = self._make_region(aligned_texts=None, source="pending")
        result = resolve_region(region)
        assert result.source == "pending"


# ---------------------------------------------------------------------------
# resolve_page
# ---------------------------------------------------------------------------


class TestResolvePage:
    def _make_region(self, region_id, source="identical", consensus_text="text",
                     bbox=None, metadata=None, **kwargs) -> AlignedRegion:
        defaults = {
            "region_id": region_id,
            "element_type": "printed_text",
            "bounding_box": bbox or [100.0, 50.0, 2400.0, 120.0],
            "engine_texts": {"pdfplumber": consensus_text},
            "consensus_text": consensus_text,
            "confidence": 1.0,
            "source": source,
            "needs_arbitration": False,
            "hitl_flag": False,
            "metadata": metadata,
        }
        defaults.update(kwargs)
        return AlignedRegion(**defaults)

    def test_reading_order_by_hierarchy_level(self):
        """Regions with Docling hierarchy_level are sorted by level first."""
        r1 = self._make_region(
            "r1", bbox=[100, 200, 500, 250],
            metadata={"hierarchy_level": 2},
        )
        r2 = self._make_region(
            "r2", bbox=[100, 50, 500, 100],
            metadata={"hierarchy_level": 1},
        )
        r3 = self._make_region(
            "r3", bbox=[100, 300, 500, 350],
            metadata={"hierarchy_level": 3},
        )
        result = resolve_page([r1, r2, r3], page_num=0)
        assert result.reading_order == ["r2", "r1", "r3"]

    def test_reading_order_spatial_fallback(self):
        """Regions without hierarchy_level fall back to spatial ordering (y1, then x1)."""
        r1 = self._make_region("r1", bbox=[100, 300, 500, 350])
        r2 = self._make_region("r2", bbox=[100, 100, 500, 150])
        r3 = self._make_region("r3", bbox=[600, 100, 1000, 150])  # same y, right of r2
        result = resolve_page([r1, r2, r3], page_num=0)
        assert result.reading_order == ["r2", "r3", "r1"]

    def test_ground_truth_weights_used(self):
        """When is_ground_truth=True, pdfplumber gets 2x weight."""
        # Construct a region where pdfplumber disagrees with others
        aligned = {
            "pdfplumber": ["h", "e", "l", "l", "o"],
            "paddleocr":  ["h", "e", "l", "1", "o"],
            "docling":    ["h", "e", "l", "l", "o"],
        }
        region = AlignedRegion(
            region_id="r1",
            element_type="printed_text",
            bounding_box=[100, 50, 500, 100],
            engine_texts={"pdfplumber": "hello", "paddleocr": "hel1o", "docling": "hello"},
            aligned_texts=aligned,
            source="pending",
        )
        result = resolve_page([region], page_num=0, is_ground_truth=True)
        assert result.page_metadata == {"is_ground_truth": True}
        # With ground truth weights, pdfplumber (2.0) + docling (1.0) = 3.0 vs paddleocr (1.0)
        assert result.regions[0].consensus_text == "hello"

    def test_born_digital_cer_zero(self):
        """Born-digital page where pdfplumber is ground truth and all texts identical -> CER 0%.

        This validates ACCY-01: born-digital pages achieve near-0% CER.
        """
        pdfplumber_text = "LAST WILL AND TESTAMENT"
        r1 = self._make_region(
            "r1",
            source="identical",
            consensus_text=pdfplumber_text,
            bbox=[100, 50, 2400, 120],
        )
        r2 = self._make_region(
            "r2",
            source="identical",
            consensus_text="Article I: Definitions",
            bbox=[100, 150, 2400, 200],
        )
        result = resolve_page([r1, r2], page_num=0, is_ground_truth=True)
        # All regions already "identical" -> consensus matches pdfplumber exactly
        assert result.regions[0].consensus_text == pdfplumber_text
        assert result.regions[1].consensus_text == "Article I: Definitions"
        # CER = 0 since consensus == pdfplumber for all regions
        for region in result.regions:
            assert region.source == "identical"
            assert region.confidence == 1.0

    def test_returns_consensus_result(self):
        """resolve_page returns a ConsensusResult with correct structure."""
        r1 = self._make_region("r1")
        result = resolve_page([r1], page_num=3)
        assert isinstance(result, ConsensusResult)
        assert result.page == 3
        assert len(result.regions) == 1
        assert result.reading_order == ["r1"]


# ---------------------------------------------------------------------------
# arbitrate_page (Plan 03-05: LLM arbitration wiring)
# ---------------------------------------------------------------------------


class TestArbitratePage:
    """Tests for arbitrate_page orchestration function.

    Covers: ALGN-05, LLM-01, LLM-04, LLM-05, LLM-06, ACCY-04.
    """

    @staticmethod
    def _make_page_image_bytes(width: int = 2550, height: int = 3300) -> bytes:
        """Create a minimal white PNG image as page fixture (300 DPI letter size)."""
        from PIL import Image as PILImage
        import io

        img = PILImage.new("RGB", (width, height), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _make_region(**kwargs) -> AlignedRegion:
        """Helper to build AlignedRegion with sensible defaults."""
        defaults = {
            "region_id": "r_001",
            "element_type": "printed_text",
            "bounding_box": [100.0, 50.0, 500.0, 120.0],
            "engine_texts": {
                "pdfplumber": "hello world",
                "paddleocr": "hello world",
                "docling": "hello world",
            },
            "aligned_texts": None,
            "consensus_text": None,
            "confidence": 0.0,
            "source": "pending",
            "needs_arbitration": False,
            "hitl_flag": False,
            "metadata": None,
        }
        defaults.update(kwargs)
        return AlignedRegion(**defaults)

    def test_arbitrate_page_skips_non_arbitration_regions(self):
        """Regions with needs_arbitration=False should not trigger arbiter.run()."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="identical", consensus_text="text1",
            confidence=1.0, needs_arbitration=False,
        )
        r2 = self._make_region(
            region_id="r2", source="voting", consensus_text="text2",
            confidence=0.9, needs_arbitration=False,
        )
        page = ConsensusResult(page=0, regions=[r1, r2], reading_order=["r1", "r2"])
        mock_arbiter = MagicMock()
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        mock_arbiter.run.assert_not_called()
        assert result.regions[0].source == "identical"
        assert result.regions[1].source == "voting"

    def test_arbitrate_page_accepted_llm_overwrites_fallback(self):
        """Accepted LLM output should overwrite voting_fallback with source='arbitration'."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="voting_fallback",
            consensus_text="fallback text", confidence=0.3,
            needs_arbitration=True,
            engine_texts={"pdfplumber": "better text", "paddleocr": "beter text", "docling": "best text"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])
        mock_arbiter = MagicMock()
        mock_arbiter.run.return_value = {
            "text": "better text",
            "source": "arbitration",
            "rejected": False,
            "hitl_flag": False,
            "warnings": [],
        }
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        assert result.regions[0].consensus_text == "better text"
        assert result.regions[0].source == "arbitration"
        assert result.regions[0].needs_arbitration is False
        assert result.regions[0].confidence == 0.95

    def test_arbitrate_page_rejected_llm_preserves_fallback(self):
        """Rejected LLM output should preserve voting_fallback and set hitl_flag."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="voting_fallback",
            consensus_text="fallback text", confidence=0.3,
            needs_arbitration=True,
            engine_texts={"pdfplumber": "text a", "paddleocr": "text b", "docling": "text c"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])
        mock_arbiter = MagicMock()
        mock_arbiter.run.return_value = {
            "text": "",
            "source": "arbitration",
            "rejected": True,
            "hitl_flag": True,
            "warnings": ["edit distance exceeded"],
        }
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        assert result.regions[0].consensus_text == "fallback text"  # preserved
        assert result.regions[0].source == "hitl_fallback"
        assert result.regions[0].hitl_flag is True
        assert result.regions[0].needs_arbitration is False

    def test_arbitrate_page_exception_preserves_fallback(self):
        """Arbiter exception should preserve voting_fallback and set hitl_flag."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="voting_fallback",
            consensus_text="fallback text", confidence=0.3,
            needs_arbitration=True,
            engine_texts={"pdfplumber": "text a", "paddleocr": "text b", "docling": "text c"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])
        mock_arbiter = MagicMock()
        mock_arbiter.run.side_effect = RuntimeError("GPU OOM")
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        assert result.regions[0].consensus_text == "fallback text"  # preserved
        assert result.regions[0].source == "hitl_fallback"
        assert result.regions[0].hitl_flag is True
        assert result.regions[0].needs_arbitration is False

    def test_arbitrate_page_anonymous_candidates(self):
        """Candidates passed to arbiter.run should have anonymous A/B/C keys, not engine names."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="voting_fallback",
            consensus_text="fallback", confidence=0.3,
            needs_arbitration=True,
            engine_texts={"pdfplumber": "alpha", "paddleocr": "beta", "docling": "gamma"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])

        captured_kwargs = {}

        def capture_run(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "text": "alpha",
                "source": "arbitration",
                "rejected": False,
                "hitl_flag": False,
                "warnings": [],
            }

        mock_arbiter = MagicMock()
        mock_arbiter.run.side_effect = capture_run
        page_image = self._make_page_image_bytes()

        arbitrate_page(page, mock_arbiter, page_image)

        assert mock_arbiter.run.called
        call_kwargs = mock_arbiter.run.call_args
        # Candidates should be dict with keys from CANDIDATE_LABELS (A, B, C), not engine names
        candidates = call_kwargs.kwargs.get("candidates") or call_kwargs[1].get("candidates")
        if candidates is None:
            # May be positional
            candidates = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs.kwargs["candidates"]
        assert set(candidates.keys()) == {"A", "B", "C"}
        assert "pdfplumber" not in candidates
        assert "paddleocr" not in candidates
        assert "docling" not in candidates

    def test_arbitrate_page_all_regions_resolved(self):
        """After arbitrate_page, no region should have source='voting_fallback' or 'pending'."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="identical",
            consensus_text="same", confidence=1.0,
        )
        r2 = self._make_region(
            region_id="r2", source="voting",
            consensus_text="voted", confidence=0.8,
        )
        r3 = self._make_region(
            region_id="r3", source="voting_fallback",
            consensus_text="fallback", confidence=0.3,
            needs_arbitration=True,
            engine_texts={"pdfplumber": "text a", "paddleocr": "text b", "docling": "text c"},
        )
        page = ConsensusResult(page=0, regions=[r1, r2, r3], reading_order=["r1", "r2", "r3"])
        mock_arbiter = MagicMock()
        mock_arbiter.run.return_value = {
            "text": "text a",
            "source": "arbitration",
            "rejected": False,
            "hitl_flag": False,
            "warnings": [],
        }
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        for region in result.regions:
            assert region.source not in ("voting_fallback", "pending"), (
                f"Region {region.region_id} has unresolved source={region.source}"
            )

    def test_arbitrate_page_crops_image(self):
        """Arbiter should receive cropped image bytes, not the full page image."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1", source="voting_fallback",
            consensus_text="fallback", confidence=0.3,
            needs_arbitration=True,
            bounding_box=[100.0, 50.0, 500.0, 120.0],
            engine_texts={"pdfplumber": "alpha", "paddleocr": "beta", "docling": "gamma"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])
        mock_arbiter = MagicMock()
        mock_arbiter.run.return_value = {
            "text": "alpha",
            "source": "arbitration",
            "rejected": False,
            "hitl_flag": False,
            "warnings": [],
        }
        page_image = self._make_page_image_bytes()

        arbitrate_page(page, mock_arbiter, page_image)

        call_kwargs = mock_arbiter.run.call_args
        image_arg = call_kwargs.kwargs.get("image_bytes") or call_kwargs[0][0]
        # Cropped image should be different from full page image (smaller)
        assert image_arg != page_image, "Arbiter received full page image instead of cropped region"
        assert len(image_arg) < len(page_image), "Cropped image should be smaller than full page"


    def test_arbitrate_page_routes_handwriting_to_run_handwriting(self):
        """LLM-07: handwriting regions use run_handwriting (relaxed edit distance)."""
        from unittest.mock import MagicMock
        from omniparse.consensus import arbitrate_page

        r1 = self._make_region(
            region_id="r1",
            element_type="handwriting",
            source="voting_fallback",
            consensus_text="fallback text",
            confidence=0.3,
            needs_arbitration=True,
            engine_texts={"trocr": "alpha", "paddleocr": "beta"},
        )
        page = ConsensusResult(page=0, regions=[r1], reading_order=["r1"])
        mock_arbiter = MagicMock()
        mock_arbiter.run_handwriting.return_value = {
            "text": "corrected",
            "source": "arbitration",
            "rejected": False,
            "hitl_flag": False,
            "warnings": [],
        }
        page_image = self._make_page_image_bytes()

        result = arbitrate_page(page, mock_arbiter, page_image)

        mock_arbiter.run_handwriting.assert_called_once()
        mock_arbiter.run.assert_not_called()
        assert result.regions[0].source == "arbitration"
        assert result.regions[0].consensus_text == "corrected"


# ---------------------------------------------------------------------------
# Handwriting-aware weights (Plan 04-01: SPEC-04)
# ---------------------------------------------------------------------------


class TestHandwritingWeights:
    """SPEC-04: TrOCR gets 2x voting weight for handwriting regions."""

    def test_handwriting_weights_constant_exists(self):
        """HANDWRITING_WEIGHTS must exist with trocr=2.0, paddleocr=1.0."""
        from omniparse.consensus import HANDWRITING_WEIGHTS
        assert HANDWRITING_WEIGHTS["trocr"] == 2.0
        assert HANDWRITING_WEIGHTS["paddleocr"] == 1.0

    def test_resolve_region_uses_handwriting_weights(self):
        """resolve_region on a handwriting region uses HANDWRITING_WEIGHTS, not DEFAULT_WEIGHTS."""
        from omniparse.consensus import HANDWRITING_WEIGHTS, resolve_region, weighted_majority_vote

        # trocr says "Smith", paddleocr says "Snith" -> trocr should win with 2x weight
        aligned = {
            "trocr":     ["S", "m", "i", "t", "h"],
            "paddleocr": ["S", "n", "i", "t", "h"],
        }
        region = AlignedRegion(
            region_id="r_hw_001",
            element_type="handwriting",
            bounding_box=[50.0, 500.0, 350.0, 580.0],
            engine_texts={"trocr": "Smith", "paddleocr": "Snith"},
            aligned_texts=aligned,
            source="pending",
        )
        result = resolve_region(region)
        # With handwriting weights, trocr (2.0) wins over paddleocr (1.0) at position 1
        assert result.consensus_text == "Smith"
        assert result.source == "voting"

    def test_resolve_page_detects_handwriting_and_applies_weights(self):
        """resolve_page auto-detects handwriting regions and applies HANDWRITING_WEIGHTS."""
        # trocr says "Smith", paddleocr says "Snith"
        aligned = {
            "trocr":     ["S", "m", "i", "t", "h"],
            "paddleocr": ["S", "n", "i", "t", "h"],
        }
        region = AlignedRegion(
            region_id="r_hw_001",
            element_type="handwriting",
            bounding_box=[50.0, 500.0, 350.0, 580.0],
            engine_texts={"trocr": "Smith", "paddleocr": "Snith"},
            aligned_texts=aligned,
            source="pending",
        )
        result = resolve_page([region], page_num=0)
        assert result.regions[0].consensus_text == "Smith"

    def test_weighted_majority_vote_with_handwriting_weights(self):
        """weighted_majority_vote with HANDWRITING_WEIGHTS favors trocr when tied."""
        from omniparse.consensus import HANDWRITING_WEIGHTS, weighted_majority_vote

        aligned = {
            "trocr":     ["A", "B"],
            "paddleocr": ["A", "X"],
        }
        result = weighted_majority_vote(aligned, HANDWRITING_WEIGHTS)
        # Position 0: both agree "A". Position 1: trocr "B" (2.0) vs paddle "X" (1.0) -> "B" wins
        assert result == "AB"

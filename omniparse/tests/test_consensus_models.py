"""Tests for Phase 3 consensus data models."""
import pytest

from omniparse.models.consensus import AlignedRegion, ConsensusResult, ArbitrationRequest


class TestAlignedRegion:
    def test_minimal_creation(self):
        ar = AlignedRegion(
            region_id="r_001",
            element_type="printed_text",
            bounding_box=[100.0, 50.0, 2400.0, 120.0],
            engine_texts={"pdfplumber": "hello"},
        )
        assert ar.region_id == "r_001"
        assert ar.source == "pending"
        assert ar.needs_arbitration is False
        assert ar.hitl_flag is False
        assert ar.confidence == 0.0

    def test_all_fields(self):
        ar = AlignedRegion(
            region_id="r_002",
            element_type="table",
            bounding_box=[0.0, 0.0, 100.0, 100.0],
            engine_texts={"pdfplumber": "a", "paddleocr": "b"},
            aligned_texts={"pdfplumber": ["a"], "paddleocr": ["b"]},
            consensus_text="a",
            confidence=0.95,
            source="identical",
            needs_arbitration=False,
            hitl_flag=False,
            metadata={"table_structure": {"rows": 2, "cols": 2}},
        )
        assert ar.consensus_text == "a"
        assert ar.confidence == 0.95
        assert ar.source == "identical"

    def test_bounding_box_validation(self):
        with pytest.raises(Exception):
            AlignedRegion(
                region_id="r_bad",
                element_type="printed_text",
                bounding_box=[1.0, 2.0],  # too short
                engine_texts={"pdfplumber": "x"},
            )

    def test_source_description_includes_voting_fallback(self):
        """AlignedRegion.source field description includes voting_fallback."""
        field = AlignedRegion.model_fields["source"]
        assert "voting_fallback" in field.description


class TestConsensusResult:
    def test_minimal_creation(self):
        cr = ConsensusResult(page=0)
        assert cr.page == 0
        assert cr.regions == []
        assert cr.reading_order == []
        assert cr.page_metadata is None

    def test_with_regions(self):
        ar = AlignedRegion(
            region_id="r_001",
            element_type="printed_text",
            bounding_box=[0.0, 0.0, 100.0, 100.0],
            engine_texts={"pdfplumber": "hello"},
        )
        cr = ConsensusResult(page=0, regions=[ar], reading_order=["r_001"])
        assert len(cr.regions) == 1
        assert cr.reading_order == ["r_001"]

    def test_page_validation(self):
        with pytest.raises(Exception):
            ConsensusResult(page=-1)


class TestArbitrationRequest:
    def test_creation(self):
        ar = ArbitrationRequest(
            region_id="r_001",
            image_bytes=b"\x89PNG",
            candidates={"A": "text a", "B": "text b"},
            element_type="printed_text",
            bounding_box=[0.0, 0.0, 100.0, 100.0],
        )
        assert ar.region_id == "r_001"
        assert ar.candidates == {"A": "text a", "B": "text b"}

    def test_bounding_box_validation(self):
        with pytest.raises(Exception):
            ArbitrationRequest(
                region_id="r_bad",
                image_bytes=b"x",
                candidates={"A": "a"},
                element_type="printed_text",
                bounding_box=[1.0, 2.0, 3.0, 4.0, 5.0],  # too long
            )


class TestConfestFixtures:
    def test_three_engine_outputs_fixture(self, three_engine_outputs):
        pdf, paddle, docling = three_engine_outputs
        assert pdf.engine == "pdfplumber"
        assert paddle.engine == "paddleocr"
        assert docling.engine == "docling"
        # Each has 3 regions (2 text + 1 table)
        assert len(pdf.regions) == 3
        assert len(paddle.regions) == 3
        assert len(docling.regions) == 3
        # Region 1 identical text
        assert pdf.regions[0].text_content == "LAST WILL AND TESTAMENT"
        assert paddle.regions[0].text_content == "LAST WILL AND TESTAMENT"
        assert docling.regions[0].text_content == "LAST WILL AND TESTAMENT"
        # Region 2 differing text
        assert pdf.regions[1].text_content == "Article I: Definitions"
        assert paddle.regions[1].text_content == "Artlcle I: Definltions"
        assert docling.regions[1].text_content == "Article I: Definitions"

    def test_two_engine_outputs_fixture(self, two_engine_outputs):
        pdf, paddle = two_engine_outputs
        assert pdf.engine == "pdfplumber"
        assert paddle.engine == "paddleocr"
        assert len(pdf.regions) == 3
        assert len(paddle.regions) == 3

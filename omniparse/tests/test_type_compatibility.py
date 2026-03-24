"""Tests for type compatibility map -- NORM-03."""
import pytest
from omniparse.type_compatibility import TYPE_COMPAT_GROUPS, are_types_compatible
from omniparse.models.region import VALID_ELEMENT_TYPES


class TestTypeCompatGroups:
    def test_all_valid_types_mapped(self):
        """Every VALID_ELEMENT_TYPE has a compatibility group."""
        for et in VALID_ELEMENT_TYPES:
            assert et in TYPE_COMPAT_GROUPS, f"{et} missing from TYPE_COMPAT_GROUPS"

    def test_text_group_cross_matching(self):
        """printed_text, header, footer, page_number are all compatible."""
        text_types = ["printed_text", "header", "footer", "page_number"]
        for a in text_types:
            for b in text_types:
                assert are_types_compatible(a, b), f"{a} should be compatible with {b}"

    def test_table_isolation(self):
        """Tables only match tables, not printed_text or header."""
        assert are_types_compatible("table", "table")
        assert not are_types_compatible("table", "printed_text")
        assert not are_types_compatible("table", "header")

    def test_handwriting_isolation(self):
        """Handwriting only matches handwriting."""
        assert are_types_compatible("handwriting", "handwriting")
        assert not are_types_compatible("handwriting", "printed_text")
        assert not are_types_compatible("handwriting", "header")

    def test_specialist_group(self):
        """formula, chart, image, seal are compatible with each other."""
        specialist_types = ["formula", "chart", "image", "seal"]
        for a in specialist_types:
            for b in specialist_types:
                assert are_types_compatible(a, b), f"{a} should be compatible with {b}"

    def test_specialist_not_text(self):
        """Specialist types do not match text types."""
        assert not are_types_compatible("formula", "printed_text")
        assert not are_types_compatible("chart", "header")

    def test_unknown_type_exact_match(self):
        """Unknown types fall back to exact string match."""
        assert are_types_compatible("unknown", "unknown")
        assert not are_types_compatible("unknown", "printed_text")
        assert not are_types_compatible("unknown", "other_unknown")

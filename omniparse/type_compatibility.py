"""Type compatibility map for cross-type spatial matching.

Compatibility groups allow spatial matching across related element types.
Original element_type is preserved on each region (D-07) -- compatibility
is for matching only, not relabeling.

Groups (per D-05):
  - "text": printed_text, header, footer, page_number
  - "table": table (standalone)
  - "handwriting": handwriting (standalone)
  - "specialist": formula, chart, image, seal (standalone)
"""

TYPE_COMPAT_GROUPS: dict[str, str] = {
    "printed_text": "text",
    "header": "text",
    "footer": "text",
    "page_number": "text",
    "table": "table",
    "handwriting": "handwriting",
    "formula": "specialist",
    "chart": "specialist",
    "image": "specialist",
    "seal": "specialist",
}


def are_types_compatible(type_a: str, type_b: str) -> bool:
    """Check if two element types can be spatially matched.

    Types in the same compatibility group can match. Unknown types
    fall back to exact string match.
    """
    group_a = TYPE_COMPAT_GROUPS.get(type_a)
    group_b = TYPE_COMPAT_GROUPS.get(type_b)
    if group_a is None or group_b is None:
        return type_a == type_b
    return group_a == group_b

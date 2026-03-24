"""Baseline accuracy thresholds for regression detection.

Each entry maps a benchmark document name to its expected accuracy metrics.
Regressions are flagged when metrics fall below these thresholds.
"""

# Baseline thresholds per benchmark document
# Keys: benchmark document name (matches filename in benchmarks volume)
# Values: dict of metric_name -> minimum acceptable value
#   - cer: Character Error Rate (lower is better; regression if actual > expected)
#   - table_teds: Table Tree-Edit-Distance-based Similarity (higher is better; regression if actual < expected)
BASELINES: dict[str, dict[str, float]] = {
    "clean_digital_legal.pdf": {
        "cer": 0.01,          # CER < 1% (ACCY-01)
        "table_teds": 0.95,   # TableTEDS > 95% (ACCY-02)
    },
    "scanned_legal.pdf": {
        "cer": 0.03,          # CER < 3% (scanned tolerance)
    },
    "handwriting_sample.pdf": {
        "cer": 0.05,          # CER < 5% (ACCY-03)
    },
}

# Global thresholds
MAX_LLM_INVOCATION_RATE_DIGITAL = 0.05   # < 5% on born-digital (ACCY-05)
MAX_LLM_INVOCATION_RATE_SCANNED = 0.15   # < 15% on scanned (ACCY-05)

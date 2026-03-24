"""Regression test runner -- pure functions for benchmark processing and metric comparison.

The runner processes benchmark documents through the pipeline, computes accuracy
metrics (CER), and compares against stored baselines to detect regressions.
Modal-specific code (cron, Volume, Dict) lives in app.py -- this module is testable locally.
"""
from __future__ import annotations

import logging
from typing import Callable

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

# Metrics where higher values indicate worse quality (regression = actual > expected)
_LOWER_IS_BETTER = {"cer"}

# Metrics where lower values indicate worse quality (regression = actual < expected)
_HIGHER_IS_BETTER = {"table_teds"}


def compute_cer(predicted: str, ground_truth: str) -> float:
    """Compute Character Error Rate between predicted and ground truth text.

    CER = Levenshtein distance / max(len(ground_truth), 1)

    Args:
        predicted: OCR output text.
        ground_truth: Reference text.

    Returns:
        CER as a float >= 0.0.  Values > 1.0 are possible when predicted
        is longer than ground_truth.
    """
    distance = Levenshtein.distance(predicted, ground_truth)
    return distance / max(len(ground_truth), 1)


def check_baselines(
    results: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
) -> list[dict]:
    """Compare benchmark results against baseline thresholds.

    For each metric in baselines:
    - CER (lower is better): regression if actual > expected
    - table_teds (higher is better): regression if actual < expected

    Args:
        results: Mapping of benchmark name to metric values.
        baselines: Mapping of benchmark name to threshold values.

    Returns:
        List of regression entries.  Empty list means all benchmarks passing.
    """
    regressions: list[dict] = []

    for doc_name, thresholds in baselines.items():
        if doc_name not in results:
            continue

        doc_results = results[doc_name]

        for metric, expected in thresholds.items():
            actual = doc_results.get(metric)
            if actual is None:
                continue

            is_regression = False
            if metric in _LOWER_IS_BETTER:
                is_regression = actual > expected
            elif metric in _HIGHER_IS_BETTER:
                is_regression = actual < expected

            if is_regression:
                regressions.append({
                    "benchmark": doc_name,
                    "metric": metric,
                    "expected": expected,
                    "actual": actual,
                    "regression": True,
                })

    return regressions


def run_benchmark(
    doc_name: str,
    doc_bytes: bytes,
    ground_truth_text: str,
    pipeline_fn: Callable,
) -> dict[str, float]:
    """Process a single benchmark document and compute accuracy metrics.

    Args:
        doc_name: Filename of the benchmark document.
        doc_bytes: Raw document bytes.
        ground_truth_text: Reference text for CER computation.
        pipeline_fn: Callable that processes (doc_bytes, doc_name) and returns
            a dict with at minimum a ``markdown`` key.

    Returns:
        Dict of metric name to value (e.g. ``{"cer": 0.012}``).
    """
    result = pipeline_fn(doc_bytes, doc_name)
    predicted = result.get("markdown", "")
    cer = compute_cer(predicted, ground_truth_text)
    return {"cer": cer}


def format_regression_report(regressions: list[dict]) -> str:
    """Format regression results as a human-readable Markdown report.

    Args:
        regressions: List of regression entries from ``check_baselines``.

    Returns:
        Markdown string with header and regression table (or passing message).
    """
    lines = ["# OmniParse Regression Report", ""]

    if not regressions:
        lines.append("All benchmarks passing.")
        return "\n".join(lines)

    lines.append(f"**{len(regressions)} regression(s) detected.**")
    lines.append("")
    lines.append("| Benchmark | Metric | Expected | Actual |")
    lines.append("|-----------|--------|----------|--------|")

    for entry in regressions:
        lines.append(
            f"| {entry['benchmark']} | {entry['metric']} "
            f"| {entry['expected']:.4f} | {entry['actual']:.4f} |"
        )

    return "\n".join(lines)

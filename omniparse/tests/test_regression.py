"""Tests for the regression test runner -- CER computation, baseline checking, and report formatting."""
from unittest.mock import MagicMock

import pytest

from omniparse.regression.runner import (
    compute_cer,
    check_baselines,
    run_benchmark,
    format_regression_report,
)
from omniparse.regression.baselines import BASELINES


# ---------------------------------------------------------------------------
# compute_cer
# ---------------------------------------------------------------------------

def test_cer_identical_strings():
    assert compute_cer("hello", "hello") == 0.0


def test_cer_completely_different():
    assert compute_cer("abc", "xyz") == 1.0


def test_cer_partial_match():
    cer = compute_cer("hello world", "hello worlx")
    assert 0.0 < cer < 0.5


def test_cer_empty_ground_truth():
    """Empty ground truth divides by max(0, 1) = 1."""
    cer = compute_cer("abc", "")
    assert cer == 3.0  # Levenshtein("abc", "") = 3, denominator = 1


# ---------------------------------------------------------------------------
# check_baselines
# ---------------------------------------------------------------------------

def test_check_baselines_all_passing():
    results = {
        "clean_digital_legal.pdf": {"cer": 0.005, "table_teds": 0.98},
        "scanned_legal.pdf": {"cer": 0.02},
        "handwriting_sample.pdf": {"cer": 0.04},
    }
    regressions = check_baselines(results, BASELINES)
    assert regressions == []


def test_check_baselines_cer_regression():
    results = {
        "clean_digital_legal.pdf": {"cer": 0.05, "table_teds": 0.98},
    }
    regressions = check_baselines(results, BASELINES)
    assert len(regressions) == 1
    assert regressions[0]["benchmark"] == "clean_digital_legal.pdf"
    assert regressions[0]["metric"] == "cer"
    assert regressions[0]["expected"] == 0.01
    assert regressions[0]["actual"] == 0.05
    assert regressions[0]["regression"] is True


def test_check_baselines_teds_regression():
    results = {
        "clean_digital_legal.pdf": {"cer": 0.005, "table_teds": 0.90},
    }
    regressions = check_baselines(results, BASELINES)
    assert len(regressions) == 1
    assert regressions[0]["metric"] == "table_teds"
    assert regressions[0]["expected"] == 0.95
    assert regressions[0]["actual"] == 0.90


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def test_run_benchmark_integration():
    """Mock pipeline function returns dict with markdown key."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {
        "markdown": "Hello world",
        "metadata": {"page_count": 1},
    }

    result = run_benchmark(
        doc_name="test.pdf",
        doc_bytes=b"fake-bytes",
        ground_truth_text="Hello worlx",
        pipeline_fn=mock_pipeline,
    )

    mock_pipeline.assert_called_once_with(b"fake-bytes", "test.pdf")
    assert "cer" in result
    assert result["cer"] > 0.0  # "world" vs "worlx"


# ---------------------------------------------------------------------------
# format_regression_report
# ---------------------------------------------------------------------------

def test_format_regression_report_passing():
    report = format_regression_report([])
    assert "All benchmarks passing." in report
    assert "# OmniParse Regression Report" in report


def test_format_regression_report_with_regressions():
    regressions = [
        {
            "benchmark": "clean_digital_legal.pdf",
            "metric": "cer",
            "expected": 0.01,
            "actual": 0.05,
            "regression": True,
        }
    ]
    report = format_regression_report(regressions)
    assert "# OmniParse Regression Report" in report
    assert "clean_digital_legal.pdf" in report
    assert "cer" in report
    assert "0.0100" in report
    assert "0.0500" in report
    assert "1 regression(s)" in report


# ---------------------------------------------------------------------------
# Cron schedule verification (source inspection)
# ---------------------------------------------------------------------------

def test_cron_schedule_defined():
    """Verify that app.py defines the cron schedule and function."""
    from pathlib import Path

    app_source = Path(__file__).parent.parent / "app.py"
    source = app_source.read_text()
    assert 'modal.Cron("0 6 * * 1")' in source
    assert "def run_regression_suite" in source

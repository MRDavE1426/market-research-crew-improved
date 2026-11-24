"""Pandas Analysis Test: the data parser tool must compute correct statistics."""
import math

import pandas as pd
import pytest

import config.settings as cfg
from crew.tools.financial_tool import (
    MetricsError,
    calculate_financial_metrics,
    compute_summary_df,
    data_quality_checks,
    dataframe_to_markdown,
    load_metrics,
    save_metrics_csv_text,
    summarize_csv_to_markdown,
)

MOCK_CSV = """company,revenue,profit
Alpha,10,2
Beta,20,4
Gamma,30,6
Delta,40,8
"""


@pytest.fixture()
def mock_csv(tmp_path):
    path = tmp_path / "mock.csv"
    path.write_text(MOCK_CSV)
    return path


def test_summary_statistics_are_correct(mock_csv):
    summary = compute_summary_df(load_metrics(mock_csv))
    rev = summary.loc["revenue"]
    assert rev["count"] == 4
    assert rev["mean"] == pytest.approx(25.0)
    assert rev["min"] == pytest.approx(10.0)
    assert rev["max"] == pytest.approx(40.0)
    # sample standard deviation (pandas default, ddof=1) of 10,20,30,40
    assert rev["std"] == pytest.approx(math.sqrt(500 / 3))
    assert summary.loc["profit", "mean"] == pytest.approx(5.0)


def test_text_column_is_excluded_from_statistics(mock_csv):
    summary = compute_summary_df(load_metrics(mock_csv))
    assert list(summary.index) == ["revenue", "profit"]


def test_markdown_table_structure(mock_csv):
    md = summarize_csv_to_markdown(mock_csv)
    assert "| metric | count | mean | std | min | max |" in md
    assert "| revenue | 4 | 25.00 | 12.91 | 10.00 | 40.00 |" in md
    assert "| --- |" in md
    assert "Data Quality Checks" in md


def test_tool_wrapper_returns_markdown(mock_csv):
    out = calculate_financial_metrics.run(csv_path=str(mock_csv))
    assert "| revenue |" in out and "25.00" in out


def test_tool_wrapper_reports_missing_file_instead_of_crashing(tmp_path):
    out = calculate_financial_metrics.run(csv_path=str(tmp_path / "nope.csv"))
    assert out.startswith("ERROR:")


def test_tool_uses_current_run_csv_if_manager_passes_bad_relative_path(tmp_path):
    output_dir = tmp_path / "run-output"
    output_dir.mkdir()
    (output_dir / "metrics.csv").write_text(MOCK_CSV, encoding="utf-8")
    cfg.set_output_dir(output_dir)
    try:
        out = calculate_financial_metrics.run(csv_path="research_output/metrics.csv")
    finally:
        cfg.set_output_dir(None)
    assert "| revenue | 4 | 25.00 |" in out


def test_currency_and_percent_strings_are_coerced(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text('company,revenue,margin\nA,"$1,200",10%\nB,"$800",20%\n')
    df = load_metrics(path)
    assert df["revenue"].tolist() == [1200, 800]
    assert compute_summary_df(df).loc["margin", "mean"] == pytest.approx(15.0)


def test_data_quality_flags_missing_duplicates_and_negatives(tmp_path):
    path = tmp_path / "dq.csv"
    path.write_text("company,x\nA,1\nA,1\nB,\nC,-5\n")
    findings = " ".join(data_quality_checks(load_metrics(path)))
    assert "Missing values" in findings
    assert "duplicate" in findings.lower()
    assert "negative" in findings


def test_single_row_std_is_dash_not_crash(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("company,x\nA,5\n")
    md = dataframe_to_markdown(compute_summary_df(load_metrics(path)))
    assert "| x | 1 | 5.00 | - | 5.00 | 5.00 |" in md


def test_empty_and_non_numeric_files_raise(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")
    with pytest.raises(MetricsError):
        load_metrics(empty)
    text_only = tmp_path / "text.csv"
    text_only.write_text("a,b\nfoo,bar\n")
    with pytest.raises(MetricsError):
        load_metrics(text_only)


def test_empty_research_csv_is_rejected_without_overwriting_saved_metrics(tmp_path):
    destination = tmp_path / "metrics.csv"
    destination.write_text("company,revenue\nAlpha,5\n", encoding="utf-8")
    original = destination.read_text(encoding="utf-8")
    with pytest.raises(MetricsError, match="no supported numeric"):
        save_metrics_csv_text("company,revenue\nAlpha,\nBeta,\n", destination)
    assert destination.read_text(encoding="utf-8") == original


def test_save_metrics_csv_text_strips_code_fences(tmp_path):
    dest = tmp_path / "out" / "metrics.csv"
    save_metrics_csv_text("```csv\ncompany,revenue\nA,1\nB,2\n```", dest)
    assert pd.read_csv(dest)["revenue"].tolist() == [1, 2]

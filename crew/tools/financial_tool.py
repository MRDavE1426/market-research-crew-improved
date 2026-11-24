"""Financial Data Parser: Pandas statistics -> clean Markdown table.

The pure functions here (``load_metrics``, ``compute_summary_df`` ...) are
what the tests and the PDF generator use. ``calculate_financial_metrics`` is
the thin ``@tool`` wrapper handed to the Data Analyst agent, and
``save_metrics_to_csv`` is the helper the Researcher uses to hand data over.
"""
from __future__ import annotations

import io
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings

from ._compat import tool

SUMMARY_STATS = ["count", "mean", "std", "min", "max"]
_NUMERIC_JUNK = re.compile(r"[,$%\s]|usd|bn|b$", flags=re.IGNORECASE)


class MetricsError(ValueError):
    """Raised when a metrics file is missing, empty or has no numeric data."""


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------
def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns like '$1,200' / '12.5%' into numbers when *all* their
    non-empty values can be parsed. Columns that cannot (e.g. company names)
    are left untouched."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        cleaned = series.map(lambda v: _NUMERIC_JUNK.sub("", v.strip()))
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().all():
            df[col] = pd.to_numeric(
                df[col].astype(str).map(lambda v: _NUMERIC_JUNK.sub("", v.strip())),
                errors="coerce",
            )
    return df


def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    path = Path(str(csv_path).strip().strip("'\""))
    if not path.exists():
        raise MetricsError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises several different error types
        raise MetricsError(f"Could not parse CSV '{path}': {exc}") from exc
    if df.empty:
        raise MetricsError(f"CSV file is empty: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    df = _coerce_numeric_columns(df)
    if df.select_dtypes("number").empty:
        raise MetricsError("CSV has no numeric columns to summarise.")
    return df


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def compute_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per numeric metric; columns = count, mean, std, min, max."""
    numeric = df.select_dtypes("number")
    if numeric.empty:
        raise MetricsError("No numeric columns to summarise.")
    summary = numeric.agg(SUMMARY_STATS).T
    summary["count"] = summary["count"].astype(int)
    summary.index.name = "metric"
    return summary


def data_quality_checks(df: pd.DataFrame) -> list[str]:
    """Human-readable numeric-consistency findings."""
    findings: list[str] = []
    findings.append(f"Rows: {len(df)}, columns: {len(df.columns)}.")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        findings.append("No missing values.")
    else:
        parts = ", ".join(f"{c} ({n})" for c, n in missing.items())
        findings.append(f"Missing values found in: {parts}.")
    dupes = int(df.duplicated().sum())
    findings.append(
        "No duplicate rows." if dupes == 0 else f"{dupes} duplicate row(s) found."
    )
    non_numeric = [c for c in df.columns if c not in df.select_dtypes("number").columns]
    if non_numeric:
        findings.append("Non-numeric (label) columns: " + ", ".join(non_numeric) + ".")
    for col in df.select_dtypes("number").columns:
        neg = int((df[col] < 0).sum())
        if neg:
            findings.append(f"Column '{col}' has {neg} negative value(s) - verify they are valid.")
    return findings


# --------------------------------------------------------------------------
# Markdown rendering (no tabulate dependency)
# --------------------------------------------------------------------------
def _fmt(value, decimals: int = 2) -> str:
    """Integers keep no decimals, floats get a fixed number, NaN becomes '-'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{decimals}f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True, decimals: int = 2) -> str:
    frame = df.reset_index() if index else df
    headers = [str(c) for c in frame.columns]
    rows = [[_fmt(v, decimals) for v in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def summarize_csv_to_markdown(csv_path: str | Path) -> str:
    """Full analyst output: stats table + data-quality bullets (all Markdown)."""
    df = load_metrics(csv_path)
    summary = compute_summary_df(df)
    checks = "\n".join(f"- {c}" for c in data_quality_checks(df))
    return (
        "### Summary Statistics\n\n"
        f"{dataframe_to_markdown(summary)}\n\n"
        "### Data Quality Checks\n\n"
        f"{checks}"
    )


# --------------------------------------------------------------------------
# CrewAI tools
# --------------------------------------------------------------------------
@tool("Calculate Financial Metrics")
def calculate_financial_metrics(csv_path: str) -> str:
    """Loads corporate metrics from a CSV file path with Pandas and returns
    summary statistics (count, mean, std, min, max for every numeric column)
    plus data-quality checks, formatted as Markdown. If a manager supplies an
    invalid relative path, use this run's configured metrics CSV when present."""
    try:
        requested = Path(str(csv_path).strip().strip("'\""))
        if not requested.is_absolute() and not requested.is_file():
            current_run_csv = get_settings().metrics_csv_path
            if current_run_csv.is_file():
                requested = current_run_csv
        return summarize_csv_to_markdown(requested)
    except MetricsError as exc:
        return f"ERROR: {exc}"


def save_metrics_csv_text(csv_content: str, destination: Path) -> str:
    """Validate CSV text and write it to ``destination``."""
    text = csv_content.strip()
    if text.startswith("```"):  # LLMs love fenced code blocks
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise MetricsError(f"CSV text could not be parsed: {exc}") from exc
    if df.empty or len(df.columns) < 2:
        raise MetricsError("CSV needs a header and at least one data row with 2+ columns.")
    numeric_values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if not numeric_values.notna().to_numpy().any():
        raise MetricsError("CSV contains no supported numeric financial values; existing metrics were preserved.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return str(destination)


@tool("Save Metrics CSV")
def save_metrics_to_csv(csv_content: str) -> str:
    """Saves collected company metrics (CSV text with a header row) to disk and
    returns the file path that the Financial Data Analyst must analyse."""
    destination = get_settings().metrics_csv_path
    if os.getenv("MARKET_RESEARCH_RUN_ID") and destination.is_file() and destination.stat().st_size:
        return f"Metrics already saved for this live research run; preserved at: {destination}"
    try:
        path = save_metrics_csv_text(csv_content, destination)
    except MetricsError as exc:
        return f"ERROR: {exc}"
    return f"Metrics saved to: {path}"

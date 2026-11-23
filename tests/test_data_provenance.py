"""Regression tests: a live report must never present stale/synthetic data as researched data."""
import pytest
from pypdf import PdfReader

import config.settings as cfg
import crew.crew as crew_mod
from crew.pipeline import run_pipeline
from crew.tools import search_tool


@pytest.fixture(autouse=True)
def _live_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    yield
    cfg.set_output_dir(None)


def _pdf_text(path) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def test_stale_metrics_csv_from_previous_run_is_not_reused(tmp_path, monkeypatch):
    run_pipeline("earlier offline run", offline=True, output_dir=tmp_path)     # leaves metrics.csv behind
    assert (tmp_path / "metrics.csv").exists()

    # Live run in which the Researcher never saves any data.
    monkeypatch.setattr(crew_mod.MarketResearchCrew, "kickoff",
                        lambda self: "## Executive Summary\nLive report.\n")
    result = run_pipeline("live run", offline=False, output_dir=tmp_path)

    text = _pdf_text(result["pdf"])
    assert result["data_fallback"] is True
    assert "Data source: FALLBACK" in text.replace("\n", " ")
    assert "DATA WARNING" in text


def test_fallback_flag_is_false_when_researcher_data_is_used(tmp_path, monkeypatch):
    def fake_kickoff(self):
        (tmp_path / "metrics.csv").write_text("company,revenue_usd_bn\nAcme,10\nBeta,20\n")
        return "## Executive Summary\nLive report.\n"

    monkeypatch.setattr(crew_mod.MarketResearchCrew, "kickoff", fake_kickoff)
    result = run_pipeline("live run", offline=False, output_dir=tmp_path)
    assert result["data_fallback"] is False
    text = _pdf_text(result["pdf"])
    assert "DATA WARNING" not in text and "Acme" in text


def test_scraped_text_is_flagged_as_untrusted(monkeypatch):
    monkeypatch.setattr(search_tool, "search_web",
                        lambda q, max_results=3: [{"title": "T", "url": "", "snippet": "s"}])
    out = search_tool.search_and_scrape("anything")
    assert "UNTRUSTED DATA" in out

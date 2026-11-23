"""End-to-end (keyless): pipeline writes market_report.pdf, .md and benchmarks.json."""
import json

import pytest
from pypdf import PdfReader

import config.settings as cfg
from crew.crew import check_live_requirements
from crew.pipeline import run_pipeline
from main import main


@pytest.fixture(autouse=True)
def _reset_output_dir():
    yield
    cfg.set_output_dir(None)


def test_offline_pipeline_produces_all_artifacts(tmp_path):
    result = run_pipeline("Tech Sector Q3 Analysis", offline=True, output_dir=tmp_path)
    pdf = tmp_path / "market_report.pdf"
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "market_report.md").read_text().startswith("## Executive Summary")

    reader = PdfReader(str(pdf))
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert "Tech Sector Q3 Analysis" in text and "revenue_usd_bn" in text
    assert f"Page 1 of {len(reader.pages)}" in text

    run = json.loads((tmp_path / "benchmarks.json").read_text())["runs"][-1]
    assert run["run_id"] == result["run_id"] and run["status"] == "completed"
    assert [t["task"] for t in run["tasks"]] == [
        "research_task", "analysis_task", "writing_task", "pdf_compilation"]
    delegations = [e for e in run["routing_history"] if e["event"] == "delegation"]
    assert [d["to"] for d in delegations] == [
        "Market Research Specialist", "Financial Data Analyst", "Technical Reports Writer"]


def test_cli_offline_exit_code_and_default_output(tmp_path, capsys):
    assert main(["--offline", "--quiet", "--query", "AI Chips", "-o", str(tmp_path)]) == 0
    assert "market_report.pdf" in capsys.readouterr().out
    assert (tmp_path / "market_report.pdf").exists()


def test_live_mode_without_keys_fails_fast_with_clear_message(tmp_path, monkeypatch, capsys):
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert main(["--quiet", "-o", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "GROQ_API_KEY" in err and "TAVILY_API_KEY" in err
    run = json.loads((tmp_path / "benchmarks.json").read_text())["runs"][-1]
    assert run["status"] == "failed"


def test_groq_is_the_default_live_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = cfg.get_settings()
    assert settings.manager_llm.startswith("groq/")
    assert settings.worker_llm.startswith("groq/")
    check_live_requirements(settings)

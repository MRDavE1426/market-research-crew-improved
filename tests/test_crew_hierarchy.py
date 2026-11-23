"""Runs the REAL CrewAI hierarchical runtime with a scripted LLM (no API keys / network).

Verifies the brief's Test Case 1 & 2 wiring: manager LLM initialises the workflow,
tasks are delegated dynamically (Researcher -> Analyst -> Writer), the trace is
logged, benchmarks.json is written and market_report.pdf is produced.
"""
import json

import pandas as pd
import pytest
from crewai import Process
from pypdf import PdfReader

import config.settings as cfg
import crew.agents as agents_mod
import crew.crew as crew_mod
from crew.benchmarks import BenchmarkLogger
from crew.pipeline import run_pipeline
from tests.scripted_llm import ScriptedLLM

ROLES = ["Market Research Specialist", "Financial Data Analyst", "Technical Reports Writer"]


@pytest.fixture()
def scripted(tmp_path, monkeypatch):
    cfg.set_output_dir(tmp_path)
    for key, val in (("GROQ_API_KEY", "gsk-test"), ("TAVILY_API_KEY", "tvly-test")):
        monkeypatch.setenv(key, val)
    llm = ScriptedLLM(csv_path=str(tmp_path / "metrics.csv"))
    monkeypatch.setattr(agents_mod, "make_llm", lambda *a, **k: llm)
    monkeypatch.setattr(crew_mod, "make_llm", lambda *a, **k: llm)
    yield llm
    cfg.set_output_dir(None)


def test_crew_is_hierarchical_with_three_specialists(scripted, tmp_path):
    settings = cfg.get_settings()
    bench = BenchmarkLogger(settings.benchmarks_path, "Tech Sector Q3 Analysis")
    crew = crew_mod.MarketResearchCrew(settings, "Tech Sector Q3 Analysis", bench).build()
    assert crew.process == Process.hierarchical
    assert crew.manager_llm is not None
    assert [a.role for a in crew.agents] == ROLES
    assert all(t.agent is None for t in crew.tasks)          # manager assigns dynamically
    assert [t.name for t in crew.tasks] == ["research_task", "analysis_task", "writing_task"]


def test_manager_delegates_and_trace_is_logged(scripted, tmp_path):
    settings = cfg.get_settings()
    bench = BenchmarkLogger(settings.benchmarks_path, "Tech Sector Q3 Analysis", manager_llm="scripted")
    report = crew_mod.MarketResearchCrew(settings, "Tech Sector Q3 Analysis", bench).kickoff()
    run = bench.finish()

    assert report.startswith("## Executive Summary")
    assert (tmp_path / "metrics.csv").exists()                # researcher's tool wrote the data
    history = [(e["event"], e["from"], e["to"]) for e in run["routing_history"]]
    assert history[0][0] == "manager_init"
    delegations = [h for h in history if h[0] == "delegation"]
    assert [d[2] for d in delegations] == ROLES               # manager -> each specialist, in order
    assert all(d[1] == "Crew Manager" for d in delegations)
    tools = [(h[1], h[2]) for h in history if h[0] == "tool_call"]
    assert ("Market Research Specialist", "Save Metrics CSV") in tools
    assert ("Financial Data Analyst", "Calculate Financial Metrics") in tools
    assert [(t["task"], t["agent"]) for t in run["tasks"]] == [
        ("research_task", ROLES[0]), ("analysis_task", ROLES[1]), ("writing_task", ROLES[2])]
    assert set(run["agent_activity"]) == set(ROLES)


def test_full_live_pipeline_writes_pdf_and_benchmarks(scripted, tmp_path):
    result = run_pipeline("Tech Sector Q3 Analysis", offline=False, output_dir=tmp_path)
    pdf = tmp_path / "market_report.pdf"
    assert pdf.exists() and result["pdf"] == str(pdf)
    text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
    assert "Scripted report for wiring tests" in text
    assert "Acme Cloud" in text                                # researcher CSV rendered in appendix
    assert "Mode: live" in text

    run = json.loads((tmp_path / "benchmarks.json").read_text())["runs"][-1]
    assert run["mode"] == "live" and run["status"] == "completed"
    assert [t["task"] for t in run["tasks"]] == [
        "research_task", "analysis_task", "writing_task", "pdf_compilation"]
    assert all(t["duration_seconds"] >= 0 for t in run["tasks"])
    assert pd.read_csv(tmp_path / "metrics.csv").shape == (3, 3)


def test_unusable_researcher_csv_falls_back_to_sample_data(scripted, tmp_path, monkeypatch):
    # researcher "saves" a CSV with no numeric columns -> the pipeline must not crash
    monkeypatch.setattr("tests.scripted_llm.CSV", "company,note\nA,x\nB,y\n")
    result = run_pipeline("Q", offline=False, output_dir=tmp_path)
    text = "\n".join(p.extract_text() for p in PdfReader(result["pdf"]).pages)
    assert "FALLBACK" in text

"""Benchmark logging: durations per task + routing history persisted to benchmarks.json."""
import json
import time

import pytest

from crew.benchmarks import BenchmarkLogger


def test_logs_task_durations_and_routing_history(tmp_path):
    path = tmp_path / "benchmarks.json"
    bench = BenchmarkLogger(path, query="Tech Sector Q3 Analysis", mode="offline", manager_llm="gpt-4o")
    with bench.task("research_task", "Market Research Specialist"):
        time.sleep(0.02)
    bench.record_event("delegation", "Crew Manager", "Financial Data Analyst", "analyse csv")
    with bench.task("analysis_task", "Financial Data Analyst"):
        pass
    bench.finish(artifacts={"pdf": "market_report.pdf"})

    data = json.loads(path.read_text())
    run = data["runs"][0]
    assert run["query"] == "Tech Sector Q3 Analysis" and run["process"] == "hierarchical"
    assert [t["task"] for t in run["tasks"]] == ["research_task", "analysis_task"]
    assert run["tasks"][0]["duration_seconds"] >= 0.02
    assert run["tasks"][0]["agent"] == "Market Research Specialist"
    assert run["total_duration_seconds"] >= run["tasks"][0]["duration_seconds"]
    events = [e["event"] for e in run["routing_history"]]
    assert "delegation" in events and events.count("task_completed") == 2
    assert run["artifacts"]["pdf"] == "market_report.pdf"


def test_runs_are_appended_not_overwritten(tmp_path):
    path = tmp_path / "b.json"
    for q in ("one", "two"):
        BenchmarkLogger(path, q).finish()
    runs = json.loads(path.read_text())["runs"]
    assert [r["query"] for r in runs] == ["one", "two"]


def test_failed_task_is_recorded_with_status(tmp_path):
    bench = BenchmarkLogger(tmp_path / "b.json", "q")
    with pytest.raises(RuntimeError):
        with bench.task("writing_task", "Technical Reports Writer"):
            raise RuntimeError("boom")
    run = bench.finish(status="failed", error="boom")
    assert run["tasks"][0]["status"] == "failed" and run["error"] == "boom"


def test_corrupt_existing_file_is_recovered(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{not json")
    BenchmarkLogger(path, "q").finish()
    assert len(json.loads(path.read_text())["runs"]) == 1


def test_agent_run_activity_is_tracked(tmp_path):
    bench = BenchmarkLogger(tmp_path / "b.json", "q")
    bench.note_worker_started("Financial Data Analyst")
    bench.record_agent_run("Financial Data Analyst", 1.25)
    bench.record_agent_run("Financial Data Analyst", 0.75)
    assert bench.latest_active_agent() == "Financial Data Analyst"
    run = bench.finish()
    assert run["agent_activity"]["Financial Data Analyst"] == {"runs": 2, "active_seconds": 2.0}

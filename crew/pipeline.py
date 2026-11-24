"""End-to-end orchestration: crew -> Markdown -> Pandas stats -> PDF -> benchmarks."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from config.settings import SAMPLE_METRICS_CSV, get_settings, set_output_dir

from .benchmarks import BenchmarkLogger
from .report_generator import generate_pdf_report
from .tools.financial_tool import MetricsError, compute_summary_df, load_metrics

log = logging.getLogger("crew.pipeline")


def run_pipeline(query: str, offline: bool = False, output_dir: str | Path | None = None) -> dict:
    """Run the whole system and return a dict with the artifact paths."""
    if output_dir:
        set_output_dir(output_dir)
    settings = get_settings()
    mode = "offline" if offline else "live"
    bench = BenchmarkLogger(settings.benchmarks_path, query, mode=mode, manager_llm=settings.manager_llm)
    log.info("Run %s | query=%r | mode=%s | output_dir=%s", bench.run_id, query, mode, settings.output_dir)

    try:
        if offline:
            from .offline import run_offline
            markdown = run_offline(settings, query, bench)
        else:
            from .crew import MarketResearchCrew, check_live_requirements
            from .tools.search_tool import clear_research_run
            check_live_requirements(settings)
            # A leftover metrics.csv from an earlier (or offline) run must never be
            # mistaken for data the Researcher collected during THIS run.
            settings.metrics_csv_path.unlink(missing_ok=True)
            previous_run_id = os.environ.get("MARKET_RESEARCH_RUN_ID")
            previous_topic = os.environ.get("MARKET_RESEARCH_TOPIC")
            os.environ["MARKET_RESEARCH_RUN_ID"] = bench.run_id
            os.environ["MARKET_RESEARCH_TOPIC"] = query
            try:
                markdown = MarketResearchCrew(settings, query, bench).kickoff()
            finally:
                clear_research_run(bench.run_id)
                if previous_run_id is None:
                    os.environ.pop("MARKET_RESEARCH_RUN_ID", None)
                else:
                    os.environ["MARKET_RESEARCH_RUN_ID"] = previous_run_id
                if previous_topic is None:
                    os.environ.pop("MARKET_RESEARCH_TOPIC", None)
                else:
                    os.environ["MARKET_RESEARCH_TOPIC"] = previous_topic

        settings.markdown_path.write_text(markdown, encoding="utf-8")

        # Post-processing: statistics table + PDF
        data_source = "metrics collected by the Researcher agent"
        data_fallback = False
        csv_path = settings.metrics_csv_path
        try:
            raw = load_metrics(csv_path)
        except MetricsError as exc:
            log.warning("Researcher metrics unusable (%s); falling back to bundled sample dataset", exc)
            raw = load_metrics(SAMPLE_METRICS_CSV)
            data_source = "FALLBACK: bundled sample dataset (researcher output unusable)"
            data_fallback = True
        if offline:
            data_source = "bundled illustrative sample dataset (offline demo)"

        with bench.task("pdf_compilation", "Report Compiler (post-processing)"):
            generate_pdf_report(
                markdown,
                compute_summary_df(raw),
                settings.pdf_path,
                metadata={
                    "title": "Market Research Report",
                    "query": query,
                    "run_id": bench.run_id,
                    "mode": mode,
                    "data_source": data_source,
                },
                raw_df=raw,
            )
        artifacts = {
            "pdf": str(settings.pdf_path),
            "markdown": str(settings.markdown_path),
            "metrics_csv": str(csv_path),
            "benchmarks": str(settings.benchmarks_path),
        }
        run = bench.finish(artifacts=artifacts)
        log.info("Done in %.2fs -> %s", run["total_duration_seconds"], settings.pdf_path)
        return {**artifacts, "run_id": bench.run_id, "total_seconds": run["total_duration_seconds"],
                "data_fallback": data_fallback}
    except Exception as exc:
        bench.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise

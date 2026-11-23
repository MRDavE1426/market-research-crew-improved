"""Keyless offline demo run.

Runs the *same* three stages (research -> analysis -> writing) and the *same*
Pandas tool and PDF compiler, but without any LLM or network access, using the
bundled illustrative dataset. It exists so the whole pipeline (benchmarks,
Markdown, PDF) can be demonstrated and tested without API keys. Output is
clearly labelled "offline" and contains NO real market data.
"""
from __future__ import annotations

import logging
import shutil

from config.settings import SAMPLE_METRICS_CSV, Settings

from .benchmarks import BenchmarkLogger
from .tools.financial_tool import (
    compute_summary_df,
    dataframe_to_markdown,
    data_quality_checks,
    load_metrics,
    summarize_csv_to_markdown,
)

log = logging.getLogger("crew.offline")
MANAGER = "Crew Manager"
RESEARCHER = "Market Research Specialist"
ANALYST = "Financial Data Analyst"
WRITER = "Technical Reports Writer"


def run_offline(settings: Settings, query: str, bench: BenchmarkLogger) -> str:
    log.info("[MANAGER] Initializing hierarchical workflow | manager_llm=%s | process=hierarchical "
             "(OFFLINE SIMULATION - no LLM calls)", settings.manager_llm)
    bench.record_event("manager_init", MANAGER, None, "offline simulation of hierarchical delegation")
    bench.mark_start()

    # 1) Researcher: "collects" metrics -> metrics.csv
    bench.record_event("delegation", MANAGER, RESEARCHER, "research_task: collect company metrics")
    with bench.task("research_task", RESEARCHER):
        bench.record_event("tool_call", RESEARCHER, "Web Search and Scrape", "skipped in offline mode")
        shutil.copyfile(SAMPLE_METRICS_CSV, settings.metrics_csv_path)
        bench.record_event("tool_call", RESEARCHER, "Save Metrics CSV", str(settings.metrics_csv_path))
        log.info("[RESEARCHER] Loaded bundled sample dataset -> %s", settings.metrics_csv_path)

    # 2) Analyst: Pandas statistics
    bench.record_event("delegation", MANAGER, ANALYST, "analysis_task: compute statistics")
    with bench.task("analysis_task", ANALYST):
        bench.record_event("tool_call", ANALYST, "Calculate Financial Metrics", str(settings.metrics_csv_path))
        summarize_csv_to_markdown(settings.metrics_csv_path)  # same tool call the analyst agent makes
        log.info("[ANALYST] Computed summary statistics with Pandas")

    # 3) Writer: templated Markdown report
    bench.record_event("delegation", MANAGER, WRITER, "writing_task: draft the report")
    with bench.task("writing_task", WRITER):
        markdown = _write_report(settings, query)
        log.info("[WRITER] Drafted %d characters of Markdown", len(markdown))
    return markdown


def _write_report(settings: Settings, query: str) -> str:
    df = load_metrics(settings.metrics_csv_path)
    summary = compute_summary_df(df)
    top_growth = df.sort_values("yoy_growth_pct", ascending=False).iloc[0]
    top_revenue = df.sort_values("revenue_usd_bn", ascending=False).iloc[0]
    table = dataframe_to_markdown(summary)
    checks = "\n".join(f"- {c}" for c in data_quality_checks(df))
    return f"""## Executive Summary
This is an **offline demonstration report** for the topic *{query}*. It was produced without
any LLM or web access from a bundled illustrative dataset of {len(df)} fictional companies,
so it contains **no real market data**. Run the crew without `--offline` (and with API keys
configured) to generate a real report.

Across the sample, mean revenue is {summary.loc['revenue_usd_bn', 'mean']:.2f} bn USD and the mean
operating margin is {summary.loc['operating_margin_pct', 'mean']:.2f}%.

## Market Overview
The illustrative sector contains cloud, semiconductor, software, hardware, analytics and security
vendors. Scale differs widely: revenue ranges from {summary.loc['revenue_usd_bn', 'min']:.2f} to
{summary.loc['revenue_usd_bn', 'max']:.2f} bn USD, which is reflected in a standard deviation of
{summary.loc['revenue_usd_bn', 'std']:.2f}.

## Key Financial Metrics
The table below is produced by the Pandas-based Financial Data Parser tool.

{table}

## Analysis & Insights
- **Largest company by revenue:** {top_revenue['company']} ({top_revenue['revenue_usd_bn']:.1f} bn USD).
- **Fastest growth:** {top_growth['company']} ({top_growth['yoy_growth_pct']:.1f}% year over year).
- **Profitability spread:** operating margin spans {summary.loc['operating_margin_pct', 'min']:.1f}% to
  {summary.loc['operating_margin_pct', 'max']:.1f}%, so margin quality differs far more than headline growth.
- **R&D intensity:** mean R&D spend is {summary.loc['rd_spend_usd_bn', 'mean']:.2f} bn USD per company.

## Risks & Limitations
- The dataset is illustrative and synthetic; conclusions must not be used for investment decisions.
- Sample size is small ({len(df)} rows), so standard deviations are unstable.
- Numeric consistency checks run on the data:

{checks}

## Conclusion
The pipeline (research, analysis, writing, PDF compilation, benchmarking) works end to end. Replace the
offline dataset with live research output by running with valid API keys.

## Sources
- Bundled file `data/sample_metrics.csv` (synthetic, for demonstration only).
"""

# Multi-Agent Market Research & Report Generator (CrewAI)

- **Repository**: https://github.com/MRDavE1426/market-research-crew-improved

A hierarchical CrewAI system: a **manager LLM** dynamically delegates work to three specialist agents,
custom `@tool`s scrape the web and crunch numbers with **Pandas**, a **ReportLab** post-processor
typesets an A4 PDF, and every run is benchmarked into `benchmarks.json`.

```
python main.py --query 'Tech Sector Q3 Analysis'
                 |
        Crew (Process.hierarchical, manager_llm)
                 |  delegates dynamically
   +-------------+------------------+
   v                                v                         v
Market Research Specialist   Financial Data Analyst   Technical Reports Writer
 - Web Search and Scrape      - Calculate Financial     (no tools; synthesises
 - Save Metrics CSV             Metrics (Pandas)          research + statistics)
        \_____ metrics.csv ____/             \____ Markdown report ____/
                                   |
                  report_generator.py (ReportLab) -> market_report.pdf
                  benchmarks.py                    -> benchmarks.json
```

## Installation

1. Create a virtual environment (Python 3.10 - 3.13) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   In Windows PowerShell, activation may be blocked by the execution policy. You can use the venv interpreter directly instead:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe main.py --query "Tech Sector Q3 Analysis"
   .\.venv\Scripts\python.exe -m pytest -v
   ```
2. Configure `.env` (copy the template, then fill in your keys):
   ```bash
   cp .env.example .env               # Windows: copy .env.example .env
   ```
   | Variable | Needed for |
   |---|---|
   | `GROQ_API_KEY` | manager LLM + worker agents (defaults: `groq/openai/gpt-oss-120b` manager and `groq/qwen/qwen3.8-27b` workers) |
   | `OPENAI_API_KEY` | only required when selecting an OpenAI model |
   | `TAVILY_API_KEY` **or** `SERPER_API_KEY` | web search (Tavily is used first if both are set) |
   | `ANTHROPIC_API_KEY` | only if you select an Anthropic model |
   | `MANAGER_LLM`, `WORKER_LLM`, `REPORT_OUTPUT_DIR` | optional overrides |
3. Set `GROQ_API_KEY` in `.env`, then run the main app:
   ```bash
   python main.py --query 'Tech Sector Q3 Analysis'
   ```
4. Run tests:
   ```bash
   python -m pytest -v
   ```

### Keyless demo mode
`python main.py --offline` runs the same three stages, the same Pandas tool and the same PDF compiler
**without any LLM or network call**, using the bundled *synthetic* dataset in `data/sample_metrics.csv`.
It is only for demos/CI; the PDF header says `Mode: offline` and the report states it contains no real data.

## Outputs (written to `output/`, or `--output-dir`)

| File | Content |
|---|---|
| `market_report.pdf` | A4 report: header banner with metadata, styled sections, Pandas statistics table, appendix with dataset + data-quality checks, `Page X of Y` footer |
| `market_report.md` | The Content Writer's Markdown draft |
| `metrics.csv` | The dataset the Researcher collected |
| `benchmarks.json` | Per-task durations, per-agent active time, and the full routing trace (manager init -> delegations -> tool calls -> completions) |

Ready-made examples are in [`sample_outputs/`](sample_outputs/).

## Project layout

```
main.py                    CLI entry point
config/                    settings.py (env/.env), agents.yaml, tasks.yaml
crew/
  agents.py  tasks.py      3 agents / 3 tasks built from the YAML configs
  crew.py                  Crew(process=Process.hierarchical, manager_llm=...)
  tracing.py               CrewAI event-bus hooks -> routing history + timings
  benchmarks.py            BenchmarkLogger -> benchmarks.json
  report_generator.py      ReportLab PDF compiler
  pipeline.py  offline.py  orchestration + keyless demo mode
  tools/
    search_tool.py         Web Scraper tool (Tavily / SerperDev + page scraping)
    financial_tool.py      Pandas Financial Data Parser tool (Markdown tables)
tests/                     pytest suite (see below)
data/sample_metrics.csv    synthetic demo dataset
```

## How the brief's requirements map to the code

| Requirement | Where |
|---|---|
| 3 agents with the exact roles | `config/agents.yaml`, `crew/agents.py` |
| `process=Process.hierarchical` + `manager_llm`, dynamic delegation | `crew/crew.py` (tasks are intentionally **not** bound to an agent, so the manager decides) |
| Web scraper `@tool` (SerperDev/Tavily) | `crew/tools/search_tool.py` |
| Pandas parser `@tool` -> Markdown table (mean/min/max/std) | `crew/tools/financial_tool.py` |
| A4 PDF: banner, styled sections, stats grid, `Page X of Y`, footer | `crew/report_generator.py` |
| `benchmarks.json`: per-task time + routing history | `crew/benchmarks.py`, `crew/tracing.py` |
| pytest: Pandas math test + PDF layout test | `tests/test_financial_tool.py`, `tests/test_pdf_report.py` |
| Secrets in `.env`, `.env.example` provided | `.env.example`, `.gitignore` |

## Verification test cases

1. **Hierarchical execution** - `python main.py --query 'Tech Sector Q3 Analysis'` logs
   `[MANAGER] Initializing hierarchical workflow | manager_llm=...`, then `[ROUTING] delegation | Crew Manager -> <specialist>`
   lines followed by each specialist's tool calls (Researcher fetching, Analyst parsing, Writer generating).
2. **Aggregation & PDF** - `output/market_report.pdf` is written with the banner, the Pandas tables and page numbers.
3. **Test suite** - `python -m pytest -v` (46 tests; no network or API keys needed).

### What the tests cover
- `test_financial_tool.py` - statistics on a mock CSV (mean/min/max/std), currency/percent coercion, data-quality flags, edge cases.
- `test_pdf_report.py` - valid A4 PDF, banner text, tables, `Page X of Y` on every page, long multi-page flow, hostile Markdown never raises.
- `test_search_tool.py` - Tavily and Serper flows with mocked HTTP, graceful degradation.
- `test_benchmarks.py` - JSON schema, appended runs, failed tasks, corrupt-file recovery.
- `test_crew_hierarchy.py` - drives the **real** CrewAI hierarchical runtime with a scripted LLM: manager delegation order, trace, PDF, fallback.
- `test_pipeline_offline.py` - CLI and end-to-end offline run, fail-fast on missing keys.
- `test_data_provenance.py` - a stale `metrics.csv` is never reused as "researched" data; fallback is flagged; scraped text is marked untrusted.

## Notes & design decisions
- **Data hand-off**: the Researcher saves what it found via the `Save Metrics CSV` tool; the Analyst runs the Pandas tool on that file; the post-processor re-reads the same CSV for the PDF tables. If the Researcher's CSV is unusable, the PDF falls back to the sample dataset and says so in the header (`Data source: FALLBACK ...`).
- **Timings**: task durations are measured between task completions (the chain is sequential); per-agent active time comes from CrewAI agent start/finish events.
- **CrewAI tracing prompt** is disabled by default (`CREWAI_TRACING_ENABLED=false`) because its interactive prompt blocks for seconds and would distort benchmarks.
- Standard PDF fonts only cover Latin-1; other characters are mapped or replaced so the PDF never fails on LLM output.

## Data provenance & safety
- At the start of every **live** run the old `metrics.csv` is deleted, so a report can only contain data collected in that run.
- If the Researcher produces no usable CSV, the PDF still renders (so the pipeline never crashes) but uses the synthetic sample
  dataset **and says so**: `Data source: FALLBACK ...` in the banner, a red *DATA WARNING* box on page 1, a console warning, and
  `data_fallback: true` in the pipeline result.
- Scraped web text is prefixed with an "untrusted data" notice so instructions hidden in web pages are not followed by the agents.
- CI: `.github/workflows/tests.yml` runs `pytest` on Python 3.11 and 3.12 for every push.

## Troubleshooting
- `Configuration error: Missing configuration ...` - fill the keys in `.env` (the app fails fast before spending tokens).
- Live quarterly research aligns each firm to the requested calendar period and labels its fiscal period. Source-backed metrics are extracted from current Tavily evidence; unsupported values remain blank. Check your search key/quota if live evidence is unavailable.
- `ModuleNotFoundError` for a provider - for Claude models install `pip install "crewai[anthropic]"`.

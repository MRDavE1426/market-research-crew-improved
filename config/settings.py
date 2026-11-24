"""Central configuration. All secrets come from the environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")

DEFAULT_QUERY = "Tech Sector Q3 Analysis"
SAMPLE_METRICS_CSV = DATA_DIR / "sample_metrics.csv"

_output_dir_override: Path | None = None


def set_output_dir(path: str | Path | None) -> None:
    """Override the output directory at runtime (used by the CLI and by tests)."""
    global _output_dir_override
    _output_dir_override = Path(path).resolve() if path else None


@dataclass(frozen=True)
class Settings:
    manager_llm: str
    worker_llm: str
    output_dir: Path
    openai_api_key: str | None
    groq_api_key: str | None
    anthropic_api_key: str | None
    tavily_api_key: str | None
    serper_api_key: str | None

    # ---- derived paths -------------------------------------------------
    @property
    def pdf_path(self) -> Path:
        return self.output_dir / "market_report.pdf"

    @property
    def markdown_path(self) -> Path:
        return self.output_dir / "market_report.md"

    @property
    def metrics_csv_path(self) -> Path:
        return self.output_dir / "metrics.csv"

    @property
    def benchmarks_path(self) -> Path:
        return self.output_dir / "benchmarks.json"

    @property
    def search_provider(self) -> str | None:
        if self.tavily_api_key:
            return "tavily"
        if self.serper_api_key:
            return "serper"
        return None


def get_settings() -> Settings:
    out = _output_dir_override or Path(
        os.getenv("REPORT_OUTPUT_DIR") or ROOT_DIR / "output"
    )
    out = Path(out)
    if not out.is_absolute():
        out = ROOT_DIR / out
    out.mkdir(parents=True, exist_ok=True)
    return Settings(
        manager_llm=os.getenv("MANAGER_LLM", "groq/openai/gpt-oss-120b"),
        worker_llm=os.getenv("WORKER_LLM", "groq/qwen/qwen3.8-27b"),
        output_dir=out,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        serper_api_key=os.getenv("SERPER_API_KEY") or None,
    )

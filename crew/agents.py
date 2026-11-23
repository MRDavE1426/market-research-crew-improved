"""The three specialist agents (definitions live in config/agents.yaml)."""
from __future__ import annotations

import logging

import yaml
from crewai import LLM, Agent

from config.settings import CONFIG_DIR, Settings

from .tools.financial_tool import calculate_financial_metrics, save_metrics_to_csv
from .tools.search_tool import web_search_and_scrape

log = logging.getLogger("crew.agents")


def load_yaml(filename: str) -> dict:
    with open(CONFIG_DIR / filename, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_llm(model: str, temperature: float = 0.2) -> LLM:
    return LLM(model=model, temperature=temperature)


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
def build_agents(settings: Settings, query: str) -> dict[str, Agent]:
    cfg = load_yaml("agents.yaml")
    llm = make_llm(settings.worker_llm)

    def make(key: str, tools: list) -> Agent:
        spec = cfg[key]
        role = spec["role"].strip()
        return Agent(
            role=role,
            goal=spec["goal"].format(query=query).strip(),
            backstory=spec["backstory"].strip(),
            tools=tools,
            llm=llm,
            allow_delegation=False,   # only the manager delegates
            verbose=True,
            max_iter=12,
        )

    return {
        "researcher": make("lead_researcher", [web_search_and_scrape, save_metrics_to_csv]),
        "analyst": make("data_analyst", [calculate_financial_metrics]),
        "writer": make("content_writer", []),
    }

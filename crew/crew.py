"""Hierarchical Crew: manager LLM delegates to Researcher -> Analyst -> Writer."""
from __future__ import annotations

import logging

from crewai import Agent, Crew, Process

from config.settings import Settings

from .agents import build_agents, make_llm
from .benchmarks import BenchmarkLogger
from .tasks import build_tasks
from .tracing import trace_crew_events

log = logging.getLogger("crew.crew")
MANAGER_LABEL = "Crew Manager"


class ConfigError(RuntimeError):
    """Raised before a live run when required API keys are missing."""


def check_live_requirements(settings: Settings) -> None:
    problems = []
    for label, model in (("manager", settings.manager_llm), ("worker", settings.worker_llm)):
        if model.lower().startswith("anthropic/"):
            if not settings.anthropic_api_key:
                problems.append(f"ANTHROPIC_API_KEY is required for the {label} model '{model}'")
        elif model.lower().startswith("groq/"):
            if not settings.groq_api_key:
                problems.append(f"GROQ_API_KEY is required for the {label} model '{model}'")
        elif not settings.openai_api_key:
            problems.append(f"OPENAI_API_KEY is required for the {label} model '{model}'")
    if not settings.search_provider:
        problems.append("TAVILY_API_KEY or SERPER_API_KEY is required for web search")
    if problems:
        raise ConfigError(
            "Missing configuration:\n  - " + "\n  - ".join(sorted(set(problems)))
            + "\nCopy .env.example to .env and fill it in, or run with --offline for a keyless demo."
        )


class MarketResearchCrew:
    def __init__(self, settings: Settings, query: str, bench: BenchmarkLogger):
        self.settings, self.query, self.bench = settings, query, bench

    def build(self) -> Crew:
        agents = build_agents(self.settings, self.query)
        tasks = build_tasks(self.query)
        roles = [a.role for a in agents.values()]

        log.info("[MANAGER] Initializing hierarchical workflow | manager_llm=%s | process=hierarchical",
                 self.settings.manager_llm)
        log.info("[MANAGER] Specialists available for delegation: %s", ", ".join(roles))
        log.info("[MANAGER] Tasks to delegate dynamically: %s", ", ".join(t.name for t in tasks))
        self.bench.record_event("manager_init", MANAGER_LABEL, None,
                                f"manager_llm={self.settings.manager_llm}; specialists={roles}")

        names = [t.name for t in tasks]

        def on_task_done(output) -> None:
            idx = len(self.bench.tasks)
            name = getattr(output, "name", None) or (names[idx] if idx < len(names) else f"task_{idx + 1}")
            agent = self.bench.latest_active_agent() or getattr(output, "agent", None) or MANAGER_LABEL
            self.bench.record_task(name, agent, summary=str(getattr(output, "raw", "")))

        manager_llm = make_llm(
            self.settings.manager_llm, temperature=0.1, max_tokens=2800
        )
        manager_agent = Agent(
            role=MANAGER_LABEL,
            goal=(
                "Route each pipeline task to its matching specialist once, accept "
                "completed tool-backed work, and continue. Keep delegation context "
                "concise: pass task instructions and a short summary of prior outputs, "
                "not full previous transcripts. Do not repeat research after the CSV "
                "has been saved."
            ),
            backstory=(
                "You coordinate the market researcher, financial analyst, and report "
                "writer. Delegate research_task to the Market Research Specialist, "
                "analysis_task to the Financial Data Analyst, and writing_task to the "
                "Technical Reports Writer."
            ),
            llm=manager_llm,
            allow_delegation=True,
            max_iter=2,
            verbose=True,
        )
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.hierarchical,                      # manager delegates dynamically
            manager_llm=manager_llm,
            manager_agent=manager_agent,
            task_callback=on_task_done,
            verbose=True,
        )

    def kickoff(self) -> str:
        crew = self.build()
        worker_roles = {a.role.strip() for a in crew.agents}
        self.bench.mark_start()
        with trace_crew_events(self.bench, worker_roles, MANAGER_LABEL):
            result = crew.kickoff()
        return str(getattr(result, "raw", result))

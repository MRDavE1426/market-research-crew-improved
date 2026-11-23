"""Task definitions (text lives in config/tasks.yaml).

Tasks are deliberately *not* bound to an agent. With ``Process.hierarchical``
the manager LLM decides, at run time, which specialist gets each task.
"""
from __future__ import annotations

from crewai import Task

from .agents import load_yaml


def build_tasks(query: str) -> list[Task]:
    cfg = load_yaml("tasks.yaml")

    def make(key: str, context: list[Task] | None = None) -> Task:
        spec = cfg[key]
        return Task(
            name=spec["name"],
            description=spec["description"].format(query=query).strip(),
            expected_output=spec["expected_output"].strip(),
            context=context or [],
        )

    research = make("research_task")
    analysis = make("analysis_task", [research])
    writing = make("writing_task", [research, analysis])
    return [research, analysis, writing]

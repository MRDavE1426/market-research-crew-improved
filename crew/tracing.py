"""Hooks CrewAI's event bus into the BenchmarkLogger.

Why events instead of ``step_callback``: step callbacks only fire for agents
that use tools (the tool-less Writer and the hierarchical manager would be
invisible). The event bus reports every agent run and every tool call -
including the manager's "Delegate work to coworker" calls - so the routing
history and the per-agent timings are complete.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager

from .benchmarks import BenchmarkLogger

log = logging.getLogger("crew.tracing")
DELEGATION_HINTS = ("delegate work", "ask question")


def _coworker(tool_args) -> str | None:
    try:
        data = json.loads(tool_args) if isinstance(tool_args, str) else dict(tool_args)
    except (TypeError, ValueError):
        return None
    value = data.get("coworker") if isinstance(data, dict) else None
    return str(value).strip() if value else None


@contextmanager
def trace_crew_events(bench: BenchmarkLogger, worker_roles: set[str], manager_label: str = "Crew Manager"):
    try:
        from crewai.events import crewai_event_bus
        from crewai.events.types.agent_events import (
            AgentExecutionCompletedEvent,
            AgentExecutionStartedEvent,
        )
        from crewai.events.types.tool_usage_events import ToolUsageStartedEvent
    except ImportError:  # very old CrewAI: degrade to task-level timings only
        log.warning("CrewAI event bus not available; routing history will be limited")
        yield
        return

    started: dict[tuple, object] = {}

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _agent_started(source, event):
            role = getattr(event.agent, "role", "?").strip()
            key = (role, str(getattr(event.task, "id", "")))
            started[key] = event.timestamp
            if role in worker_roles:
                bench.note_worker_started(role)
                bench.record_event("agent_started", role, None, getattr(event.task, "name", "") or "")
            else:
                bench.record_event("manager_started", manager_label, None, getattr(event.task, "name", "") or "")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def _agent_completed(source, event):
            role = getattr(event.agent, "role", "?").strip()
            key = (role, str(getattr(event.task, "id", "")))
            begin = started.pop(key, None)
            seconds = (event.timestamp - begin).total_seconds() if begin else 0.0
            if role in worker_roles:
                bench.record_agent_run(role, seconds)
                bench.record_event("agent_completed", role, None, f"{seconds:.2f}s")

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _tool_started(source, event):
            role = (event.agent_role or "?").strip()
            name = event.tool_name or "?"
            if any(h in name.lower() for h in DELEGATION_HINTS):
                label = manager_label if role not in worker_roles else role
                bench.record_event("delegation", label, _coworker(event.tool_args), f"{name}: {event.tool_args}")
            else:
                bench.record_event("tool_call", role, name, str(event.tool_args))

        try:
            yield
        finally:
            crewai_event_bus.flush()

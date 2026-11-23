"""A deterministic fake LLM that drives the REAL CrewAI hierarchical runtime.

It answers in CrewAI's ReAct text format, so the manager really delegates via
the "Delegate work to coworker" tool, workers really call our @tool functions,
and all step/task callbacks fire exactly as they would with a hosted model.
"""
from __future__ import annotations

import json

from crewai import BaseLLM

CSV = ("company,revenue_usd_bn,net_income_usd_bn\n"
       "Acme Cloud,40.5,9.1\nBeta Chips,25.0,6.2\nGamma Soft,12.3,2.4\n")

REPORT = """## Executive Summary
Scripted report for wiring tests.

## Key Financial Metrics
| metric | mean |
| --- | --- |
| revenue_usd_bn | 25.93 |

## Sources
- scripted
"""


def _text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


def _react(thought: str, action: str, payload: dict) -> str:
    return f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(payload)}"


def _final(answer: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {answer}"


class ScriptedLLM(BaseLLM):
    llm_type: str = "scripted"

    def __init__(self, csv_path: str = "", **kw):
        super().__init__(model="scripted-llm", **kw)
        self._csv_path = csv_path
        self._manager_calls = 0
        self.calls: list[str] = []

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 100_000

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        role = getattr(from_agent, "role", "") or ""
        text = _text(messages)
        self.calls.append(role)

        if role == "Crew Manager":
            i, self._manager_calls = self._manager_calls, self._manager_calls + 1
            script = [
                _react("Research first", "Delegate work to coworker",
                       {"task": "collect metrics", "context": "tech sector", "coworker": "Market Research Specialist"}),
                _final("Research finished: metrics CSV saved."),
                _react("Now analyse", "Delegate work to coworker",
                       {"task": "analyse the csv", "context": "csv saved", "coworker": "Financial Data Analyst"}),
                _final("Analysis finished."),
                _react("Now write", "Delegate work to coworker",
                       {"task": "write the report", "context": "analysis done", "coworker": "Technical Reports Writer"}),
                _final(REPORT),
            ]
            return script[i]
        if role == "Market Research Specialist":
            if "Metrics saved to" in text:
                return _final("Saved metrics. Sources: scripted.")
            return _react("Save data", "Save Metrics CSV", {"csv_content": CSV})
        if role == "Financial Data Analyst":
            if "Summary Statistics" in text and "Observation" in text:
                return _final("Statistics computed; no data-quality issues.")
            return _react("Run stats", "Calculate Financial Metrics", {"csv_path": self._csv_path})
        if role == "Technical Reports Writer":
            return _final(REPORT)
        return _final("ok")

"""Import shim for the ``@tool`` decorator.

The brief shows ``from crewai_tools import tool`` (older CrewAI releases).
Current CrewAI releases expose it as ``crewai.tools.tool``. We support both.
"""
try:  # modern CrewAI
    from crewai.tools import tool
except ImportError:  # pragma: no cover - older CrewAI
    from crewai_tools import tool  # type: ignore

__all__ = ["tool"]

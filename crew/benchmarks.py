"""Execution benchmarking: per-task timings + task-routing history -> benchmarks.json."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("crew.benchmarks")
MAX_RUNS_KEPT = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BenchmarkLogger:
    """Collects timings for one pipeline run and appends it to ``benchmarks.json``."""

    def __init__(self, path: str | Path, query: str, mode: str = "live", manager_llm: str = "-"):
        self.path = Path(path)
        self.run_id = uuid.uuid4().hex[:8]
        self.query = query
        self.mode = mode
        self.manager_llm = manager_llm
        self.started_at = _now_iso()
        self._t0 = time.perf_counter()
        self._last_mark = self._t0
        self._lock = threading.Lock()
        self.tasks: list[dict] = []
        self.routing_history: list[dict] = []
        self._agent_activity: dict[str, dict] = {}
        self._latest_worker: str | None = None
        self._finished: dict | None = None

    # ---- routing / trace ---------------------------------------------------
    def record_event(self, event: str, source: str, target: str | None = None, detail: str = "") -> None:
        entry = {
            "timestamp": _now_iso(),
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
            "event": event,
            "from": source,
            "to": target,
            "detail": (detail or "")[:300],
        }
        with self._lock:
            self.routing_history.append(entry)
        log.info("[ROUTING] %s | %s -> %s | %s", event, source, target or "-", entry["detail"])

    def note_worker_started(self, role: str) -> None:
        """Remember which specialist is currently working (for task attribution)."""
        with self._lock:
            self._latest_worker = role

    def record_agent_run(self, role: str, seconds: float) -> None:
        """One finished agent execution (one delegated task) and how long it took."""
        with self._lock:
            act = self._agent_activity.setdefault(role, {"runs": 0, "active_seconds": 0.0})
            act["runs"] += 1
            act["active_seconds"] += max(seconds, 0.0)

    def latest_active_agent(self) -> str | None:
        """Role of the specialist that started working most recently."""
        with self._lock:
            return self._latest_worker

    # ---- task timing ------------------------------------------------------------
    def mark_start(self) -> None:
        self._last_mark = time.perf_counter()

    def record_task(self, name: str, agent: str, duration_s: float | None = None,
                    status: str = "completed", summary: str = "") -> dict:
        """Log one finished task. If ``duration_s`` is omitted the time since the
        previous task finished (or since ``mark_start``) is used - exact for the
        sequential task chain used by this crew."""
        now = time.perf_counter()
        if duration_s is None:
            duration_s = now - self._last_mark
        self._last_mark = now
        entry = {
            "task": name,
            "agent": agent,
            "duration_seconds": round(duration_s, 3),
            "status": status,
            "finished_at": _now_iso(),
            "output_preview": (summary or "")[:160],
        }
        with self._lock:
            self.tasks.append(entry)
        self.record_event("task_completed", agent, None, f"{name} finished in {entry['duration_seconds']}s")
        return entry

    @contextmanager
    def task(self, name: str, agent: str):
        """Context manager for code that runs a task in-process (offline mode)."""
        self.record_event("task_started", "Crew Manager", agent, name)
        start = time.perf_counter()
        status = "completed"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            self.record_task(name, agent, time.perf_counter() - start, status=status)

    # ---- output --------------------------------------------------------------------
    def finish(self, status: str = "completed", error: str | None = None, artifacts: dict | None = None) -> dict:
        total = round(time.perf_counter() - self._t0, 3)
        activity = {
            role: {"runs": a["runs"], "active_seconds": round(a["active_seconds"], 3)}
            for role, a in self._agent_activity.items()
        }
        run = {
            "run_id": self.run_id,
            "query": self.query,
            "mode": self.mode,
            "manager_llm": self.manager_llm,
            "process": "hierarchical",
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "total_duration_seconds": total,
            "status": status,
            "error": error,
            "tasks": self.tasks,
            "agent_activity": activity,
            "routing_history": self.routing_history,
            "artifacts": artifacts or {},
        }
        self._finished = run
        self._append_to_file(run)
        return run

    def _append_to_file(self, run: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {"runs": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                log.warning("benchmarks file unreadable; starting a fresh one")
        data["runs"] = (data["runs"] + [run])[-MAX_RUNS_KEPT:]
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.chmod(tmp, 0o644)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

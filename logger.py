"""
logger.py — structured, per-step observability for the agent loop (Milestone 3).

Every Perceive/Reason/Act/Reflect call gets one JSON line written to
logs/<session_id>.jsonl via StructuredLogger.step(), capturing: iteration,
step name, duration_ms, a compact input/output snapshot, and any error.
One-off events (retries, fallbacks, guardrail trips) go through .event().
A short human-readable line is also echoed to the console when verbose=True.
"""

from __future__ import annotations

import os
import json
import time
import contextlib
from datetime import datetime, timezone


def _jsonable(value, limit: int = 400) -> str:
    """Best-effort compact, JSON-safe rendering of an arbitrary payload."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


class StructuredLogger:
    def __init__(self, session_id: str, log_dir: str = "logs", verbose: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{session_id}.jsonl")
        self.session_id = session_id
        self.verbose = verbose
        self._fh = open(self.path, "a", encoding="utf-8")

    def _write(self, record: dict) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["session_id"] = self.session_id
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._fh.flush()

    def event(self, event: str, **fields) -> None:
        """One-off structured events: retries, fallbacks, guardrail trips."""
        self._write({"kind": "event", "event": event, **fields})
        if self.verbose:
            detail = ", ".join(f"{k}={v}" for k, v in fields.items())
            print(f"  [log] {event}: {detail}")

    @contextlib.contextmanager
    def step(self, step_name: str, iteration: int, inputs=None):
        """Wrap one Perceive/Reason/Act/Reflect call.

            with logger.step("reason", state.iteration) as rec:
                ... do the step ...
                rec["output"] = {...}   # optional, set by the caller
        """
        record = {"kind": "step", "step": step_name, "iteration": iteration,
                  "input_raw": inputs, "output": None, "error": None}
        start = time.perf_counter()
        try:
            yield record
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
            record["input"] = _jsonable(record.pop("input_raw"))
            record["output"] = _jsonable(record.get("output"))
            self._write(record)
            if self.verbose:
                status = "ERROR" if record["error"] else "ok"
                print(f"  [log] {step_name:8s} iter={iteration} "
                      f"{record['duration_ms']:7.1f}ms [{status}]")

    def close(self) -> None:
        self._fh.close()

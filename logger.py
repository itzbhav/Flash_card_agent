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

    def iteration_start(self, iteration: int, max_iterations: int) -> None:
        """Mark the beginning of a new iteration in the log."""
        self._write({"kind": "iteration_start", "iteration": iteration,
                     "max_iterations": max_iterations})
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  >> ITERATION {iteration + 1} / {max_iterations}")
            print(f"{'=' * 60}")

    def iteration_end(self, iteration: int, cards_so_far: int,
                      tokens_used: int, token_budget: int) -> None:
        """Mark the end of an iteration with a progress snapshot."""
        self._write({"kind": "iteration_end", "iteration": iteration,
                     "cards_so_far": cards_so_far,
                     "tokens_used": tokens_used,
                     "token_budget": token_budget})
        if self.verbose:
            print(f"  {'-' * 50}")
            print(f"  [OK] Iteration {iteration + 1} complete | "
                  f"Cards: {cards_so_far} | "
                  f"Tokens: {tokens_used}/{token_budget}")

    def session_summary(self, total_iterations: int, max_iterations: int,
                        total_cards: int, tokens_used: int,
                        token_budget: int, stop_reason: str,
                        model: str, duration_s: float) -> None:
        """Write a final summary record capturing the full session outcome."""
        self._write({"kind": "session_summary",
                     "total_iterations": total_iterations,
                     "max_iterations": max_iterations,
                     "total_cards": total_cards,
                     "tokens_used": tokens_used,
                     "token_budget": token_budget,
                     "stop_reason": stop_reason,
                     "model": model,
                     "duration_s": round(duration_s, 2)})
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  [*] SESSION SUMMARY")
            print(f"{'=' * 60}")
            print(f"  Stop reason   : {stop_reason}")
            print(f"  Iterations    : {total_iterations} / {max_iterations}")
            print(f"  Cards created : {total_cards}")
            print(f"  Tokens used   : {tokens_used} / {token_budget}")
            print(f"  Model         : {model}")
            print(f"  Duration      : {duration_s:.1f}s")
            print(f"{'=' * 60}")

    def actor_draft(self, iteration: int, concept: str, front: str, back: str, revision: int = 0) -> None:
        """Log an explicit Actor draft generation step."""
        self._write({
            "kind": "actor_output",
            "role": "actor",
            "iteration": iteration,
            "concept": concept,
            "front": front,
            "back": back,
            "revision": revision
        })
        if self.verbose:
            print(f"  [Actor] Draft (rev {revision}): concept='{concept}' | front='{front[:40]}...'")

    def judge_eval(self, iteration: int, concept: str, scores: dict, feedback: str, passed: bool, rubric: list[str] | None = None) -> None:
        """Log an explicit Judge evaluation step against a named rubric."""
        self._write({
            "kind": "judge_output",
            "role": "judge",
            "iteration": iteration,
            "concept": concept,
            "scores": scores,
            "rubric": rubric or ["Accuracy", "Clarity", "Atomicity"],
            "feedback": feedback,
            "passed": passed
        })
        if self.verbose:
            verdict = "PASS" if passed else "REVISE"
            print(f"  [Judge] Verdict: {verdict} | Scores: {scores} | Feedback: {feedback[:50]}...")

    def revision_step(self, iteration: int, concept: str, revision: int, prev_front: str, new_front: str) -> None:
        """Log a revision transition step taking judge feedback into account."""
        self._write({
            "kind": "revision_applied",
            "role": "actor",
            "iteration": iteration,
            "concept": concept,
            "revision": revision,
            "prev_front": prev_front,
            "new_front": new_front
        })
        if self.verbose:
            print(f"  [Revision] Applied rev #{revision} for '{concept}'")

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
                arrow = "[X]" if record["error"] else "[+]"
                print(f"  {arrow} {step_name:8s}  | iter {iteration + 1}  | "
                      f"{record['duration_ms']:7.1f}ms [{status}]")

    def close(self) -> None:
        self._fh.close()

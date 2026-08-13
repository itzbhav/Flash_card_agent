"""
harness.py — production scaffolding around the agent loop (Milestone 3):
retries with exponential backoff + jitter, explicit fallback strategies per
failure mode, and guardrails (hard iteration cap, token budget, stuck-loop
detection).

This module knows nothing about flashcards specifically. It wraps three
kinds of risky calls the loop makes -- LLM calls, tool calls, memory reads --
and exposes guardrail checks the loop consults every iteration. cognition.py
calls into this through a Harness instance attached to AgentState.harness;
when that's None (e.g. the Milestone 1/2 demos), callers fall back to
calling the underlying functions directly, so nothing here is mandatory.
"""

from __future__ import annotations

import time
import random

import openai

from config import HarnessConfig
from logger import StructuredLogger
from llm import call_llm, extract_usage


# ---------------------------------------------------------------------------
# Guardrail signals -- raised, never silently swallowed, so the loop always
# knows exactly why it stopped.
# ---------------------------------------------------------------------------
class GuardrailTripped(Exception):
    """Base class for a guardrail forcing the loop to stop."""


class TokenBudgetExceeded(GuardrailTripped):
    pass


class StuckLoopDetected(GuardrailTripped):
    pass


# Transient, retryable failures. Anything else (bad API key, bad request
# shape, a 400) fails fast instead of burning retry budget on an error that
# will never resolve itself no matter how many times it's retried.
_RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _sleep_with_backoff(attempt: int, cfg, rng=random) -> float:
    """Exponential backoff (base * 2^attempt, capped) with +/- jitter so a
    fleet of retries doesn't all wake up and hit the API at the same instant
    (the classic thundering-herd problem behind retry storms)."""
    delay = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
    jittered = delay + rng.uniform(-cfg.jitter * delay, cfg.jitter * delay)
    jittered = max(0.0, jittered)
    time.sleep(jittered)
    return jittered


class Harness:
    """Bundles config + logger + guardrail state for one agent run."""

    def __init__(self, config: HarnessConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.tokens_used = 0
        self._recent_actions: list[str] = []

    # ---- FAILURE MODE: transient API/tool failures -> retry ---------------
    def retry(self, fn, *, label: str, retryable=_RETRYABLE_EXCEPTIONS, **kwargs):
        """Call fn(**kwargs) with exponential backoff + jitter. Retries only
        on `retryable` exceptions (rate limits, timeouts, transient 5xxs);
        anything else propagates on the first attempt."""
        cfg = self.config.retry
        last_err = None
        for attempt in range(cfg.max_retries + 1):
            try:
                return fn(**kwargs)
            except retryable as e:
                last_err = e
                if attempt >= cfg.max_retries:
                    break
                delay = _sleep_with_backoff(attempt, cfg)
                self.logger.event("retry", label=label, attempt=attempt + 1,
                                  max_retries=cfg.max_retries,
                                  delay_s=round(delay, 2),
                                  error=f"{type(e).__name__}: {e}")
        self.logger.event("retry_exhausted", label=label,
                          error=f"{type(last_err).__name__}: {last_err}")
        raise last_err

    # ---- LLM calls: retried, token-budget tracked --------------------------
    def call_llm(self, **kwargs):
        model = kwargs.pop("model", self.config.model)
        resp = self.retry(lambda: call_llm(model=model, **kwargs), label="llm_call")
        usage = extract_usage(resp)
        self.tokens_used += usage.get("total_tokens", 0)
        self.logger.event("token_usage", used_this_call=usage.get("total_tokens", 0),
                          cumulative=self.tokens_used,
                          budget=self.config.guardrails.token_budget)
        if self.tokens_used > self.config.guardrails.token_budget:
            raise TokenBudgetExceeded(
                f"used {self.tokens_used} > budget "
                f"{self.config.guardrails.token_budget}")
        return resp

    # ---- FAILURE MODE: failed tool calls -> retry, then fallback ----------
    def call_tool(self, name: str, fn, *args, **kwargs) -> dict:
        """Retries a tool handler on transient failures. If retries are
        exhausted, or the handler raises a non-retryable error (e.g. it
        couldn't produce parseable JSON even after its own internal retry),
        returns a fallback failure dict instead of raising -- callers treat
        that as a normal (if disappointing) tool result and keep the loop
        alive rather than crashing on one bad tool call."""
        try:
            result = self.retry(lambda: fn(*args, **kwargs), label=f"tool:{name}")
            return {"ok": True, "result": result}
        except GuardrailTripped:
            raise  # a guardrail trip is never a "this tool failed" event
        except Exception as e:
            self.logger.event("tool_call_failed", tool=name,
                              error=f"{type(e).__name__}: {e}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---- FAILURE MODE: memory read failures -> fallback to empty ----------
    def recall_safe(self, memory, query: str, **kwargs) -> list:
        if memory is None:
            return []
        try:
            return memory.recall(query, **kwargs)
        except Exception as e:
            self.logger.event("memory_recall_failed", query=query[:80],
                              error=f"{type(e).__name__}: {e}")
            return []  # agent reasons without memory context rather than crash

    def save_safe(self, memory, item: dict, session_id: str) -> None:
        if memory is None:
            return
        try:
            memory.save(item, session_id)
        except Exception as e:
            self.logger.event("memory_save_failed",
                              error=f"{type(e).__name__}: {e}")

    # ---- FAILURE MODE: unparseable LLM output ------------------------------
    # No dedicated method: cognition.py's reason()/reflect() already carry a
    # deterministic fallback action / generic note for this case. This
    # harness's job is making sure the attempt was retried and the failure
    # was logged (via call_llm above) before that fallback kicks in.

    # ---- GUARDRAIL: hard iteration cap -------------------------------------
    # Enforced by the loop's `while state.iteration < state.max_iterations`
    # condition (models.py / loop.py) -- max_iterations is itself sourced
    # from config.guardrails.max_iterations, never hardcoded in loop code.

    # ---- GUARDRAIL: stuck-loop detection -----------------------------------
    def record_action(self, action_name: str, action_args: dict) -> None:
        sig = f"{action_name}:{sorted(action_args.items())}"
        self._recent_actions.append(sig)
        window = self.config.guardrails.stuck_window
        self._recent_actions = self._recent_actions[-window:]

    def is_stuck(self) -> bool:
        window = self.config.guardrails.stuck_window
        threshold = self.config.guardrails.stuck_threshold
        if len(self._recent_actions) < window:
            return False
        most_common = max(set(self._recent_actions), key=self._recent_actions.count)
        return self._recent_actions.count(most_common) >= threshold

    def check_stuck(self) -> None:
        if self.is_stuck():
            self.logger.event("stuck_loop_detected", recent=self._recent_actions)
            raise StuckLoopDetected(
                f"same action repeated {self.config.guardrails.stuck_threshold}+ "
                f"times within the last {self.config.guardrails.stuck_window} "
                f"iterations")

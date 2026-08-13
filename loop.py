

from __future__ import annotations

import json

import cognition
from models import AgentState
from memory import Memory, new_session_id
from config import HarnessConfig, load_config
from harness import Harness, TokenBudgetExceeded, StuckLoopDetected
from logger import StructuredLogger


def _truncate(text: str, n: int = 140) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def _log_iteration(state: AgentState) -> None:
    """Print a readable trace of one Thought -> Action -> Observation cycle."""
    action = state.last_action or {}
    print(f"\n--- Iteration {state.iteration} " + "-" * 40)
    if state.last_recall:
        preview = "; ".join(
            f"{r['concept']}(score={r['_score']}, session={r['session_id']})"
            for r in state.last_recall)
        print(f"  MEMORY READ : {len(state.last_recall)} hit(s) -> {preview}")
    else:
        print(f"  MEMORY READ : (no relevant memories yet)")
    print(f"  ACT   : {action.get('name','?')}  args={action.get('input',{})}")
    print(f"  OBSERV: {_truncate(state.last_observation)}")
    if state.reflection_notes:
        print(f"  REFLECT: {_truncate(state.reflection_notes[-1])}")
    if state.last_new_cards:
        print(f"  MEMORY WRITE: saved {len(state.last_new_cards)} new "
              f"card(s) + this reflection note")
    if state.harness is not None:
        budget = state.harness.config.guardrails.token_budget
        print(f"  TOKENS: {state.harness.tokens_used} / {budget}")
    print(f"  STATE : {state.summary()}")


def run_agent(material: str,
              max_iterations: int | None = None,
              verbose: bool = True,
              memory: Memory | None = None,
              session_id: str | None = None,
              config: HarnessConfig | None = None,
              harness: Harness | None = None) -> AgentState:
    """Run the full Perceive->Reason->Act->Reflect loop on `material`.

    `memory` / `session_id` let a caller share one persistent Memory store and
    control session boundaries across multiple run_agent() calls (see
    demo_memory.py). `config` / `harness` wire in the Milestone 3 production
    scaffolding (retries, guardrails, structured logging); if omitted, a
    config is loaded from config.json (or FLASHCARD_* env vars, see
    config.py) and a fresh Harness is built from it, so a single call still
    gets harnessed behavior for free. `max_iterations` overrides the
    config-supplied guardrail cap for this one call.

    Returns the final AgentState. state.stop_reason explains why it stopped:
    "task_complete" | "iteration_cap" | "token_budget_exceeded" |
    "stuck_loop_detected".
    """
    config = config or load_config()
    session_id = session_id or new_session_id()
    logger = StructuredLogger(session_id, log_dir=config.log_dir,
                              verbose=verbose and config.verbose)
    harness = harness or Harness(config, logger)

    state = AgentState(
        raw_material=material,
        max_iterations=max_iterations or config.guardrails.max_iterations,
    )
    state.memory = memory if memory is not None else Memory(path=config.memory.path)
    state.session_id = session_id
    state.harness = harness

    if verbose:
        print(f"session_id: {state.session_id}  model={config.model}  "
              f"max_iterations={state.max_iterations}  "
              f"token_budget={config.guardrails.token_budget}")
        print(f"(memory store has {state.memory.count()} record(s) so far)")
        print("=" * 56)
        print("FLASHCARD AGENT - starting")
        print("=" * 56)

    try:
        while not state.done and state.iteration < state.max_iterations:
            with logger.step("perceive", state.iteration):
                state = cognition.perceive(state)

            with logger.step("reason", state.iteration) as rec:
                state = cognition.reason(state)
                rec["output"] = state.last_action

            action = state.last_action or {}
            # GUARDRAIL: stuck-loop detection -- record the action reason()
            # just chose, then check whether recent history is a repeat cycle
            # before we spend an Act/Reflect pair executing it again.
            harness.record_action(action.get("name", "?"), action.get("input", {}) or {})
            harness.check_stuck()

            with logger.step("act", state.iteration, action) as rec:
                state = cognition.act(state)
                rec["output"] = state.last_observation

            with logger.step("reflect", state.iteration) as rec:
                state = cognition.reflect(state)
                rec["output"] = state.reflection_notes[-1] if state.reflection_notes else None

            if verbose:
                _log_iteration(state)

            state.iteration += 1

        # GUARDRAIL: hard iteration cap -- the while condition above is the
        # enforcement; this just records *why* the loop exited for the report.
        state.stop_reason = "task_complete" if state.done else "iteration_cap"
        if state.stop_reason == "iteration_cap":
            logger.event("iteration_cap_reached", iteration=state.iteration,
                        max_iterations=state.max_iterations)

    except TokenBudgetExceeded as e:
        state.stop_reason = "token_budget_exceeded"
        logger.event("guardrail_stop", reason="token_budget_exceeded", detail=str(e))
    except StuckLoopDetected as e:
        state.stop_reason = "stuck_loop_detected"
        logger.event("guardrail_stop", reason="stuck_loop_detected", detail=str(e))

    # ---- termination report -------------------------------------------------
    if verbose:
        print("\n" + "=" * 56)
        print(f"FINISHED (stop_reason={state.stop_reason}) after "
              f"{state.iteration} iterations with {len(state.flashcards)} cards.")
        print(f"Tokens used: {harness.tokens_used} / {config.guardrails.token_budget}")
        print(f"Memory store now has {state.memory.count()} record(s) total "
              f"(persists to {state.memory.path}).")
        print(f"Structured log: {logger.path}")
        print("=" * 56)

    logger.close()
    return state


def save_deck(state: AgentState, path: str = "flashcards.json") -> str:
    """Finalize: write the deck to a JSON file (saving is a code step, not a
    tool). Returns the path written."""
    payload = {
        "topic": state.perception.topic if state.perception else "",
        "count": len(state.flashcards),
        "iterations": state.iteration,
        "completed": state.done,
        "stop_reason": state.stop_reason,
        "flashcards": [c.to_dict() for c in state.flashcards],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

# Flashcard Agent

An AI agent that turns raw study material into a reviewed deck of flashcards — built from scratch as a four-step reasoning loop (Perceive → Reason → Act → Reflect), with a persistent memory layer and a production-grade harness wrapped around it. No agent framework (LangChain, etc.) is used anywhere.

Built in three milestones, each layered on top of the last without changing what came before:

| Milestone | What it adds |
|---|---|
| **1 — Agent loop** | The four-step reasoning loop, five tools, hard iteration cap, clean-completion stop condition |
| **2 — Memory** | A persistent store the loop reads from and writes to every iteration, surviving across sessions |
| **3 — Harness** | Retries, fallback strategies, structured logging, and guardrails so it can run unsupervised |

---

## Quick start

```bash
pip install -r requirements.txt
# put your key in .env as OPENAI_API_KEY="sk-..."

python test_setup.py      # sanity-check the API key + model access
python demo.py             # Milestone 1: run the loop on sample material
python demo_memory.py      # Milestone 2: prove memory persists across sessions
python main.py              # Milestone 3: the config-driven production entry point
```

`main.py` also accepts your own material and a custom config:

```bash
python main.py notes.txt
python main.py notes.txt my_config.json
```

---

## Architecture — the four-step loop

Everything flows through one object, `AgentState` (`models.py`), passed through every step of every iteration:

```
Perceive → Reason → Act → Reflect → (loop until done or iteration cap)
```

| Step | File | What it does |
|---|---|---|
| **Perceive** | `cognition.py::perceive` | Pure Python, no LLM call. Chunks the raw material into paragraphs and guesses a topic. Runs once. |
| **Reason** | `cognition.py::reason` | The ReAct decision step. Recalls relevant memory, then asks the LLM to pick exactly one tool given the current status. |
| **Act** | `cognition.py::act` | Executes the chosen tool, resolves indexes to real cards/text, updates the deck. |
| **Reflect** | `cognition.py::reflect` | The Reflexion step. Critiques the last action in 1-2 sentences, writes new cards + the critique to memory, decides `done`. |

The loop itself lives in `loop.py::run_agent`, driven by `while not state.done and state.iteration < state.max_iterations`.

### Tools (`tools.py`)

Five tools, each tagged with the reasoning pattern it serves:

| Tool | Pattern |
|---|---|
| `extract_key_concepts` | Chain-of-Thought — decompose before producing |
| `generate_flashcards` | Tree-of-Thoughts hook — can branch N candidate cards |
| `score_flashcard` | Value function that ToT/LATS-style branching depends on |
| `revise_flashcard` | Reflexion actuator — acts on critique |
| `review_deck` | Reflexion signal — produces the critique that drives `done` |

**ReAct** is the loop's backbone; **Reflexion** is the quality loop (`review_deck` → critique → `revise_flashcard`); **Chain-of-Thought** lives inside `extract_key_concepts`. **LATS** was deliberately not implemented — full trajectory search is overkill for a single linear deck-building task.

---

## Memory (Milestone 2)

**Backend:** a single JSON file (`memory.py`), scored with lexical similarity (token-overlap + sequence-match) — no vector DB, no embeddings API. It's small, textual data; a flat file is enough, and it's instantly inspectable and dependency-free.

**Three operations:**

```python
memory.save(item, session_id)              # store a flashcard or reflection note
memory.recall(query, k=5, exclude_session=None)  # top-k relevant records
memory.clear_session(session_id)            # forget one session's records
```

**Where it's wired in:**
- **Read** — start of `reason()`: builds a query from the topic + current chunk, recalls up to 3 relevant records (cards *and* past reflection notes) from this session's earlier iterations and from **previous sessions**, and injects them into the prompt so the LLM avoids re-generating cards for concepts it already covered.
- **Write** — end of `reflect()`: every new card from that iteration, plus the reflection note itself, is saved.

**Proof it matters, not just plumbing:** `demo_memory.py` runs the agent twice on identical material sharing one store. Run 2's very first reasoning step — before it has made a single card of its own — already recalls three of Run 1's cards. That's a separate process run being measurably influenced by what an earlier run wrote to disk.

Default store path: `memory_store.json` (configurable — see below).

---

## Harness (Milestone 3)

Production scaffolding wrapped around the loop so it can run unsupervised. Nothing here changes the four-step shape — it wraps the risky calls the loop already makes (LLM calls, tool calls, memory reads) and adds guardrails the loop checks every iteration.

### Retry logic (`harness.py`)

Every LLM call and every tool call goes through `Harness.retry()` — exponential backoff (`base_delay * 2^attempt`, capped at `max_delay`) plus random jitter, so concurrent retries don't all fire at once and re-trigger the same rate limit. Only a narrow, explicit set of transient exceptions retry (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`); anything else (a bad request, a bug) fails on the first attempt instead of burning retry budget on an error that will never resolve.

### Fallback strategies — one per failure mode

| Failure mode | Fallback |
|---|---|
| Unparseable LLM output (Reason) | Falls through to a deterministic next action (drain pending concepts → extract → review) so the loop never stalls |
| Unparseable LLM output (Reflect) | Logs a generic `"(reflection parse issue: ...)"` note and continues |
| Failed tool call (retries exhausted) | Recorded as a normal observation string; the loop advances to the next iteration instead of crashing |
| Hitting the iteration cap | Loop stops cleanly, `state.stop_reason = "iteration_cap"`, partial deck is still saved |
| Memory read failure | `recall_safe()` catches the exception, logs it, returns `[]` — the agent reasons without memory context rather than crashing the step |

### Observability (`logger.py`)

`StructuredLogger` writes one JSON line per Perceive/Reason/Act/Reflect call to `logs/<session_id>.jsonl` — step name, iteration, duration, compact input/output, and any error, captured automatically by a context manager. One-off events (retries, fallbacks, guardrail trips, token usage) are logged the same way.

### Guardrails

- **Hard iteration cap** — `config.guardrails.max_iterations`, enforced by the loop's own `while` condition.
- **Token budget** — `Harness.call_llm()` sums `total_tokens` from every call (including calls made inside tool handlers) against `config.guardrails.token_budget`; raises `TokenBudgetExceeded` when crossed.
- **Stuck-loop detection** — a rolling window of the last N chosen actions; if one repeats past a threshold, raises `StuckLoopDetected` before wasting another Act/Reflect pair on it.

### Configuration (`config.py` / `config.json`)

Every runtime parameter — model, retry settings, memory backend/path, iteration cap, token budget, stuck-loop thresholds — is set here, with `FLASHCARD_*` environment variable overrides on top. `loop.py` and `cognition.py` never hardcode any of these.

```json
{
  "model": "gpt-4o",
  "retry": { "max_retries": 4, "base_delay": 1.0, "max_delay": 20.0, "jitter": 0.5 },
  "memory": { "backend": "json_file", "path": "memory_store.json" },
  "guardrails": {
    "max_iterations": 15,
    "token_budget": 60000,
    "stuck_window": 4,
    "stuck_threshold": 3
  }
}
```

---

## File structure

```
models.py          AgentState, Flashcard, Chunk, Perception — the shared data
llm.py              OpenAI client wrapper (call_llm, extract_usage)
tools.py             The five tools: schemas + handlers
prompts.py           REASON / REFLECT prompt templates
cognition.py         perceive / reason / act / reflect
memory.py            Memory: save / recall / clear_session
config.py            HarnessConfig, loaded from config.json + env vars
config.json           Default runtime configuration
harness.py            Harness: retry, fallback wrappers, guardrails
logger.py             StructuredLogger: JSONL step + event logging
loop.py               run_agent — wires cognition + memory + harness together
main.py               Production entry point (config-driven)
demo.py               Milestone 1 demo (sample material, no harness needed)
demo_memory.py         Milestone 2 demo (two runs, shared memory, cross-session proof)
test_setup.py          Pre-flight check: API key + model access
```

Generated at runtime (not source):

```
memory_store.json         Persistent memory store (default path)
memory_store_demo.json     Separate store used by demo_memory.py
flashcards.json             Finished deck output
logs/<session_id>.jsonl     Structured per-step logs
```

---

## Stop reasons

`main.py` / `run_agent()` always report why a run ended, via `state.stop_reason`:

| Value | Meaning |
|---|---|
| `task_complete` | Material fully covered, deck reviewed, no unresolved weak cards |
| `iteration_cap` | Hit `max_iterations` before finishing — partial deck still saved |
| `token_budget_exceeded` | Cumulative token usage crossed the configured budget |
| `stuck_loop_detected` | The same action repeated past the stuck-loop threshold |

`main.py` exits with code `1` on anything but `task_complete`, so an unsupervised caller (cron, CI) can detect an incomplete run without parsing logs.
#   F l a s h _ c a r d _ a g e n t  
 #   F l a s h _ c a r d _ a g e n t  
 
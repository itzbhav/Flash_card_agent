# Flashcard Agent

An AI agent that turns raw study material into a reviewed deck of flashcards — built from scratch as a four-step reasoning loop (**Perceive → Reason → Act → Reflect**), with a persistent memory layer and a production-grade harness wrapped around it. No agent framework (LangChain, LlamaIndex, etc.) is used anywhere; the loop, the tools, the memory, and the harness are all hand-written.

Built in three milestones, each layered on top of the last without changing what came before:

| Milestone | What it adds |
|---|---|
| **1 — Agent loop** | The four-step reasoning loop, five tools, a hard iteration cap, and a clean-completion stop condition |
| **2 — Memory** | A persistent store the loop reads from and writes to every iteration, surviving across sessions |
| **3 — Harness** | Retries, fallback strategies, structured logging, and guardrails so it can run unsupervised |
| **4 — Judge** | A six-dimension weighted LLM-as-a-Judge with deterministic scoring, a pass threshold, and an Actor–Critic revision loop |

---

## Quick start

```bash
pip install -r requirements.txt          # openai, python-dotenv

# put your key in a .env file:
echo 'OPENAI_API_KEY="sk-..."' > .env

python test_setup.py     # sanity-check the API key + model access
python demo.py           # Milestone 1: run the loop on sample material
python demo_memory.py    # Milestone 2: prove memory persists across sessions
python main.py           # Milestone 3: the config-driven production entry point
```

`main.py` also accepts your own material and an optional custom config:

```bash
python main.py notes.txt
python main.py notes.txt my_config.json
```

With no arguments, `main.py` runs on a built-in sample passage (the water cycle) and uses `config.json`.

---

## Architecture — the four-step loop

Everything flows through one object, `AgentState` (`models.py`), passed through every step of every iteration:

```
Perceive → Reason → Act → Reflect → (loop until done or iteration cap)
```

| Step | Location | What it does |
|---|---|---|
| **Perceive** | `cognition.py::perceive` | Pure Python, no LLM call. Chunks the raw material into paragraphs and guesses a topic. Runs once, up front. |
| **Reason** | `cognition.py::reason` | The ReAct decision step. Recalls relevant memory, then asks the LLM to pick exactly one tool given the current status. |
| **Act** | `cognition.py::act` | Executes the chosen tool, resolves indexes to real cards/text, and updates the deck. |
| **Reflect** | `cognition.py::reflect` | The Reflexion step. Critiques the last action in 1–2 sentences, writes new cards + the critique to memory, and decides `done`. |

The loop itself lives in `loop.py::run_agent`, driven by `while not state.done and state.iteration < state.max_iterations`.

### Tools (`tools.py`)

Five tools, each tagged with the reasoning pattern it serves:

| Tool | Pattern | Role |
|---|---|---|
| `extract_key_concepts` | Chain-of-Thought | Decompose a chunk into testable concepts before producing cards |
| `generate_flashcards` | Actor–Critic loop + Tree-of-Thoughts | Draft *N* candidates, judge each, keep the best, and revise it until it passes |
| `score_flashcard` | LLM-as-a-Judge | Independent six-dimension **weighted** judge with deterministic Python aggregation |
| `revise_flashcard` | Reflexion actuator | Acts on Judge feedback to rewrite a sub-threshold card |
| `review_deck` | Reflexion signal | Audits the deck and produces the critique that drives `done` |

**ReAct** is the loop's backbone; **Reflexion** is the quality loop (`review_deck` → critique → `revise_flashcard`); **Chain-of-Thought** lives inside `extract_key_concepts`. Full trajectory search (LATS) was deliberately left out — it's overkill for a single linear deck-building task.

Each tool ships as a schema (name, description, `input_schema`) plus a `handle_*` function; the schemas are converted to OpenAI's function-calling format in `llm.py`.

### The Actor–Critic Judge (`tools.py`)

`generate_flashcards` is not a one-shot generator — it runs an **inner Actor–Critic revision loop**, the LLM-as-a-Judge pattern applied to card quality:

1. **Actor drafts** *N* candidate cards for the concept (creative temperature).
2. **Judge scores** each draft on a **six-dimension weighted rubric** at temperature 0 — accuracy (25%), completeness (20%), relevance (15%), clarity (15%), pedagogical value (15%), conciseness (10%). The LLM returns six 0–10 sub-scores; **Python computes the weighted average** (`_weighted_score`) and compares it to the pass threshold (default **8.5/10**). The verdict is never the model's to make.
3. **Best-of-N selection** keeps the highest-scoring draft (a Tree-of-Thoughts touch).
4. If that card is below threshold, its Judge feedback is fed back to the **reviser** and the result re-judged, up to `max_revisions` times (Self-Refine). Whatever version scored highest is returned — a **best-version fallback** so a hard concept never blocks the deck.

The Judge is kept **separate from the Actor** and run **greedily (T=0)** so it behaves as an independent, stable reviewer rather than rubber-stamping its own drafts (self-enhancement bias, survey §4.2.1). Every approved card carries its `score`, per-dimension `scores`, `passed` flag, and `revisions` count through to the saved deck.

---

## Memory (Milestone 2)

**Backend:** a single JSON file (`memory.py`), scored with lexical similarity (token-overlap + sequence-match) — no vector DB, no embeddings API. The data is small and textual, so a flat file is enough: it's instantly inspectable and dependency-free.

**Three operations:**

```python
memory.save(item, session_id)                     # store a flashcard or reflection note
memory.recall(query, k=5, exclude_session=None)   # top-k relevant records
memory.clear_session(session_id)                  # forget one session's records
```

**Where it's wired in:**

- **Read** — at the start of `reason()`: builds a query from the topic + current chunk, recalls up to 3 relevant records (cards *and* past reflection notes) from earlier iterations in this session **and from previous sessions**, and injects them into the prompt so the LLM avoids re-generating cards for concepts it already covered.
- **Write** — at the end of `reflect()`: every new card from that iteration, plus the reflection note itself, is saved.

**Proof it matters, not just plumbing:** `demo_memory.py` runs the agent twice on identical material sharing one store. Run 2's very first reasoning step — before it has made a single card of its own — already recalls cards written by Run 1. That's a separate process being measurably influenced by what an earlier run wrote to disk.

Default store path: `memory_store.json` (configurable — see below). The demo uses its own `memory_store_demo.json` so it never pollutes the real store.

---

## Harness (Milestone 3)

Production scaffolding wrapped around the loop so it can run unsupervised. Nothing here changes the four-step shape — it wraps the risky calls the loop already makes (LLM calls, tool calls, memory reads) and adds guardrails the loop checks every iteration.

### Retry logic (`harness.py`)

Every LLM call and every tool call goes through `Harness.retry()` — exponential backoff (`base_delay * 2^attempt`, capped at `max_delay`) plus random jitter, so concurrent retries don't all fire at once and re-trigger the same rate limit. Only a narrow, explicit set of transient exceptions are retried (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`); anything else (a bad request, a bug) fails on the first attempt instead of burning retry budget on an error that will never resolve.

### Fallback strategies — one per failure mode

| Failure mode | Fallback |
|---|---|
| Unparseable LLM output (Reason) | Falls through to a deterministic next action (drain pending concepts → extract → review) so the loop never stalls |
| Unparseable LLM output (Reflect) | Logs a generic `"(reflection parse issue: ...)"` note and continues |
| Failed tool call (retries exhausted) | Recorded as a normal observation string; the loop advances to the next iteration instead of crashing |
| Hitting the iteration cap | Loop stops cleanly, `state.stop_reason = "iteration_cap"`, partial deck is still saved |
| Memory read failure | `recall_safe()` catches the exception, logs it, and returns `[]` — the agent reasons without memory context rather than crashing the step |

### Observability (`logger.py`)

`StructuredLogger` writes one JSON line per Perceive/Reason/Act/Reflect call to `logs/<session_id>.jsonl` — step name, iteration, duration, compact input/output, and any error, captured automatically by a context manager. One-off events (retries, fallbacks, guardrail trips, token usage) are logged the same way.

### Guardrails

- **Hard iteration cap** — `config.guardrails.max_iterations`, enforced by the loop's own `while` condition.
- **Token budget** — `Harness.call_llm()` sums `total_tokens` from every call (including calls made inside tool handlers) against `config.guardrails.token_budget`, and raises `TokenBudgetExceeded` when crossed.
- **Stuck-loop detection** — a rolling window of the last *N* chosen actions; if one repeats past a threshold, it raises `StuckLoopDetected` before wasting another Act/Reflect pair on it.

### Configuration (`config.py` / `config.json`)

Every runtime parameter — model, retry settings, memory backend/path, iteration cap, token budget, stuck-loop thresholds — is set here, with `FLASHCARD_*` environment-variable overrides layered on top. `loop.py` and `cognition.py` never hardcode any of these.

```json
{
  "model": "gpt-4o",
  "log_dir": "logs",
  "verbose": true,
  "retry":   { "max_retries": 4, "base_delay": 1.0, "max_delay": 20.0, "jitter": 0.5 },
  "memory":  { "backend": "json_file", "path": "memory_store.json" },
  "guardrails": {
    "max_iterations": 15,
    "token_budget": 60000,
    "stuck_window": 4,
    "stuck_threshold": 3
  },
  "judge": {
    "threshold": 8.5,
    "max_revisions": 3,
    "temperature": 0.0,
    "weights": {
      "accuracy": 0.25, "completeness": 0.20, "relevance": 0.15,
      "clarity": 0.15, "pedagogical_value": 0.15, "conciseness": 0.10
    }
  }
}
```

Supported environment overrides: `FLASHCARD_MODEL`, `FLASHCARD_MAX_ITERATIONS`, `FLASHCARD_TOKEN_BUDGET`, `FLASHCARD_MAX_RETRIES`, `FLASHCARD_MEMORY_PATH`, `FLASHCARD_LOG_DIR`, `FLASHCARD_JUDGE_THRESHOLD`, `FLASHCARD_JUDGE_MAX_REVISIONS`. An env override always wins over the file, so a container/CI run can tune one value without shipping a new `config.json`.

---

## Model configuration

The agent talks to the OpenAI Chat Completions API through a thin wrapper in `llm.py`. The default model is `gpt-4o`; change it in one place — either `config.json` (`"model"`), the `FLASHCARD_MODEL` env var, or the `MODEL` constant in `llm.py`. Your `OPENAI_API_KEY` is read from the environment (via a `.env` file) and the client is created lazily on first use, so the modules import cleanly even without a key set.

---

## Grounding in the LLM-as-a-Judge survey

The quality half of this agent implements the core apparatus described in *A Survey on LLM-as-a-Judge* (Gu, Jiang, Shi, Guo et al., arXiv:2411.15594). The table below maps the survey's concepts to concrete code.

**Implemented**

| Survey concept | Where in the code |
|---|---|
| LLM-as-a-Judge — *generating scores* (ICL, §2.1.1) | `score_flashcard` / `_judge_card` rate a card on a rubric (`tools.py`) |
| **Criteria decomposition** (§3.1.1) | Six named dimensions scored separately (`_JUDGE_SYS`) |
| **Deterministic weighted aggregation** | `_weighted_score()` computes the 0–10 total in Python; the LLM never sets the final score |
| **Deterministic pass threshold** | `config.judge.threshold` (8.5) enforced in Python, not by the model |
| **Self-enhancement bias mitigation** (§4.2.1) | Judge is a separate role run at **temperature 0**, distinct from the Actor |
| **Self-Refine / Reflexion revision loop** (§2.1.2, §7.1) | Actor–Critic loop in `generate_flashcards`: draft → judge → revise → re-judge |
| **Best-of-N selection** (Tree-of-Thoughts, §2.1.3) | `generate_flashcards` judges *N* drafts and keeps the best |
| Standardizing output format — structured JSON (§3.1.2) | Every tool returns JSON; `_llm_json` re-prompts "JSON only" on a parse failure |
| Evaluations *with explanations* (§3.1.2) | Judge returns per-dimension sub-scores + actionable `feedback` |
| Yes/No-style verdict (§2.1.2) | `passed` flag; `review_deck.overall: good\|needs_work` |
| LLM-as-a-Judge *for agents* (Fig 7) | Actor = generate/reason, Judge = score, self-reflection = reflect, experience = memory |

The design deliberately mirrors the survey's central reliability recommendation: the LLM supplies only the per-dimension sub-scores, while the score that actually gates a card is computed deterministically in Python and checked against a fixed threshold — so the pass/fail decision cannot drift with the model's own arithmetic or self-preference.

**Not implemented (natural next steps, per the survey)**

- **Pairwise comparison** (§2.1.3) — scoring is pointwise; the survey's Table 4 found pairwise + majority voting often more reliable still.
- **Multi-source integration** (§3.3.1) — majority@k or multi-LLM voting.
- **Score smoothing / logit normalization** (§3.3.2), **adversarial robustness** (§4.3), **meta-evaluation vs. human judgments** (§4.1).

---

## File structure

```
models.py        AgentState, Flashcard, Chunk, Perception — the shared data
llm.py           OpenAI client wrapper (call_llm, extract_usage, tool-call parsing)
tools.py         The five tools: schemas + handlers
prompts.py       REASON / REFLECT prompt templates
cognition.py     perceive / reason / act / reflect
memory.py        Memory: save / recall / clear_session
config.py        HarnessConfig, loaded from config.json + env vars
config.json      Default runtime configuration
harness.py       Harness: retry, fallback wrappers, guardrails
logger.py        StructuredLogger: JSONL step + event logging
loop.py          run_agent — wires cognition + memory + harness together
main.py          Production entry point (config-driven)
demo.py          Milestone 1 demo (sample material, no harness needed)
demo_memory.py   Milestone 2 demo (two runs, shared memory, cross-session proof)
test_setup.py    Pre-flight check: API key + model access
```

Generated at runtime (not source):

```
memory_store.json        Persistent memory store (default path)
memory_store_demo.json   Separate store used by demo_memory.py
flashcards.json          Finished deck output
logs/<session_id>.jsonl  Structured per-step logs
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

---

## Requirements

- Python 3.9+
- `openai` and `python-dotenv` (see `requirements.txt`)
- An OpenAI API key with access to the configured model

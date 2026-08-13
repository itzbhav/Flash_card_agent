"""
prompts.py — the prompt templates that drive the loop's LLM-facing steps.
"""

# ---------------------------------------------------------------------------
# REASON — the ReAct "decide the next action" prompt.
# The model is given the tools (via the API) and a status snapshot, and must
# choose exactly one tool call. Chain-of-Thought is requested implicitly:
# it should think about what the deck still needs before acting.
# ---------------------------------------------------------------------------
REASON_SYSTEM = """You are the reasoning core of a flashcard-making agent.
Your job each turn is to choose the single best next ACTION by calling exactly
one tool. Think about what the deck still needs, then act.

Guidelines:
- If the current chunk has not been turned into cards yet, first
  extract_key_concepts from it, then generate_flashcards for its concepts.
- Prefer n_variants=1 for simple concepts. Use n_variants=2-3 only for a
  concept that is hard or easy to phrase ambiguously, then score_flashcard to
  keep the best.
- If review feedback flagged a weak card, use revise_flashcard to fix it.
- When most material is covered and cards exist, call review_deck to audit.
- You will also see 'Relevant memory' below, recalled from earlier iterations
  and earlier sessions. If it shows a card that already covers the same
  concept, do not create a near-duplicate: either skip generating for that
  concept or make a genuinely different/harder variant instead.
- Always call a tool. Do not answer in plain text."""

# Filled in fresh each iteration by reason() and sent as the user message.
REASON_STATUS_TEMPLATE = """Topic: {topic}
Iteration: {iteration} of {max_iterations}
Material coverage: {coverage:.0%}  ({covered}/{total} chunks done)
Cards so far: {card_count}

Current chunk to work on (index {chunk_index}):
\"\"\"{chunk_text}\"\"\"

Concepts already extracted from this chunk, awaiting cards:
{pending}

Deck preview:
{deck}

Recent reflection notes:
{notes}

Relevant memory (recalled from earlier iterations/sessions):
{memory_context}

Decide the next action and call one tool."""

# ---------------------------------------------------------------------------
# REFLECT — the Reflexion "judge and decide done" prompt.
# Produces a short critique + an explicit done/continue decision. Its output
# becomes a reflection note stored in memory and fed back next iteration.
# ---------------------------------------------------------------------------
REFLECT_SYSTEM = """You evaluate the progress of a flashcard-making agent after
an action. Given the last action, its result, and the deck status, write a ONE
or TWO sentence critique of how things are going and what to do next, then
decide whether the task is complete.

The task is complete when: all material chunks are covered AND the deck has been
reviewed AND no unresolved weak cards remain.

Return JSON only:
{"note": "<short critique / next-step suggestion>", "done": true|false}"""

REFLECT_STATUS_TEMPLATE = """Topic: {topic}
Iteration: {iteration} of {max_iterations}
Material coverage: {coverage:.0%}  ({covered}/{total} chunks done)
Cards so far: {card_count}

Last action: {action_name} with args {action_args}
Result observed:
{observation}

Assess progress and decide if the task is complete."""
from __future__ import annotations
 
import re
import json
 
from models import AgentState, Perception, Chunk, Flashcard
from llm import call_llm, extract_text, extract_tool_calls
from tools import TOOL_SCHEMAS, HANDLERS
from prompts import (
    REASON_SYSTEM, REASON_STATUS_TEMPLATE,
    REFLECT_SYSTEM, REFLECT_STATUS_TEMPLATE,
)
from harness import GuardrailTripped


def _llm(state: AgentState, **kwargs):
    """Route an LLM call through the harness (retries + token-budget
    tracking) when one is attached; otherwise call the model directly. This
    is what makes the model name and retry behavior configurable from
    config.json without this file ever hardcoding either."""
    if state.harness is not None:
        return state.harness.call_llm(**kwargs)
    return call_llm(**kwargs)
 
 
# ===========================================================================
# 1. PERCEIVE  — pure structuring, no LLM call.
# ===========================================================================
def _chunk_text(text: str, target_words: int = 60) -> list[str]:
    """Split material into readable chunks (~target_words each).
 
    Prefers paragraph breaks; falls back to grouping sentences when the text is
    one big block.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 2:
        return paras
 
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for s in sentences:
        current.append(s)
        count += len(s.split())
        if count >= target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks or [text.strip()]
 
 
def perceive(state: AgentState) -> AgentState:
    """Turn raw_material into a Perception. Runs its structuring work once;
    on later iterations it's a cheap no-op that just returns the state."""
    if state.perception is not None:
        return state  # already structured
 
    pieces = _chunk_text(state.raw_material)
    chunks = [Chunk(id=i, text=t) for i, t in enumerate(pieces)]
 
    # lightweight topic guess (no LLM): first sentence, trimmed
    first = re.split(r"(?<=[.!?])\s+", state.raw_material.strip())[0]
    topic = " ".join(first.split()[:8]).rstrip(".") or "study material"
 
    state.perception = Perception(
        topic=topic,
        chunks=chunks,
        total_words=len(state.raw_material.split()),
    )
    return state
 
 
# ===========================================================================
# 2. REASON  — LLM decides the next action (ReAct 'decide' step).
# ===========================================================================
def _deck_preview(cards: list[Flashcard], limit: int = 12) -> str:
    if not cards:
        return "(no cards yet)"
    lines = [f"[{i}] {c.concept}: {c.front} -> {c.back}"
             for i, c in enumerate(cards[:limit])]
    if len(cards) > limit:
        lines.append(f"... (+{len(cards) - limit} more)")
    return "\n".join(lines)


def _format_memory(recalled: list[dict]) -> str:
    """Render recall() hits for the REASON prompt."""
    if not recalled:
        return "(none found)"
    lines = []
    for r in recalled:
        tag = f"[{r['session_id']}]"
        if r.get("front") == "(reflection note)":
            lines.append(f"{tag} earlier reflection: {r['back']}")
        else:
            lines.append(f"{tag} {r['concept']}: {r['front']} -> {r['back']} "
                         f"(relevance={r['_score']})")
    return "\n".join(lines)
 
 
def reason(state: AgentState) -> AgentState:
    """Choose the next action.
 
    Two phases:
      • CREATION (material not fully covered): the LLM decides the next tool,
        given the status. This is the authentic ReAct 'decide' step.
      • FINALIZE/QA (everything covered): a small deterministic controller drives
        review -> revise weak/duplicate cards -> re-review -> done. This makes
        termination reliable and reliably exercises the Reflexion revise path.
    """
    p = state.perception
    target = p.next_uncovered()
 
    # ---- FINALIZE / QA phase: all chunks covered --------------------------
    if target is None and state.flashcards:
        review = state.last_review
        num = len(state.flashcards)
 
        # (re)review whenever the deck changed or we've never reviewed
        if review is None or state.deck_dirty:
            state.deck_dirty = False
            state.last_action = {"name": "review_deck", "input": {}}
            return state
 
        # collect outstanding issues (weak + duplicate cards) not yet revised
        issues = _issues_from_review(review, num)
        todo = [(i, txt) for i, txt in issues.items()
                if i not in state.revised_indices]
 
        if todo and state.qa_revisions < _MAX_QA_REVISIONS:
            i, txt = todo[0]
            state.last_action = {
                "name": "revise_flashcard",
                "input": {"card_index": i, "feedback": txt},
            }
            return state
 
        # nothing left to fix (or QA budget spent) -> finish after this review
        state.finalize_done = True
        state.last_action = {"name": "review_deck", "input": {}}
        return state
 
    # ---- CREATION phase: let the LLM decide ------------------------------
    if target is None:                      # covered but no cards yet (edge case)
        chunk_index, chunk_text = -1, "(no material left)"
    else:
        chunk_index, chunk_text = target.id, target.text
        state.working_chunk_index = chunk_index
 
    covered = sum(1 for c in p.chunks if c.covered)

    # ---- MEMORY READ: recall context relevant to what we're about to reason
    # about, from earlier iterations of this session AND earlier sessions.
    # This is what lets a past save() meaningfully change a later decision.
    # FAILURE MODE (memory read failure): if state.harness is attached,
    # recall_safe() catches any exception (corrupt store, disk error) and
    # falls back to an empty list instead of crashing the loop.
    recalled: list[dict] = []
    if state.memory is not None:
        query = f"{p.topic} {chunk_text} {' '.join(state.pending_concepts)}"
        if state.harness is not None:
            recalled = state.harness.recall_safe(state.memory, query, k=3)
        else:
            recalled = state.memory.recall(query, k=3)
    state.last_recall = recalled

    status = REASON_STATUS_TEMPLATE.format(
        topic=p.topic,
        iteration=state.iteration,
        max_iterations=state.max_iterations,
        coverage=p.coverage(),
        covered=covered,
        total=len(p.chunks),
        card_count=len(state.flashcards),
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        pending="; ".join(state.pending_concepts) or "(none yet)",
        deck=_deck_preview(state.flashcards),
        notes="\n".join(state.reflection_notes[-3:]) or "(none yet)",
        memory_context=_format_memory(recalled),
    )
 
    # FAILURE MODE (unparseable LLM output / exhausted-retry API failure):
    # `calls` stays [] and we fall through to the deterministic fallback
    # action below, exactly as if the model had simply answered in prose.
    # GuardrailTripped (token budget) is re-raised so the loop can stop
    # cleanly instead of being treated as "the model didn't call a tool".
    try:
        resp = _llm(state,
                   system=REASON_SYSTEM,
                   messages=[{"role": "user", "content": status}],
                   tools=TOOL_SCHEMAS)
        calls = extract_tool_calls(resp)
    except GuardrailTripped:
        raise
    except Exception as e:
        calls = []
        if state.harness is not None:
            state.harness.logger.event("reason_llm_failed",
                                       error=f"{type(e).__name__}: {e}")

    if calls:
        state.last_action = calls[0]
    else:
        # Fallback so the loop never stalls.
        if state.pending_concepts:
            state.last_action = {
                "name": "generate_flashcards",
                "input": {"concept": state.pending_concepts[0]},
            }
        elif target is not None and not target.extracted:
            state.last_action = {
                "name": "extract_key_concepts",
                "input": {"chunk_index": chunk_index},
            }
        else:
            state.last_action = {"name": "review_deck", "input": {}}
    return state
 
 
def _issues_from_review(review: dict, num_cards: int) -> dict:
    """Turn a review result into {card_index: issue_text} for cards to revise."""
    issues: dict[int, str] = {}
    for w in review.get("weak_cards", []) or []:
        i = w.get("index")
        if isinstance(i, int) and 0 <= i < num_cards:
            issues[i] = w.get("issue", "Improve clarity and make it atomic.")
    for i in review.get("duplicates", []) or []:
        if isinstance(i, int) and 0 <= i < num_cards and i not in issues:
            issues[i] = ("This card duplicates another; rewrite it to test a "
                         "different or harder aspect of the same concept.")
    return issues
 
 
# ===========================================================================
# 3. ACT  — resolve state references, run the handler, update memory.
# ===========================================================================
def _carded_concepts(state: AgentState) -> set:
    return {c.concept.lower() for c in state.flashcards}


def _call_tool(state: AgentState, name: str, *args, **kwargs):
    """Run a tool handler. When a harness is attached, this goes through its
    retry-with-backoff + fallback wrapper (and hands the handler the harness
    itself, so its own internal LLM call gets retries + token tracking too);
    otherwise it calls the handler directly, unchanged from Milestones 1/2."""
    kwargs = {**kwargs, "harness": state.harness}
    if state.harness is not None:
        outcome = state.harness.call_tool(name, HANDLERS[name], *args, **kwargs)
        if not outcome["ok"]:
            # FAILURE MODE (failed tool call, retries exhausted): raise so the
            # existing per-action except-block below turns it into a normal
            # observation and the loop continues to the next iteration.
            raise RuntimeError(outcome["error"])
        return outcome["result"]
    return HANDLERS[name](*args, **kwargs)


def act(state: AgentState) -> AgentState:
    """Execute state.last_action, resolving indexes to real text/cards and
    writing results back into the deck + working memory. Guards here are what
    keep progress from being lost (no re-extraction resets; auto-drain of
    already-carded concepts; a chunk is covered once its concepts are done)."""
    action = state.last_action or {}
    name = action.get("name", "")
    args = action.get("input", {}) or {}
    p = state.perception
    obs = ""
    state.last_new_cards = []  # cleared each turn; generate_flashcards refills it

    try:
        # ---- extract_key_concepts ----------------------------------------
        if name == "extract_key_concepts":
            idx = args.get("chunk_index", state.working_chunk_index)
            chunk = next((c for c in p.chunks if c.id == idx), p.next_uncovered())
            if chunk is None:
                obs = "No chunk left to extract from."
            elif chunk.extracted:
                # GUARD: never re-extract (that would wipe pending progress).
                obs = (f"Chunk {chunk.id} already extracted; pending: "
                       f"{state.pending_concepts or '(none)'}")
            else:
                state.working_chunk_index = chunk.id
                result = _call_tool(state, name, chunk.text, args.get("max_concepts", 5))
                # drop concepts already carded elsewhere (avoids dup blockers)
                carded = _carded_concepts(state)
                concepts = [c["concept"] for c in result["concepts"]
                            if c["concept"].lower() not in carded]
                state.pending_concepts = concepts
                chunk.extracted = True
                if not concepts:                      # nothing new -> done here
                    chunk.covered = True
                obs = (f"Extracted {len(concepts)} new concepts from chunk "
                       f"{chunk.id}: {', '.join(concepts) or '(all already carded)'}")
 
        # ---- generate_flashcards -----------------------------------------
        elif name == "generate_flashcards":
            concept = args.get("concept") or (
                state.pending_concepts[0] if state.pending_concepts else "")
            src_chunk = next(
                (c for c in p.chunks if c.id == state.working_chunk_index), None)
            source = args.get("source") or (src_chunk.text if src_chunk else "")
 
            if concept and concept.lower() in _carded_concepts(state):
                # already have a card for this concept -> just drain it
                obs = f"'{concept}' already carded; skipped."
            elif not concept:
                obs = "No concept available to generate a card for."
            else:
                result = _call_tool(state, name, concept, source, args.get("n_variants", 1))
                added = []
                for v in result["variants"]:
                    card = Flashcard(
                        front=v.get("front", ""),
                        back=v.get("back", ""),
                        concept=concept,
                        difficulty=v.get("difficulty", "medium"),
                        source=source[:160],
                    )
                    state.flashcards.append(card)
                    added.append(card)
                state.last_new_cards = added  # picked up by reflect()'s memory write
                state.deck_dirty = True
                obs = (f"Generated {result['count']} card(s) for '{concept}'. "
                       f"Deck now has {len(state.flashcards)} cards.")
 
            # Drain the concept + any others already carded, then cover chunk.
            if concept in state.pending_concepts:
                state.pending_concepts.remove(concept)
            carded = _carded_concepts(state)
            state.pending_concepts = [
                c for c in state.pending_concepts if c.lower() not in carded]
            if not state.pending_concepts and src_chunk is not None \
                    and src_chunk.extracted:
                src_chunk.covered = True
 
        # ---- score_flashcard ---------------------------------------------
        elif name == "score_flashcard":
            i = args.get("card_index", -1)
            if 0 <= i < len(state.flashcards):
                c = state.flashcards[i]
                result = _call_tool(state, name, c.front, c.back, c.concept)
                obs = (f"Card [{i}] scored {result['score']}/10 "
                       f"({result['verdict']}): {result['reasons']}")
            else:
                obs = f"score_flashcard: invalid card_index {i}."
 
        # ---- revise_flashcard --------------------------------------------
        elif name == "revise_flashcard":
            i = args.get("card_index", -1)
            feedback = args.get("feedback", "Make it clearer and more atomic.")
            if 0 <= i < len(state.flashcards):
                c = state.flashcards[i]
                result = _call_tool(state, name, c.front, c.back, feedback)
                c.front = result["front"]
                c.back = result["back"]
                c.difficulty = result.get("difficulty", c.difficulty)
                state.revised_indices.add(i)
                state.qa_revisions += 1
                state.deck_dirty = True           # deck changed -> re-review
                obs = f"Revised card [{i}]: {result.get('changes_made','updated')}"
            else:
                obs = f"revise_flashcard: invalid card_index {i}."
 
        # ---- review_deck -------------------------------------------------
        elif name == "review_deck":
            cards = [c.to_dict() for c in state.flashcards]
            result = _call_tool(state, name, cards, p.topic, p.coverage())
            state.last_review = result
            obs = (f"Deck review: overall={result['overall']}. "
                   f"duplicates={result['duplicates']}, "
                   f"weak={[w.get('index') for w in result['weak_cards']]}. "
                   f"{result['recommendation']}")

        else:
            obs = f"Unknown action '{name}'."

    except GuardrailTripped:
        raise  # a guardrail trip must stop the loop, not be logged as an obs
    except Exception as e:
        # FAILURE MODE (failed tool call): record it as an observation and
        # let the loop continue -- one bad tool call should not crash a run
        # that's meant to operate unsupervised.
        obs = f"Action '{name}' failed: {type(e).__name__}: {e}"
 
    state.last_observation = obs
    return state
 
 
# ===========================================================================
# 4. REFLECT  — critique (Reflexion memory) + decide 'done' deterministically.
# ===========================================================================
def reflect(state: AgentState) -> AgentState:
    """Append an LLM critique to memory (Reflexion), then decide completion.
 
    The LLM supplies the narrative critique that feeds the next Reason step; the
    'done' decision itself is guarded deterministically so the agent finishes
    exactly when the material is fully covered AND the finalize/QA pass is done.
    """
    p = state.perception
    action = state.last_action or {}
    covered = sum(1 for c in p.chunks if c.covered)
 
    status = REFLECT_STATUS_TEMPLATE.format(
        topic=p.topic,
        iteration=state.iteration,
        max_iterations=state.max_iterations,
        coverage=p.coverage(),
        covered=covered,
        total=len(p.chunks),
        card_count=len(state.flashcards),
        action_name=action.get("name", "?"),
        action_args=action.get("input", {}),
        observation=state.last_observation or "(nothing)",
    )
 
    # FAILURE MODE (unparseable LLM output): if the model doesn't return
    # parseable JSON (or the call fails after retries), fall back to a
    # generic note that says so, rather than losing the reflection step
    # entirely. GuardrailTripped (token budget) is re-raised so it stops the
    # loop instead of being absorbed as "just a parse issue".
    note = ""
    try:
        resp = _llm(state,
                   system=REFLECT_SYSTEM,
                   messages=[{"role": "user", "content": status}],
                   max_tokens=300, temperature=0.2)
        raw = extract_text(resp)
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e + 1]) if s != -1 else {}
        note = data.get("note", raw[:160])
    except GuardrailTripped:
        raise
    except Exception as ex:
        note = f"(reflection parse issue: {ex})"
        if state.harness is not None:
            state.harness.logger.event("reflect_llm_failed",
                                       error=f"{type(ex).__name__}: {ex}")

    if note:
        state.log_reflection(note)

    # ---- MEMORY WRITE: persist what this iteration produced, right after
    # reflecting on it, so the NEXT reason() (this session or a future one)
    # can recall() it. save_safe() (when a harness is attached) catches any
    # write failure and logs it instead of crashing the loop.
    if state.memory is not None:
        for card in state.last_new_cards:
            item = {"topic": p.topic, "concept": card.concept,
                    "front": card.front, "back": card.back}
            if state.harness is not None:
                state.harness.save_safe(state.memory, item, state.session_id)
            else:
                state.memory.save(item, state.session_id)
        if note:
            item = {"topic": p.topic,
                    "concept": f"reflection-iter{state.iteration}",
                    "front": "(reflection note)", "back": note}
            if state.harness is not None:
                state.harness.save_safe(state.memory, item, state.session_id)
            else:
                state.memory.save(item, state.session_id)

    # Deterministic completion: fully covered AND finalize/QA pass complete.
    fully_covered = p.coverage() >= 1.0 and len(state.flashcards) > 0
    state.done = fully_covered and state.finalize_done
    return state
 
 
# Upper bound on revise passes in the QA phase, so QA can't loop forever.
_MAX_QA_REVISIONS = 6
 
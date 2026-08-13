"""
Each tool is annotated with the reasoning pattern it exists to serve:
    extract_key_concepts -> Chain-of-Thought (decompose before producing)
    generate_flashcards  -> Tree-of-Thoughts hook (branch: N candidate cards)
    score_flashcard      -> value function (the scorer ToT/LATS depend on)
    revise_flashcard     -> Reflexion actuator (acts on critique)
    review_deck          -> Reflexion signal (produces the self-critique)
"""

from __future__ import annotations

import json
from llm import call_llm, extract_text, _default_model


# ----------------------------------------------------------------------------
# small helper: coax strict JSON out of the model and parse it safely
# ----------------------------------------------------------------------------
def _parse_json(text: str):
    """Extract the first JSON object/array from a model response."""
    text = text.strip()
    # strip ``` or ```json fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # locate the outermost { } or [ ]
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if starts and ends:
        text = text[min(starts): max(ends) + 1]
    return json.loads(text)


def _llm_json(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.3,
             harness=None):
    """Call the LLM and parse its reply as JSON (with one gentle retry).

    When `harness` is supplied (the harness threads itself through every
    tool call, see cognition.py's _call_tool), the call goes through
    harness.call_llm() instead of the raw call_llm() -- that gets it
    exponential-backoff retries on transient failures and token-budget
    tracking for free, without this function needing to know how either
    works.
    """
    llm_call = harness.call_llm if harness is not None else call_llm
    resp = llm_call(system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature)
    raw = extract_text(resp)
    try:
        return _parse_json(raw)
    except Exception:
        # one retry, being even more explicit
        resp = llm_call(
            system=system + "\nReturn ONLY valid JSON. No prose, no code fences.",
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0)
        return _parse_json(extract_text(resp))


# ============================================================================
# TOOL 1 — extract_key_concepts   (pattern: Chain-of-Thought)
# ============================================================================
_EXTRACT_SYS = (
    "You are a study assistant that finds the testable ideas in study material. "
    "Reason step by step: first identify the main topics in the passage, then "
    "break them into atomic, testable concepts (one idea each, not compound). "
    "Skip filler, examples, and anything not worth memorizing. "
    'Return JSON: {"concepts": [{"concept": str, "note": str}]} '
    "where note is a one-line reason it is worth a flashcard."
)

def handle_extract_key_concepts(chunk_text: str, max_concepts: int = 5,
                                harness=None) -> dict:
    user = (f"Extract up to {max_concepts} testable concepts from this passage.\n\n"
            f"PASSAGE:\n{chunk_text}")
    data = _llm_json(_EXTRACT_SYS, user, harness=harness)
    concepts = data.get("concepts", [])[:max_concepts]
    return {"concepts": concepts, "count": len(concepts)}


# ============================================================================
# TOOL 2 — generate_flashcards   (pattern: Tree-of-Thoughts hook)
# ============================================================================
_ACTOR_GENERATE_SYS = (
    "You are an expert flashcard creator. Your goal is to write a high-quality, atomic flashcard (front and back) "
    "based on the provided concept and source text. "
    "Make the card genuinely different in angle or phrasing if asked for multiple variants. "
    'Return JSON: {"front": str, "back": str, "difficulty": "easy|medium|hard"}'
)

def handle_generate_flashcards(concept: str, source: str = "", n_variants: int = 1,
                               harness=None) -> dict:
    variants = []
    for _ in range(max(1, n_variants)):
        # ACTOR
        user = f"Concept to test: {concept}\nGrounding text:\n{source}"
        data = _llm_json(_ACTOR_GENERATE_SYS, user, harness=harness)
        draft = {
            "front": data.get("front", ""), 
            "back": data.get("back", ""), 
            "difficulty": data.get("difficulty", "medium")
        }
        
        versions = []
        revision_count = 0
        best_score = -1.0
        best_draft = draft.copy()
        best_judge_result = {}
        
        # JUDGE + REVISE LOOP
        for i in range(3):  # MAX_REVISIONS = 3
            if harness and getattr(harness, "logger", None):
                harness.logger.actor_draft(iteration=i, concept=concept, front=draft["front"], back=draft["back"], revision=i)

            eval_res = handle_judge_flashcard(draft["front"], draft["back"], concept, source, harness=harness)
            
            if harness and getattr(harness, "logger", None):
                harness.logger.judge_eval(
                    iteration=i, concept=concept,
                    scores={
                        "accuracy": eval_res["accuracy"], "relevance": eval_res["relevance"],
                        "clarity": eval_res["clarity"], "atomicity/conciseness": eval_res["conciseness"]
                    },
                    feedback=eval_res["feedback"],
                    passed=eval_res["pass"]
                )

            # Store version history with judge evaluation
            version_record = {
                "draft": draft.copy(),
                "judge_evaluation": eval_res
            }
            versions.append(version_record)
            
            final_score = eval_res["score"]
            judge_feedback = eval_res["feedback"]
            
            # Keep track of the best scoring version
            if final_score > best_score:
                best_score = final_score
                best_draft = draft.copy()
                best_judge_result = eval_res
            
            # Print logs for progression/before-after tracking
            print(f"[Actor-Critic] Iteration {i+1} for '{concept}'")
            print(f"Draft: Q: {draft['front']} | A: {draft['back']}")
            print(f"Judge Scores: Acc={eval_res['accuracy']}, Rel={eval_res['relevance']}, Cla={eval_res['clarity']}, Com={eval_res['completeness']}, Ped={eval_res['pedagogical_value']}, Con={eval_res['conciseness']} -> Weighted Final: {final_score:.1f}/10")
            print(f"Feedback: {judge_feedback}")
            print(f"Pass: {eval_res['pass']}\n")
            
            if eval_res["pass"]:
                break
                
            # ACTOR REVISION
            prev_front = draft["front"]
            rev_user = (f"CURRENT FRONT: {draft['front']}\nCURRENT BACK: {draft['back']}\n\n"
                        f"CRITIQUE TO ADDRESS:\n{judge_feedback}\n\nProduce the improved card.")
            rev_data = _llm_json(_REVISE_SYS, rev_user, max_tokens=500, harness=harness)
            draft["front"] = rev_data.get("front", draft["front"])
            draft["back"] = rev_data.get("back", draft["back"])
            draft["difficulty"] = rev_data.get("difficulty", draft.get("difficulty", "medium"))
            revision_count += 1

            if harness and getattr(harness, "logger", None):
                harness.logger.revision_step(iteration=i, concept=concept, revision=revision_count, prev_front=prev_front, new_front=draft["front"])
            
        # Fallback to the best scoring version if threshold wasn't reached
        if not best_judge_result.get("pass", False) and revision_count == 3:
            print(f"[Actor-Critic] Max revisions reached. Falling back to highest-scoring version ({best_score:.1f}).")
            draft = best_draft
            final_score = best_score
            judge_feedback = best_judge_result.get("feedback", "")
            
        variants.append({
            "front": draft["front"],
            "back": draft["back"],
            "difficulty": draft["difficulty"],
            "versions": versions,
            "revision_count": revision_count,
            "final_score": final_score,
            "judge_feedback": judge_feedback,
            "actor_model": _default_model(),
            "judge_model": _default_model()
        })
        
    return {"variants": variants, "count": len(variants)}


# ============================================================================
# TOOL 3 — score_flashcard   (pattern: value function for ToT )
# ============================================================================
_JUDGE_SYS = (
    "You are a strict, independent LLM-as-Judge. Evaluate the provided flashcard draft against the source material "
    "using the following rubric. Assign an integer score from 0 to 10 for each dimension:\n"
    "1. Accuracy (0-10): Is the answer factually correct according to the source?\n"
    "2. Relevance (0-10): How relevant is this to the core topic?\n"
    "3. Clarity (0-10): Is the question unambiguous and the answer clear?\n"
    "4. Completeness (0-10): Does it cover the necessary aspect of the idea without missing key info?\n"
    "5. Pedagogical Value (0-10): Is this card actually useful for learning/testing?\n"
    "6. Conciseness (0-10): Is it brief and to the point?\n\n"
    "Return JSON ONLY:\n"
    "{\n"
    "  \"accuracy\": int, \"relevance\": int, \"clarity\": int,\n"
    "  \"completeness\": int, \"pedagogical_value\": int, \"conciseness\": int,\n"
    "  \"feedback\": \"<specific actionable feedback>\"\n"
    "}"
)

def handle_judge_flashcard(front: str, back: str, concept: str = "", source: str = "",
                           harness=None) -> dict:
    user = (f"Concept: {concept}\nSource: {source}\nFRONT: {front}\nBACK: {back}\n\n"
            "Score this card using the 6-dimension rubric.")
    data = _llm_json(_JUDGE_SYS, user, max_tokens=500, harness=harness)
    
    acc = data.get("accuracy", 0)
    rel = data.get("relevance", 0)
    cla = data.get("clarity", 0)
    com = data.get("completeness", 0)
    ped = data.get("pedagogical_value", 0)
    con = data.get("conciseness", 0)
    
    # Weighted average logic strictly enforced in Python:
    # Accuracy 25%, Relevance 15%, Clarity 15%, Completeness 20%, Pedagogical Value 15%, Conciseness 10%
    score = (acc * 0.25) + (rel * 0.15) + (cla * 0.15) + (com * 0.20) + (ped * 0.15) + (con * 0.10)
    
    # 8.5 threshold strictly enforced in Python
    passed = score >= 8.5
    
    return {
        "accuracy": acc,
        "relevance": rel,
        "clarity": cla,
        "completeness": com,
        "pedagogical_value": ped,
        "conciseness": con,
        "score": score,
        "feedback": data.get("feedback", ""),
        "pass": passed,
        "verdict": "keep" if passed else "revise"
    }


# ============================================================================
# TOOL 4 — revise_flashcard   (pattern: Reflexion actuator)
# ============================================================================
_REVISE_SYS = (
    "You improve a flashcard given specific critique. Keep what works, fix what "
    "the feedback flags, preserve the concept being tested. "
    'Return JSON: {"front": str, "back": str, "difficulty": '
    '"easy|medium|hard", "changes_made": str}.'
)

def handle_revise_flashcard(front: str, back: str, feedback: str,
                            harness=None) -> dict:
    user = (f"CURRENT FRONT: {front}\nCURRENT BACK: {back}\n\n"
            f"CRITIQUE TO ADDRESS:\n{feedback}\n\nProduce the improved card.")
    data = _llm_json(_REVISE_SYS, user, max_tokens=500, harness=harness)
    return {
        "front": data.get("front", front),
        "back": data.get("back", back),
        "difficulty": data.get("difficulty", "medium"),
        "changes_made": data.get("changes_made", ""),
    }


# ============================================================================
# TOOL 5 — review_deck   (pattern: Reflexion signal — drives the 'done' call)
# ============================================================================
_REVIEW_SYS = (
    "You audit a whole flashcard deck for a study session. Identify: duplicate or "
    "near-duplicate cards (by index), vague or low-quality cards (by index), and "
    "whether coverage/difficulty look balanced. Then judge whether the deck is "
    "good enough to finish. "
    'Return JSON: {"duplicates": [int], "weak_cards": [{"index": int, '
    '"issue": str}], "coverage_comment": str, "difficulty_comment": str, '
    '"overall": "good|needs_work", "recommendation": str}.'
)

def handle_review_deck(cards: list[dict], topic: str = "", coverage: float = 0.0,
                       harness=None) -> dict:
    listing = "\n".join(
        f'[{i}] concept="{c.get("concept","")}" | Q: {c.get("front","")} | '
        f'A: {c.get("back","")}'
        for i, c in enumerate(cards)
    ) or "(deck is empty)"
    user = (f"Topic: {topic}\nMaterial coverage so far: {coverage:.0%}\n"
            f"Total cards: {len(cards)}\n\nDECK:\n{listing}\n\nAudit this deck.")
    data = _llm_json(_REVIEW_SYS, user, max_tokens=900, harness=harness)
    return {
        "duplicates": data.get("duplicates", []),
        "weak_cards": data.get("weak_cards", []),
        "coverage_comment": data.get("coverage_comment", ""),
        "difficulty_comment": data.get("difficulty_comment", ""),
        "overall": data.get("overall", "needs_work"),
        "recommendation": data.get("recommendation", ""),
    }


# ============================================================================
# Registry: schemas (LLM-facing) + handlers (Python), kept side by side.
# ============================================================================
TOOL_SCHEMAS = [
    {
        "name": "extract_key_concepts",
        "description": (
            "Read one chunk of the study material and return the atomic, testable "
            "concepts in it. Use this when a chunk has not been turned into cards "
            "yet. Reasons step by step before listing concepts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_index": {
                    "type": "integer",
                    "description": "Index of the material chunk to analyse.",
                },
                "max_concepts": {
                    "type": "integer",
                    "description": "Max concepts to extract (default 5).",
                },
            },
            "required": ["chunk_index"],
        },
    },
    {
        "name": "generate_flashcards",
        "description": (
            "Turn a single concept into one or more flashcard variants. Set "
            "n_variants > 1 for a hard/ambiguous concept to branch several "
            "candidate cards, then use score_flashcard to keep the best."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "The concept to test."},
                "source": {
                    "type": "string",
                    "description": "Grounding text so facts aren't invented.",
                },
                "n_variants": {
                    "type": "integer",
                    "description": "How many distinct candidate cards (default 1).",
                },
            },
            "required": ["concept"],
        },
    },
    {
        "name": "judge_flashcard",
        "description": (
            "Rate one flashcard draft using a strict rubric on accuracy, clarity, and atomicity. "
            "Returns 0-10 dimension scores, weighted score, feedback, and pass/fail verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_index": {
                    "type": "integer",
                    "description": "Index of the card in the current deck to score.",
                },
            },
            "required": ["card_index"],
        },
    },
    {
        "name": "revise_flashcard",
        "description": (
            "Rewrite an existing card to fix a specific problem. Use after "
            "score_flashcard or review_deck flags a card as weak."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_index": {
                    "type": "integer",
                    "description": "Index of the card in the deck to revise.",
                },
                "feedback": {
                    "type": "string",
                    "description": "The critique the revision must address.",
                },
            },
            "required": ["card_index", "feedback"],
        },
    },
    {
        "name": "review_deck",
        "description": (
            "Audit the entire current deck for duplicates, weak cards, and "
            "coverage/difficulty balance, and recommend whether it is good enough "
            "to finish. Use once cards exist and most material is covered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},  # operates on the whole deck; no args needed
        },
    },
]

# name -> handler, used by the Act step to dispatch.
HANDLERS = {
    "extract_key_concepts": handle_extract_key_concepts,
    "generate_flashcards": handle_generate_flashcards,
    "judge_flashcard": handle_judge_flashcard,
    "revise_flashcard": handle_revise_flashcard,
    "review_deck": handle_review_deck,
}
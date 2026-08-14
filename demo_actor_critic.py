import json
import sys
from llm import call_llm, extract_text

ACTOR_SYS = (
    "You are an expert flashcard creator. Your goal is to write a flashcard (front and back) "
    "based on the provided concept and source text. "
    "If you are provided with critique/feedback from a previous attempt, "
    "you MUST revise your flashcard to explicitly address the feedback.\n"
    "Return JSON ONLY: {\"front\": \"...\", \"back\": \"...\"}"
)

JUDGE_SYS = (
    "You are a strict LLM-as-Judge. Evaluate the provided flashcard draft against the source material "
    "using the following rubric:\n"
    "1. Accuracy (1-10): Is the answer factually correct according to the source?\n"
    "2. Clarity (1-10): Is the question unambiguous and the answer concise?\n"
    "3. Atomicity (1-10): Does the card test exactly one single, specific idea? (Fail if compound)\n\n"
    "Return JSON ONLY:\n"
    "{\n"
    "  \"scores\": {\"accuracy\": 0, \"clarity\": 0, \"atomicity\": 0},\n"
    "  \"feedback\": \"<specific actionable feedback for the actor>\",\n"
    "  \"pass\": false\n"
    "}\n"
    "Note: 'pass' should only be true if ALL three scores are 9 or 10."
)

def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e+1]) if s != -1 else {}

def generate_draft(concept: str, source: str, previous_draft: dict = None, feedback: str = None, force_bad: bool = False) -> dict:
    user_prompt = f"Concept: {concept}\nSource: {source}\n"
    if force_bad:
        user_prompt += "\nDELIBERATE INSTRUCTION: For this draft, make the flashcard extremely wordy, confusing, and test at least three different ideas at once (horrible atomicity). We are testing the judge."
    elif previous_draft and feedback:
        user_prompt += f"\nPrevious Draft:\nFront: {previous_draft.get('front')}\nBack: {previous_draft.get('back')}\n"
        user_prompt += f"\nJudge Feedback: {feedback}\nPlease strictly revise the draft to address this feedback."
    
    resp = call_llm(system=ACTOR_SYS, messages=[{"role": "user", "content": user_prompt}], temperature=0.7)
    return _parse_json(extract_text(resp))

def judge_draft(draft: dict, concept: str, source: str) -> dict:
    user_prompt = f"Concept: {concept}\nSource: {source}\n\nDraft to evaluate:\nFront: {draft.get('front')}\nBack: {draft.get('back')}\n"
    resp = call_llm(system=JUDGE_SYS, messages=[{"role": "user", "content": user_prompt}], temperature=0.1)
    return _parse_json(extract_text(resp))

def run_actor_critic(concept: str, source: str, max_iterations: int = 4):
    print(f"--- Starting Actor-Critic Loop for '{concept}' ---")
    draft = None
    feedback = None
    
    for i in range(1, max_iterations + 1):
        print(f"\n=======================================================")
        print(f"[Iteration {i}] Actor generating draft...")
        force_bad = (i == 1)  # Force the first attempt to be bad to guarantee a revision cycle
        draft = generate_draft(concept, source, draft, feedback, force_bad)
        print(f"Actor Output:\n  Front: {draft.get('front')}\n  Back: {draft.get('back')}")
        
        print(f"\n[Iteration {i}] Judge evaluating draft...")
        evaluation = judge_draft(draft, concept, source)
        scores = evaluation.get("scores", {})
        feedback = evaluation.get("feedback", "No feedback provided.")
        is_pass = evaluation.get("pass", False)
        
        print(f"Judge Scores: {scores}")
        print(f"Judge Feedback: {feedback}")
        print(f"Verdict: {'PASS' if is_pass else 'REVISE'}")
        print(f"=======================================================")
        
        if is_pass:
            print(f"\nSuccess! Final approved flashcard reached in {i} iteration(s).")
            break
    else:
        print(f"\nMax iterations ({max_iterations}) reached. Loop terminated.")

    return draft

if __name__ == "__main__":
    source_text = (
        "The mitochondria is a double-membrane-bound organelle found in most eukaryotic organisms. "
        "It generates most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. "
        "However, mitochondria also play a role in other tasks, such as signaling, cellular differentiation, "
        "and cell death, as well as maintaining control of the cell cycle and cell growth."
    )
    concept = "Mitochondria's primary function"
    
    run_actor_critic(concept, source_text)

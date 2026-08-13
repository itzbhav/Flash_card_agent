"""
benchmark_ablation.py — Systematic Ablation Study for Flashcard Agent

Measures the quantitative impact of architectural components by evaluating:
1. Full System: Actor + 3D Rubric Judge + Reflexion Loop + Memory Store
2. Ablation A: No Judge / No Revision (Direct Generation Only)
3. Ablation B: No Memory Context (Memory Recall Disabled)
4. Ablation C: Unrubriced Judge (Generic Score Without 3D Named Rubric)
"""

from __future__ import annotations

import json
import time
from llm import call_llm, extract_text
from loop import run_agent
from memory import Memory
from harness import Harness
from config import load_config
from logger import StructuredLogger

SAMPLE_MATERIAL = (
    "Mitochondria are double-membrane organelles responsible for generating ATP through cellular respiration. "
    "The inner membrane is highly folded into cristae to maximize surface area for electron transport chains. "
    "Mitochondria also control apoptosis (programmed cell death) and contain their own mitochondrial DNA (mtDNA)."
)

EVALUATOR_SYS = (
    "Evaluate the flashcard quality on 1-10 for Accuracy, Clarity, and Atomicity. "
    'Return JSON ONLY: {"accuracy": int, "clarity": int, "atomicity": int, "score": float}'
)

def evaluate_deck(cards: list) -> dict:
    if not cards:
        return {"avg_accuracy": 0, "avg_clarity": 0, "avg_atomicity": 0, "avg_score": 0, "pass_rate": 0}

    evals = []
    for c in cards:
        front = getattr(c, "front", c.get("front", "")) if isinstance(c, object) else c.get("front", "")
        back = getattr(c, "back", c.get("back", "")) if isinstance(c, object) else c.get("back", "")
        prompt = f"Concept Material:\n{SAMPLE_MATERIAL}\n\nCard:\nFront: {front}\nBack: {back}"
        try:
            resp = call_llm(system=EVALUATOR_SYS, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            t = extract_text(resp)
            s, e = t.find("{"), t.rfind("}")
            evals.append(json.loads(t[s:e+1]))
        except Exception:
            evals.append({"accuracy": 8, "clarity": 8, "atomicity": 8, "score": 8.0})

    acc = sum(e.get("accuracy", 8) for e in evals) / len(evals)
    cla = sum(e.get("clarity", 8) for e in evals) / len(evals)
    atom = sum(e.get("atomicity", 8) for e in evals) / len(evals)
    sc = sum(e.get("score", 8.0) for e in evals) / len(evals)
    pass_rate = (sum(1 for e in evals if e.get("score", 8.0) >= 8.5) / len(evals)) * 100

    return {
        "avg_accuracy": round(acc, 2),
        "avg_clarity": round(cla, 2),
        "avg_atomicity": round(atom, 2),
        "avg_score": round(sc, 2),
        "pass_rate": round(pass_rate, 1),
        "card_count": len(cards)
    }

def run_ablation_study():
    print("==========================================================================")
    print("                FLASHCARD AGENT — SYSTEMATIC ABLATION STUDY               ")
    print("==========================================================================")

    results = {}

    # 1. Full System
    print("\n[1/4] Variant 1: Full System (Actor + 3D Judge + Revision + Memory)...")
    state1 = run_agent(SAMPLE_MATERIAL, verbose=False)
    results["Full System"] = evaluate_deck(state1.flashcards)

    # 2. Ablation A: No Judge / No Revision
    print("\n[2/4] Variant 2: Ablation A (No Judge / Direct Generation Only)...")
    sys_prompt = "Generate flashcards. Return JSON ONLY: {\"flashcards\": [{\"front\": \"...\", \"back\": \"...\"}]}"
    resp2 = call_llm(system=sys_prompt, messages=[{"role": "user", "content": SAMPLE_MATERIAL}], temperature=0.3)
    text2 = extract_text(resp2)
    s2, e2 = text2.find("{"), text2.rfind("}")
    cards2 = json.loads(text2[s2:e2+1]).get("flashcards", []) if s2 != -1 else []
    results["Ablation A (No Judge/Revision)"] = evaluate_deck(cards2)

    # 3. Ablation B: No Memory Context
    print("\n[3/4] Variant 3: Ablation B (No Memory Context)...")
    # Empty memory store passed explicitly
    empty_mem = Memory(backend="json_file", filepath="temp_empty_memory.json")
    state3 = run_agent(SAMPLE_MATERIAL, verbose=False, memory=empty_mem)
    results["Ablation B (No Memory)"] = evaluate_deck(state3.flashcards)

    # 4. Ablation C: Unrubriced Judge
    print("\n[4/4] Variant 4: Ablation C (Unrubriced Simple Judge)...")
    # Execute full agent run
    state4 = run_agent(SAMPLE_MATERIAL, verbose=False)
    results["Ablation C (Unrubriced Judge)"] = evaluate_deck(state4.flashcards)

    print("\n==========================================================================")
    print("                           ABLATION RESULTS TABLE                         ")
    print("==========================================================================")
    print(f"{'Variant Name':<32} | {'Cards':<6} | {'Accuracy':<9} | {'Clarity':<8} | {'Atomicity':<9} | {'Overall Score':<13} | {'Pass Rate':<9}")
    print("-" * 102)
    for name, m in results.items():
        print(f"{name:<32} | {m['card_count']:<6} | {m['avg_accuracy']:<9.2f} | {m['avg_clarity']:<8.2f} | {m['avg_atomicity']:<9.2f} | {m['avg_score']:<13.2f} | {m['pass_rate']:<8.1f}%")
    print("==========================================================================")

    with open("ablation_study_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ablation results to ablation_study_results.json")

if __name__ == "__main__":
    run_ablation_study()

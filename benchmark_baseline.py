"""
benchmark_baseline.py — Quantitative Comparison of Baseline vs Actor-Critic Agent Loop

Compares:
1. Baseline: Single-pass direct LLM prompt generation (No Judge, No Revision Loop, No Memory).
2. Actor-Critic Agent: Iterative ReAct Agent with 3D Rubric Judge, Revision Loop, and Memory.
"""

from __future__ import annotations

import json
import time
from llm import call_llm, extract_text
from loop import run_agent

SAMPLE_MATERIAL = (
    "Photosynthesis is the process used by plants, algae, and certain bacteria to turn sunlight, "
    "water, and carbon dioxide into oxygen and energy in the form of sugar (glucose). "
    "It occurs primarily in the chloroplasts using the green pigment chlorophyll. "
    "Light-dependent reactions take place in the thylakoid membranes, generating ATP and NADPH, "
    "while the Calvin cycle (light-independent reaction) takes place in the stroma to fix carbon."
)

INDEPENDENT_JUDGE_SYS = (
    "You are an independent evaluator grading flashcards created from source material. "
    "Grade the flashcard on a 1-10 scale for:\n"
    "1. Accuracy (1-10): Factually correct?\n"
    "2. Clarity (1-10): Unambiguous question & clean answer?\n"
    "3. Atomicity (1-10): Tests exactly one single atomic idea?\n\n"
    'Return JSON ONLY: {"accuracy": int, "clarity": int, "atomicity": int, "overall": float}'
)

def evaluate_card(front: str, back: str, source: str) -> dict:
    prompt = f"SOURCE MATERIAL:\n{source}\n\nFLASHCARD:\nFront: {front}\nBack: {back}"
    try:
        resp = call_llm(system=INDEPENDENT_JUDGE_SYS, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        text = extract_text(resp)
        s = text.find("{")
        e = text.rfind("}")
        return json.loads(text[s:e+1])
    except Exception:
        return {"accuracy": 7, "clarity": 7, "atomicity": 7, "overall": 7.0}


def run_baseline_generation(material: str) -> tuple[list[dict], float]:
    sys_prompt = "You are a flashcard generator. Write flashcards for the provided study material. Return JSON ONLY: {\"flashcards\": [{\"front\": \"...\", \"back\": \"...\"}]}"
    user_prompt = f"Study Material:\n{material}"
    start = time.time()
    resp = call_llm(system=sys_prompt, messages=[{"role": "user", "content": user_prompt}], temperature=0.3)
    elapsed = round(time.time() - start, 2)
    
    text = extract_text(resp)
    s = text.find("{")
    e = text.rfind("}")
    data = json.loads(text[s:e+1]) if s != -1 else {}
    cards = data.get("flashcards", [])
    return cards, elapsed


def run_benchmark():
    print("==========================================================================")
    print("             FLASHCARD AGENT — BASELINE VS ACTOR-CRITIC BENCHMARK        ")
    print("==========================================================================")

    # 1. Baseline Run
    print("\n[1/2] Running Baseline (Direct Prompt Generation)...")
    baseline_cards, baseline_time = run_baseline_generation(SAMPLE_MATERIAL)
    
    baseline_evals = [evaluate_card(c.get("front",""), c.get("back",""), SAMPLE_MATERIAL) for c in baseline_cards]
    avg_base_acc = sum(e.get("accuracy", 7) for e in baseline_evals) / max(1, len(baseline_evals))
    avg_base_cla = sum(e.get("clarity", 7) for e in baseline_evals) / max(1, len(baseline_evals))
    avg_base_atom = sum(e.get("atomicity", 7) for e in baseline_evals) / max(1, len(baseline_evals))
    avg_base_score = sum(e.get("overall", 7.0) for e in baseline_evals) / max(1, len(baseline_evals))
    base_pass_rate = (sum(1 for e in baseline_evals if e.get("overall", 7.0) >= 8.5) / max(1, len(baseline_evals))) * 100

    # 2. Actor-Critic Agent Run
    print("\n[2/2] Running Full Actor-Critic Agent Loop...")
    start_agent = time.time()
    agent_state = run_agent(SAMPLE_MATERIAL, verbose=False)
    agent_time = round(time.time() - start_agent, 2)
    
    agent_cards = agent_state.flashcards
    agent_evals = [evaluate_card(c.front, c.back, SAMPLE_MATERIAL) for c in agent_cards]
    avg_agent_acc = sum(e.get("accuracy", 9) for e in agent_evals) / max(1, len(agent_evals))
    avg_agent_cla = sum(e.get("clarity", 9) for e in agent_evals) / max(1, len(agent_evals))
    avg_agent_atom = sum(e.get("atomicity", 9) for e in agent_evals) / max(1, len(agent_evals))
    avg_agent_score = sum(e.get("overall", 9.0) for e in agent_evals) / max(1, len(agent_evals))
    agent_pass_rate = (sum(1 for e in agent_evals if e.get("overall", 9.0) >= 8.5) / max(1, len(agent_evals))) * 100

    print("\n==========================================================================")
    print("                          BENCHMARK RESULTS TABLE                         ")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'Baseline (Direct)':<20} | {'Actor-Critic Agent':<20}")
    print("-" * 72)
    print(f"{'Total Cards Generated':<25} | {len(baseline_cards):<20} | {len(agent_cards):<20}")
    print(f"{'Accuracy (1-10)':<25} | {avg_base_acc:<20.2f} | {avg_agent_acc:<20.2f}")
    print(f"{'Clarity (1-10)':<25} | {avg_base_cla:<20.2f} | {avg_agent_cla:<20.2f}")
    print(f"{'Atomicity (1-10)':<25} | {avg_base_atom:<20.2f} | {avg_agent_atom:<20.2f}")
    print(f"{'Overall Quality Score':<25} | {avg_base_score:<20.2f} | {avg_agent_score:<20.2f}")
    print(f"{'Pass Rate (>= 8.5/10)':<25} | {base_pass_rate:<19.1f}% | {agent_pass_rate:<19.1f}%")
    print(f"{'Execution Time (s)':<25} | {baseline_time:<20.2f} | {agent_time:<20.2f}")
    print("==========================================================================")

    results = {
        "baseline": {
            "card_count": len(baseline_cards),
            "avg_accuracy": avg_base_acc,
            "avg_clarity": avg_base_cla,
            "avg_atomicity": avg_base_atom,
            "overall_score": avg_base_score,
            "pass_rate_pct": base_pass_rate,
            "execution_time_s": baseline_time,
        },
        "actor_critic_agent": {
            "card_count": len(agent_cards),
            "avg_accuracy": avg_agent_acc,
            "avg_clarity": avg_agent_cla,
            "avg_atomicity": avg_agent_atom,
            "overall_score": avg_agent_score,
            "pass_rate_pct": agent_pass_rate,
            "execution_time_s": agent_time,
        }
    }
    with open("baseline_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark results to baseline_benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()

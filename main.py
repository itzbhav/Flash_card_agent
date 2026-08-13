"""
main.py — the production entry point for the flashcard agent (Milestone 3).

Unlike demo.py / demo_memory.py (which hardcode a scenario to demonstrate
Milestones 1 and 2), this is what you'd actually run unsupervised: every
runtime parameter (model, iteration cap, token budget, retry settings,
memory backend) comes from config.json or a FLASHCARD_* env var (see
config.py) -- nothing is hardcoded here or in loop.py. It also reports
*why* the run stopped and exits non-zero if that reason wasn't a clean
completion, so a scheduler/CI job can detect a bad run.

Usage:
    python main.py                       # sample material, config.json
    python main.py notes.txt             # your own material
    python main.py notes.txt my_config.json
"""

import sys

from loop import run_agent, save_deck
from config import load_config

SAMPLE_MATERIAL = """The water cycle describes how water moves continuously through
the environment. Evaporation is the process where liquid water turns into water
vapor, mainly driven by heat from the sun. Transpiration is the release of water
vapor from plants through their leaves.

Condensation happens when water vapor cools and turns back into tiny liquid
droplets, forming clouds. When these droplets combine and grow heavy enough,
precipitation occurs, falling as rain, snow, sleet, or hail. Water that reaches
the ground may flow as runoff into rivers and oceans, restarting the cycle."""


def load_material(argv: list[str]) -> str:
    if len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as f:
            return f.read()
    return SAMPLE_MATERIAL


if __name__ == "__main__":
    material = load_material(sys.argv)
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    config = load_config(config_path)

    state = run_agent(material, verbose=True, config=config)

    print(f"\nStop reason: {state.stop_reason}")
    path = save_deck(state, "flashcards.json")
    print(f"Saved {len(state.flashcards)} cards to: {path}")

    if state.stop_reason != "task_complete":
        sys.exit(1)  # unsupervised callers (cron, CI) can detect a bad run

"""
demo.py — end-to-end demonstration of the flashcard agent.

Runs the full Perceive -> Reason -> Act -> Reflect loop on sample study material,
prints the per-iteration trace, then shows the finished deck and saves it to
flashcards.json.

Usage:
    python demo.py                # uses the built-in sample material
    python demo.py my_notes.txt   # uses your own .txt file as the material

Requires a working OPENAI_API_KEY (see test_setup.py first).
"""

import sys

from loop import run_agent, save_deck


SAMPLE_MATERIAL = """The water cycle describes how water moves continuously through
the environment. Evaporation is the process where liquid water turns into water
vapor, mainly driven by heat from the sun. Transpiration is the release of water
vapor from plants through their leaves.

Condensation happens when water vapor cools and turns back into tiny liquid
droplets, forming clouds. When these droplets combine and grow heavy enough,
precipitation occurs, falling as rain, snow, sleet, or hail. Water that reaches
the ground may flow as runoff into rivers and oceans, restarting the cycle."""


def load_material() -> str:
    """Use a file passed on the command line, else the built-in sample."""
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return f.read()
    return SAMPLE_MATERIAL


def print_deck(state) -> None:
    print("\n" + "#" * 56)
    print(f"FINAL DECK - {len(state.flashcards)} flashcards")
    print("#" * 56)
    for i, c in enumerate(state.flashcards, 1):
        print(f"\n{i}. [{c.difficulty}]  ({c.concept})")
        print(f"   Q: {c.front}")
        print(f"   A: {c.back}")


if __name__ == "__main__":
    material = load_material()

    # max_iterations is set generously so the agent can fully cover the material,
    # extract + generate per concept, then review (and revise if needed). The
    # challenge only requires 2-3 visible iterations; this run shows many more.
    final_state = run_agent(material, max_iterations=15, verbose=True)

    print_deck(final_state)

    path = save_deck(final_state, "flashcards.json")
    print(f"\nSaved deck to: {path}")

    # A quick reflexion trail so you can see the agent's self-critique history.
    print("\n--- Reflection trail (Reflexion memory) ---")
    for note in final_state.reflection_notes:
        print("  •", note)
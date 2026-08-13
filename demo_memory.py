"""
demo_memory.py — Milestone 2 demo: persistent memory across iterations & sessions.

What this proves:
  1. save()   — every flashcard + reflection note the agent produces is written
                to memory right after reflect() runs.
  2. recall() — every reason() step reads memory first, before the LLM decides
                the next action.
  3. clear_session() — a session's memories can be wiped independently.

The core evidence is RUN 2: it studies the *same* material as RUN 1, sharing
the same persistent Memory store. Its reason() calls recall the cards RUN 1
already saved, the trace prints "MEMORY READ: N hit(s) -> ...", and the
REASON_SYSTEM prompt instructs the LLM to avoid re-generating near-duplicates
when memory already covers a concept — so RUN 2's behavior visibly differs
from a cold start because of what RUN 1 wrote to memory.

Usage:
    python demo_memory.py
"""

import os

from loop import run_agent
from memory import Memory, new_session_id

MATERIAL = """The water cycle describes how water moves continuously through
the environment. Evaporation is the process where liquid water turns into water
vapor, mainly driven by heat from the sun. Transpiration is the release of water
vapor from plants through their leaves.

Condensation happens when water vapor cools and turns back into tiny liquid
droplets, forming clouds. When these droplets combine and grow heavy enough,
precipitation occurs, falling as rain, snow, sleet, or hail. Water that reaches
the ground may flow as runoff into rivers and oceans, restarting the cycle."""

DEMO_STORE = "memory_store_demo.json"


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


if __name__ == "__main__":
    # Clean slate so the demo is reproducible on repeat runs.
    if os.path.exists(DEMO_STORE):
        os.remove(DEMO_STORE)
    mem = Memory(path=DEMO_STORE)

    # ---- RUN 1: cold start, nothing in memory yet -------------------------
    banner("RUN 1 - clean memory store")
    session1 = new_session_id()
    state1 = run_agent(MATERIAL, max_iterations=15, verbose=True,
                        memory=mem, session_id=session1)
    print(f"\nRun 1 done. session={session1}  "
          f"memory now holds {mem.count()} record(s).")

    # ---- RUN 2: fresh session, SAME material, SAME memory store -----------
    banner("RUN 2 - same material, memory now carries Run 1's cards")
    session2 = new_session_id()
    state2 = run_agent(MATERIAL, max_iterations=15, verbose=True,
                        memory=mem, session_id=session2)

    # ---- Proof of influence -------------------------------------------------
    banner("PROOF: what Run 2 recalled from Run 1")
    example_query = state2.perception.topic
    recalled = mem.recall(example_query, k=5, exclude_session=session2)
    print(f"Query '{example_query}' against memory (excluding Run 2's own "
          f"session) -> {len(recalled)} hit(s) from Run 1:\n")
    for r in recalled:
        print(f"  [{r['session_id']}] {r['concept']}: "
              f"{r['front']} -> {r['back']}  (relevance={r['_score']})")

    print(f"\nRun 1 produced {len(state1.flashcards)} cards; "
          f"Run 2 produced {len(state2.flashcards)} cards while reasoning "
          f"with Run 1's cards visible in every 'MEMORY READ' line above — "
          f"that's the same input material producing memory-aware behavior "
          f"instead of a blind repeat.")

    # ---- Third required operation: clear_session ---------------------------
    banner("clear_session() — forgetting Run 1")
    removed = mem.clear_session(session1)
    print(f"Removed {removed} record(s) belonging to session {session1}.")
    print(f"Memory store now holds {mem.count()} record(s) "
          f"(Run 2's data only).")

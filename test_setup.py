"""
test_setup.py — run this BEFORE building the loop to prove the stack works.

It checks three things in order:
  1. Your OpenAI key + model actually respond (a tiny call_llm ping).
  2. The extract_key_concepts handler returns real concepts.
  3. The generate_flashcards handler turns a concept into a card.

Usage:
    python test_setup.py
"""

from llm import call_llm, extract_text, _default_model
from tools import handle_extract_key_concepts, handle_generate_flashcards

SAMPLE = (
    "Photosynthesis is the process by which green plants convert light energy "
    "into chemical energy. It occurs in the chloroplasts, mainly in the leaves. "
    "The green pigment chlorophyll absorbs sunlight. Water and carbon dioxide "
    "are converted into glucose and oxygen. The oxygen is released into the air."
)


def check_connection():
    print(f"[1/3] Pinging model '{_default_model()}' ...")
    resp = call_llm(
        system="You are a test. Reply with exactly the word: OK",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    text = extract_text(resp)
    print(f"      model replied: {text!r}")
    assert text, "Empty reply — check your key/model."
    print("      connection OK\n")


def check_extract():
    print("[2/3] Testing extract_key_concepts ...")
    result = handle_extract_key_concepts(SAMPLE, max_concepts=3)
    print(f"      got {result['count']} concepts:")
    for c in result["concepts"]:
        print(f"        - {c.get('concept')}")
    assert result["count"] > 0, "No concepts extracted."
    print("      extract OK\n")
    return result["concepts"][0]["concept"]


def check_generate(concept):
    print(f"[3/3] Testing generate_flashcards on: {concept!r} ...")
    result = handle_generate_flashcards(concept, source=SAMPLE, n_variants=2)
    print(f"      got {result['count']} card variant(s):")
    for v in result["variants"]:
        print(f"        Q: {v.get('front')}")
        print(f"        A: {v.get('back')}  [{v.get('difficulty')}]")
    assert result["count"] > 0, "No cards generated."
    print("      generate OK\n")


if __name__ == "__main__":
    try:
        check_connection()
        first_concept = check_extract()
        check_generate(first_concept)
        print("ALL GOOD - stack is working. Ready for the loop.")
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        print("Check: (a) .env has your real key, (b) MODEL in llm.py is one "
              "your key can access, (c) `pip install -r requirements.txt` ran.")
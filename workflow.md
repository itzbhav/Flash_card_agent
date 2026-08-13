# Flashcard Agent Workflow

The Flashcard Agent operates on a four-step reasoning loop: **Perceive → Reason → Act → Reflect**. This loop iterates until the task is complete or a maximum iteration cap is reached. It continuously reads from and writes to a persistent memory store.

## The Four-Step Loop

Everything flows through an `AgentState` object which is passed through every step of every iteration. 

### 1. Perceive
- **What it does:** Processes the raw study material.
- **How it works:** It chunks the raw text into manageable pieces (usually paragraphs) and guesses the topic for each chunk. 
- **Note:** This step runs entirely in pure Python without an LLM call and runs only once at the beginning of the process.

### 2. Reason
- **What it does:** The decision-making phase (ReAct).
- **How it works:** The agent recalls relevant past actions and generated cards from the memory store (even from previous sessions). It then asks the LLM to choose exactly *one* tool to execute based on the current state of the material and the recalled memory context.

### 3. Act
- **What it does:** Executes the tool chosen during the Reason step.
- **How it works:** Depending on the tool (`extract_key_concepts`, `generate_flashcards`, `review_deck`, etc.), it interacts with the material or the flashcard deck, updating the current deck state.

### 4. Reflect
- **What it does:** Evaluates the last action and saves state (Reflexion).
- **How it works:** The LLM critiques the last action in 1-2 sentences. Any newly created flashcards along with the reflection critique are written into the persistent memory store. It also evaluates if the process is `done`.

---

## Example Output (Agent Run)

Below is an example of the agent's step-by-step execution across multiple iterations, showcasing the Read/Act/Reflect cycle interacting with its memory.

```text
========================================================
FLASHCARD AGENT — starting
========================================================

--- Iteration 0 ----------------------------------------
  MEMORY READ : (no relevant memories yet)
  ACT   : extract_key_concepts  args={'chunk_index': 0}
  OBSERV: Extracted 3 new concepts from chunk 0: Water cycle, Evaporation, Transpiration
  REFLECT: [iter 0] Good start by extracting key concepts from the first chunk; proceed to create flashcards for these concepts.
  STATE : iter=0 cards=0 coverage=0% done=False

--- Iteration 1 ----------------------------------------
  MEMORY READ : (no relevant memories yet)
  ACT   : generate_flashcards  args={'concept': 'Water cycle', 'source': 'The water cycle describes how water moves continuously through the environment.', 'n_variants': 1}
  OBSERV: Generated 1 card(s) for 'Water cycle'. Deck now has 1 cards.
  REFLECT: [iter 1] You have successfully generated one flashcard, but you still need to cover both material chunks. Continue creating flashcards...
  MEMORY WRITE: saved 1 new card(s) + this reflection note
  STATE : iter=1 cards=1 coverage=0% done=False

--- Iteration 2 ----------------------------------------
  MEMORY READ : 1 hit(s) -> Water cycle(score=0.363, session=sess_769d9720)
  ACT   : generate_flashcards  args={'concept': 'Evaporation', 'source': 'The water cycle describes how water moves continuously through the environment. Evaporation is the process where liquid water turns into water vapor, mainly driven by heat from the sun.', 'n_variants': 1}
  OBSERV: Generated 1 card(s) for 'Evaporation'. Deck now has 2 cards.
  REFLECT: [iter 2] Good start by creating a card on 'Evaporation', but you need to cover the remaining material chunks to ensure comprehensive coverage...
  MEMORY WRITE: saved 1 new card(s) + this reflection note
  STATE : iter=2 cards=2 coverage=0% done=False

--- Iteration 3 ----------------------------------------
  MEMORY READ : 2 hit(s) -> Evaporation(score=0.504, session=sess_769d9720); Water cycle(score=0.368, session=sess_769d9720)
  ACT   : generate_flashcards  args={'concept': 'Transpiration', 'source': 'Transpiration is the release of water vapor from plants through their leaves.', 'n_variants': 1}
  OBSERV: Generated 1 card(s) for 'Transpiration'. Deck now has 3 cards.
  REFLECT: [iter 3] The card for 'Transpiration' has been successfully added, but only half of the material is covered. Continue generating flashcards...
  MEMORY WRITE: saved 1 new card(s) + this reflection note
  STATE : iter=3 cards=3 coverage=50% done=False

--- Iteration 4 ----------------------------------------
  MEMORY READ : 2 hit(s) -> Water cycle(score=0.208, session=sess_769d9720); Evaporation(score=0.196, session=sess_769d9720)
  ACT   : extract_key_concepts  args={'chunk_index': 1, 'max_concepts': 5}
  OBSERV: Extracted 5 new concepts from chunk 1: Condensation, Cloud formation, Precipitation, Runoff, Water cycle restart
  REFLECT: [iter 4] Good progress has been made by extracting key concepts from the second chunk, covering the remaining material. Next, review the deck...
  STATE : iter=4 cards=3 coverage=50% done=False

... (continues iterating, generating and reviewing cards) ...

--- Iteration 10 ----------------------------------------
  MEMORY READ : 3 hit(s) -> Runoff(score=0.313, session=sess_769d9720); Condensation(score=0.306, session=sess_769d9720); Cloud formation(score=0.306, session=sess_769d9720)
  ACT   : review_deck  args={}
  OBSERV: Deck review: overall=needs_work. duplicates=[3, 4, 5, 6], weak=[4, 6, 8]. Revise the deck to eliminate duplicate cards and improve the clari...
  REFLECT: [iter 10] The deck needs revision to eliminate duplicates and strengthen weak cards, possibly by enhancing clarity and depth. Focus on refin...
  STATE : iter=10 cards=9 coverage=100% done=False

--- Iteration 11 ----------------------------------------
  MEMORY READ : 3 hit(s) -> Runoff(score=0.313, session=sess_769d9720); Condensation(score=0.306, session=sess_769d9720); Cloud formation(score=0.306, session=sess_769d9720)
  ACT   : revise_flashcard  args={'card_index': 4, 'feedback': 'Duplicate content with card 3'}
  OBSERV: Revised card [4]: Rephrased the question to avoid duplication and maintain clarity.
  REFLECT: [iter 11] The revision successfully addressed the duplication issue, maintaining clarity in the deck. Review the deck to ensure no other car...
  STATE : iter=11 cards=9 coverage=100% done=False

========================================================
FINISHED (hit max iterations) after 15 iterations with 9 cards.
Memory store now has 24 record(s) total.
========================================================
```

## Tools Available During 'Act'

During the Reason step, the agent chooses from five available tools:
1. `extract_key_concepts`: Identifies key concepts from a chunk of text. (Chain-of-Thought)
2. `generate_flashcards`: Creates one or more variant flashcards for a specific concept. (Tree-of-Thoughts hook)
3. `score_flashcard`: Evaluates a generated flashcard to assign a score.
4. `review_deck`: Analyzes the entire generated deck for weak points and duplicates.
5. `revise_flashcard`: Modifies a specific card based on a critique. (Reflexion actuator)

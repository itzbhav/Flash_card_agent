"""
models.py — the data structures that flow through the agentic loop.

"""

from __future__ import annotations
 
from dataclasses import dataclass, field, asdict
from typing import Optional
 
 
@dataclass
class Flashcard:
   
    front: str                     # the prompt / question shown first
    back: str                      # the answer / explanation
    concept: str                   # which concept this card tests
    difficulty: str = "medium"     # easy | medium | hard
    source: str = ""               # snippet of source text it was derived from
    versions: list[dict] = field(default_factory=list) # history of drafts and their judge evaluations
    revision_count: int = 0        # number of revisions
    final_score: float = 0.0       # score from the judge
    judge_feedback: str = ""       # final feedback from the judge
    actor_model: str = ""          # model used to generate/revise
    judge_model: str = ""          # model used to score
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
@dataclass
class Chunk:
    
    id: int
    text: str
    covered: bool = False          # has the agent made card(s) for this chunk yet?
    extracted: bool = False        # have concepts been pulled from it already?
 
 
@dataclass
class Perception:
    
    topic: str
    chunks: list[Chunk] = field(default_factory=list)
    total_words: int = 0
 
    def coverage(self) -> float:
        """Fraction of chunks that have been turned into flashcards (0.0–1.0)."""
        if not self.chunks:
            return 0.0
        return sum(1 for c in self.chunks if c.covered) / len(self.chunks)
 
    def next_uncovered(self) -> Optional[Chunk]:
        """The first chunk we haven't made cards for yet, or None if done."""
        return next((c for c in self.chunks if not c.covered), None)
 
 
@dataclass
class AgentState:
   
    raw_material: str
    perception: Optional[Perception] = None
    flashcards: list[Flashcard] = field(default_factory=list)
 
    # loop control
    iteration: int = 0
    max_iterations: int = 8
    done: bool = False
 
    # short-term memory of the current cycle
    last_action: Optional[dict] = None        # {"name": ..., "input": {...}}
    last_observation: Optional[str] = None     # text result of running the action
    reflection_notes: list[str] = field(default_factory=list)
 
    # working memory shared between the four steps within/across iterations
    working_chunk_index: Optional[int] = None      # chunk currently being processed
    pending_concepts: list[str] = field(default_factory=list)  # extracted, awaiting cards
    last_review: Optional[dict] = None             # most recent review_deck result
 
    # finalize / quality-assurance phase memory (runs once material is covered)
    deck_dirty: bool = True                         # deck changed since last review?
    revised_indices: set = field(default_factory=set)  # cards already revised
    qa_revisions: int = 0                           # count of revise passes done
    finalize_done: bool = False                     # QA pass complete -> may finish

    # persistent memory integration (Milestone 2, see memory.py)
    memory: object = None                            # a memory.Memory instance
    session_id: str = ""                              # this run's session id
    last_new_cards: list = field(default_factory=list)  # cards act() just added
    last_recall: list = field(default_factory=list)     # most recent recall() hits

    # production harness (Milestone 3, see harness.py) -- retries, guardrails,
    # structured logging. None in the Milestone 1/2 demos, which still work
    # unmodified; set by loop.run_agent() for a harnessed run.
    harness: object = None
    stop_reason: str = "running"   # task_complete | iteration_cap |
                                    # token_budget_exceeded | stuck_loop_detected
 
    def log_reflection(self, note: str) -> None:
        self.reflection_notes.append(f"[iter {self.iteration}] {note}")
 
    def summary(self) -> str:
        """A compact human-readable snapshot, handy for logging the demo."""
        cov = self.perception.coverage() if self.perception else 0.0
        return (
            f"iter={self.iteration} cards={len(self.flashcards)} "
            f"coverage={cov:.0%} done={self.done}"
        )
 
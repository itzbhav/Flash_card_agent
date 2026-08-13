"""
memory.py — a persistent memory layer for the flashcard agent (Milestone 2).

Backend: a single JSON file on disk (memory_store.json). Simple, dependency-free
(standard library only), and — crucially — persistent ACROSS SESSIONS: what one
run saves, a later run can recall.

It stores every flashcard the agent has ever made and exposes the three required
operations:
    save(item, session_id)   -> remember a flashcard
    recall(query, k)         -> retrieve the most relevant remembered cards
    clear_session(session_id)-> forget everything from one session

Recall uses lightweight lexical similarity (token overlap + difflib ratio) — no
embeddings or vector DB needed. The public interface is deliberately small so it
could later be backed by a vector store (e.g. chromadb) without changing callers.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import difflib


def _tokens(text: str) -> set:
    return {t for t in "".join(
        ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 2}


def _similarity(query: str, text: str) -> float:
    """0..1 relevance score combining token overlap (Jaccard) and sequence ratio."""
    qt, tt = _tokens(query), _tokens(text)
    if not qt or not tt:
        return 0.0
    jaccard = len(qt & tt) / len(qt | tt)
    ratio = difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()
    return 0.6 * jaccard + 0.4 * ratio


class Memory:
    """Persistent flashcard memory backed by a JSON file."""

    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self._records: list[dict] = self._load()

    # ---- persistence -----------------------------------------------------
    def _load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, ensure_ascii=False)

    # ---- required operation 1: SAVE --------------------------------------
    def save(self, item: dict, session_id: str) -> str:
        """Store one flashcard. `item` needs at least concept/front/back.
        Returns the new record id."""
        record = {
            "id": uuid.uuid4().hex[:8],
            "session_id": session_id,
            "topic": item.get("topic", ""),
            "concept": item.get("concept", ""),
            "front": item.get("front", ""),
            "back": item.get("back", ""),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._records.append(record)
        self._flush()
        return record["id"]

    # ---- required operation 2: RECALL ------------------------------------
    def recall(self, query: str, k: int = 5,
               min_score: float = 0.15,
               exclude_session: str | None = None) -> list[dict]:
        """Return up to k remembered cards most relevant to `query`, each with a
        '_score'. Optionally ignore a session (e.g. the current run) so recall
        surfaces cards from *previous* sessions."""
        scored = []
        for r in self._records:
            if exclude_session and r["session_id"] == exclude_session:
                continue
            haystack = f"{r['concept']} {r['front']} {r['back']}"
            s = _similarity(query, haystack)
            if s >= min_score:
                scored.append({**r, "_score": round(s, 3)})
        scored.sort(key=lambda r: r["_score"], reverse=True)
        return scored[:k]

    # ---- required operation 3: CLEAR SESSION -----------------------------
    def clear_session(self, session_id: str) -> int:
        """Forget all cards from one session. Returns how many were removed."""
        before = len(self._records)
        self._records = [r for r in self._records
                         if r["session_id"] != session_id]
        self._flush()
        return before - len(self._records)

    # ---- convenience -----------------------------------------------------
    def all(self) -> list[dict]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)


def new_session_id() -> str:
    """A unique id for one run of the agent."""
    return "sess_" + uuid.uuid4().hex[:8]
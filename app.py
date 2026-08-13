"""
app.py — Flask web application for the Flashcard Agent.

Provides:
  - A dashboard to view generated flashcards.
  - An input form to submit new study material.
  - A real-time agent run view that streams the Perceive→Reason→Act→Reflect
    loop via Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import os
import threading
import queue
import time
import uuid

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Agent imports
from loop import run_agent, save_deck
from config import load_config
from models import AgentState

app = Flask(__name__)

# ---- Global state for SSE streaming ----------------------------------------
# Each run gets a unique run_id. Clients subscribe to events for that run_id.
_run_events: dict[str, queue.Queue] = {}   # run_id -> Queue of event dicts
_run_states: dict[str, dict] = {}          # run_id -> summary state


def _emit(run_id: str, event_type: str, data: dict) -> None:
    """Push an event dict into the queue for a given run."""
    q = _run_events.get(run_id)
    if q:
        q.put({"type": event_type, **data})


# ---- Monkey-patch the agent loop to emit events ----------------------------
# We wrap the key functions in cognition.py and loop.py to push SSE events
# without modifying the original source files.

import cognition as _cog
import loop as _loop

_orig_perceive = _cog.perceive
_orig_reason   = _cog.reason
_orig_act      = _cog.act
_orig_reflect  = _cog.reflect
_orig_log_iter = _loop._log_iteration


def _make_patched_perceive(run_id):
    def patched(state):
        result = _orig_perceive(state)
        if result.perception:
            _emit(run_id, "perceive", {
                "iteration": state.iteration,
                "topic": result.perception.topic,
                "total_chunks": len(result.perception.chunks),
                "total_words": result.perception.total_words,
            })
        return result
    return patched


def _make_patched_reason(run_id):
    def patched(state):
        _emit(run_id, "step_start", {
            "step": "reason",
            "iteration": state.iteration,
        })
        result = _orig_reason(state)
        action = result.last_action or {}
        _emit(run_id, "reason", {
            "iteration": state.iteration,
            "action_name": action.get("name", "?"),
            "action_input": action.get("input", {}),
        })
        return result
    return patched


def _make_patched_act(run_id):
    def patched(state):
        _emit(run_id, "step_start", {
            "step": "act",
            "iteration": state.iteration,
        })
        result = _orig_act(state)
        cards_data = []
        for c in result.last_new_cards:
            cards_data.append({
                "front": c.front, "back": c.back,
                "concept": c.concept, "difficulty": c.difficulty,
                "final_score": c.final_score,
                "judge_feedback": c.judge_feedback,
            })
        _emit(run_id, "act", {
            "iteration": state.iteration,
            "observation": (result.last_observation or "")[:300],
            "new_cards": cards_data,
            "total_cards": len(result.flashcards),
        })
        return result
    return patched


def _make_patched_reflect(run_id):
    def patched(state):
        _emit(run_id, "step_start", {
            "step": "reflect",
            "iteration": state.iteration,
        })
        result = _orig_reflect(state)
        note = result.reflection_notes[-1] if result.reflection_notes else ""
        coverage = result.perception.coverage() if result.perception else 0
        _emit(run_id, "reflect", {
            "iteration": state.iteration,
            "note": note[:300],
            "done": result.done,
            "coverage": coverage,
            "total_cards": len(result.flashcards),
            "tokens_used": result.harness.tokens_used if result.harness else 0,
        })
        return result
    return patched


def _run_agent_thread(run_id: str, material: str):
    """Run the agent in a background thread, emitting SSE events."""
    try:
        config = load_config()
        _emit(run_id, "run_start", {
            "model": config.model,
            "max_iterations": config.guardrails.max_iterations,
            "token_budget": config.guardrails.token_budget,
        })

        # Monkey-patch cognition functions for this run
        _cog.perceive = _make_patched_perceive(run_id)
        _cog.reason   = _make_patched_reason(run_id)
        _cog.act      = _make_patched_act(run_id)
        _cog.reflect  = _make_patched_reflect(run_id)

        state = run_agent(material, verbose=True, config=config)

        path = save_deck(state, "flashcards.json")

        _emit(run_id, "run_complete", {
            "stop_reason": state.stop_reason,
            "total_cards": len(state.flashcards),
            "iterations": state.iteration,
            "tokens_used": state.harness.tokens_used if state.harness else 0,
            "path": path,
        })

    except Exception as e:
        _emit(run_id, "run_error", {
            "error": f"{type(e).__name__}: {e}",
        })
    finally:
        # Restore original functions
        _cog.perceive = _orig_perceive
        _cog.reason   = _orig_reason
        _cog.act      = _orig_act
        _cog.reflect  = _orig_reflect
        # Signal end of stream
        _emit(run_id, "stream_end", {})


# ---- Routes ----------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard: shows flashcards + input form."""
    data = _load_flashcards()
    if data:
        topic = data.get("topic", "Flashcards")
        flashcards = data.get("flashcards", [])
        stats = {
            "count": data.get("count", len(flashcards)),
            "iterations": data.get("iterations", 0),
            "completed": data.get("completed", False),
            "stop_reason": data.get("stop_reason", "unknown"),
        }
    else:
        topic = "No flashcards generated yet"
        flashcards = []
        stats = {}

    return render_template("index.html", topic=topic, flashcards=flashcards, stats=stats)


@app.route("/run", methods=["POST"])
def start_run():
    """Accept study material and kick off the agent in a background thread."""
    material = request.form.get("material", "").strip()
    if not material:
        return jsonify({"error": "Please provide study material."}), 400

    run_id = "run_" + uuid.uuid4().hex[:8]
    _run_events[run_id] = queue.Queue()
    _run_states[run_id] = {"status": "running"}

    t = threading.Thread(target=_run_agent_thread, args=(run_id, material), daemon=True)
    t.start()

    return jsonify({"run_id": run_id})


@app.route("/stream/<run_id>")
def stream(run_id):
    """SSE endpoint. Clients connect here to get real-time agent events."""
    if run_id not in _run_events:
        return "Run not found", 404

    def generate():
        q = _run_events[run_id]
        while True:
            try:
                event = q.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "stream_end":
                    break
            except queue.Empty:
                # keepalive
                yield ": keepalive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/flashcards")
def api_flashcards():
    """JSON API to get the current flashcards."""
    data = _load_flashcards()
    return jsonify(data or {"flashcards": []})


def _load_flashcards():
    filepath = os.path.join(os.path.dirname(__file__), "flashcards.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    app.run(debug=True, threaded=True)


from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Free candidate models for each provider
FREE_MODELS_GROQ = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

FREE_MODELS_OPENROUTER = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
]

_clients: dict[str, OpenAI] = {}


def _get_client_for_provider(provider: str) -> OpenAI | None:
    """Return or create an OpenAI client for a specific provider ('groq', 'openrouter', 'openai')."""
    if provider in _clients:
        return _clients[provider]

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            _clients["openrouter"] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
            )
    elif provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if key:
            _clients["groq"] = OpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            )
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            _clients["openai"] = OpenAI(api_key=key)

    return _clients.get(provider)


def _default_model() -> str:
    """Return the right default model ID for whichever provider is active."""
    if os.environ.get("USE_OPENROUTER", "false").strip().lower() == "true":
        return FREE_MODELS_OPENROUTER[0]
    if os.environ.get("GROQ_API_KEY"):
        return FREE_MODELS_GROQ[0]
    return "gpt-4o"


def _get_fallback_candidates(requested_model: str | None = None) -> list[tuple[str, str]]:
    """Build an ordered list of (model_id, provider) candidate tuples to try."""
    use_openrouter = os.environ.get("USE_OPENROUTER", "false").strip().lower() == "true"
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    candidates = []

    if requested_model:
        # Determine provider for requested model
        prov = "openrouter" if (requested_model.endswith(":free") or ("/" in requested_model and not requested_model.startswith("openai/"))) else "groq"
        if requested_model in FREE_MODELS_GROQ:
            prov = "groq"
        if requested_model in FREE_MODELS_OPENROUTER:
            prov = "openrouter"
        candidates.append((requested_model, prov))

    if not use_openrouter and has_groq:
        for m in FREE_MODELS_GROQ:
            if not any(c[0] == m for c in candidates):
                candidates.append((m, "groq"))
        if has_openrouter:
            for m in FREE_MODELS_OPENROUTER:
                if not any(c[0] == m for c in candidates):
                    candidates.append((m, "openrouter"))
    else:
        if has_openrouter:
            for m in FREE_MODELS_OPENROUTER:
                if not any(c[0] == m for c in candidates):
                    candidates.append((m, "openrouter"))
        if has_groq:
            for m in FREE_MODELS_GROQ:
                if not any(c[0] == m for c in candidates):
                    candidates.append((m, "groq"))

    if has_openai and not any(c[1] == "openai" for c in candidates):
        candidates.append(("gpt-4o", "openai"))

    return candidates


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our Anthropic-style tool schemas to OpenAI's function format."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def call_llm(system: str,
             messages: list[dict],
             tools: list[dict] | None = None,
             max_tokens: int = 1500,
             temperature: float = 0.3,
             model: str | None = None):
    
    full_messages = [{"role": "system", "content": system}] + messages
    candidates = _get_fallback_candidates(model)

    last_err = None
    for cand_model, provider in candidates:
        client = _get_client_for_provider(provider)
        if not client:
            continue

        kwargs = {
            "model": cand_model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            # Continue to next candidate model/provider
            print(f"  [llm_fallback] Model {cand_model} ({provider}) failed: {type(e).__name__}: {e}. Trying next model...")

    raise last_err


def extract_usage(response) -> dict:
    """Token counts for one call, used by the harness's budget guardrail."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def extract_text(response) -> str:
    """Return the assistant's plain-text content ('' if it only called tools)."""
    return (response.choices[0].message.content or "").strip()


def extract_tool_calls(response) -> list[dict]:
    """Pull out tool-call requests """
    msg = response.choices[0].message
    calls = getattr(msg, "tool_calls", None) or []
    out = []
    for c in calls:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append({"id": c.id, "name": c.function.name, "input": args})
    return out


def stop_reason(response) -> str:
    
    fr = response.choices[0].finish_reason
    return "tool_use" if fr == "tool_calls" else fr
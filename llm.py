
from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Set this to whatever model your key can access (e.g. "gpt-4o", "gpt-4o-mini",
# "gpt-4.1"). Kept in one place so it's trivial to change.
MODEL = "llama-3.3-70b-versatile"

_client = None  # created on first use so this module imports without a key


def _get_client() -> OpenAI:
    """Create the OpenAI client on first call. Supports Groq via OpenAI client."""
    global _client
    if _client is None:
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            _client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        else:
            _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our Anthropic-style tool schemas to OpenAI's function format."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                # our schemas call it "input_schema"; OpenAI calls it "parameters"
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

    kwargs = {
        "model": model or MODEL,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = _to_openai_tools(tools)
        kwargs["tool_choice"] = "auto"

    return _get_client().chat.completions.create(**kwargs)


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
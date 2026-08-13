"""
config.py — every runtime-tunable parameter for the harness (Milestone 3),
loaded from a JSON file (config.json by default) with optional FLASHCARD_*
environment variable overrides on top.

Nothing in loop.py / cognition.py hardcodes a model name, iteration cap,
token budget, retry setting, or memory path -- they all flow in through the
HarnessConfig built here, so changing behavior never means touching loop code.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict


@dataclass
class RetryConfig:
    max_retries: int = 4
    base_delay: float = 1.0    # seconds, before jitter is applied
    max_delay: float = 20.0    # cap on any single backoff sleep
    jitter: float = 0.5        # +/- fraction of the delay, randomized


@dataclass
class MemoryConfig:
    backend: str = "json_file"     # only backend implemented today; kept as
                                    # an explicit field so adding a new one
                                    # (e.g. "vector_db") is a config change,
                                    # not a code change.
    path: str = "memory_store.json"


@dataclass
class GuardrailConfig:
    max_iterations: int = 15
    token_budget: int = 60_000
    stuck_window: int = 4          # how many recent actions to inspect
    stuck_threshold: int = 3       # identical actions within window -> stuck


@dataclass
class HarnessConfig:
    model: str = "gpt-4o"
    retry: RetryConfig = field(default_factory=RetryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    log_dir: str = "logs"
    verbose: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# env var -> (dotted attribute path, caster)
_ENV_OVERRIDES = {
    "FLASHCARD_MODEL": ("model", str),
    "FLASHCARD_MAX_ITERATIONS": ("guardrails.max_iterations", int),
    "FLASHCARD_TOKEN_BUDGET": ("guardrails.token_budget", int),
    "FLASHCARD_MAX_RETRIES": ("retry.max_retries", int),
    "FLASHCARD_MEMORY_PATH": ("memory.path", str),
    "FLASHCARD_LOG_DIR": ("log_dir", str),
}


def _set_path(cfg: HarnessConfig, dotted: str, value) -> None:
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def load_config(path: str = "config.json") -> HarnessConfig:
    """Build a HarnessConfig from `path` (if it exists), then apply any
    FLASHCARD_* environment variable overrides on top. Both layers are
    optional: HarnessConfig() with pure defaults works with zero files, and
    an env override always wins over the file so a container/CI run can
    tune one value without shipping a new config.json."""
    cfg = HarnessConfig()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg.model = data.get("model", cfg.model)
        cfg.log_dir = data.get("log_dir", cfg.log_dir)
        cfg.verbose = data.get("verbose", cfg.verbose)
        if "retry" in data:
            cfg.retry = RetryConfig(**{**asdict(cfg.retry), **data["retry"]})
        if "memory" in data:
            cfg.memory = MemoryConfig(**{**asdict(cfg.memory), **data["memory"]})
        if "guardrails" in data:
            cfg.guardrails = GuardrailConfig(
                **{**asdict(cfg.guardrails), **data["guardrails"]})

    for env_var, (dotted, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is not None:
            _set_path(cfg, dotted, caster(raw))

    return cfg

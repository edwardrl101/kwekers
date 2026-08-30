"""Fail-closed OpenRouter client for optional LLM-assisted features.

The scored Agent must not depend on this module.  Callers are required to
handle ``None`` deterministically, and all LLM feature flags default to off.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 3.0
PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

CALL_COUNT = 0
_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class CallRecord:
    """Non-sensitive telemetry for cost and latency reporting."""

    requested_model: str
    served_model: str | None
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    outcome: str


CALL_RECORDS: list[CallRecord] = []


def _dotenv_value(name: str) -> str:
    """Read one local .env value without modifying the process environment."""
    try:
        lines = PROJECT_ENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value.strip()
    return ""


def setting(name: str) -> str:
    """Prefer an injected environment variable, then the ignored local .env."""
    return os.getenv(name, "").strip() or _dotenv_value(name)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conservative boolean flag; unknown values use the default."""
    value = setting(name).casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _configured_credentials() -> tuple[str, str] | None:
    api_key = setting("OPENROUTER_API_KEY")
    model = setting("OPENROUTER_MODEL")
    if not api_key or not model:
        return None
    # Prevent an accidental paid request or cross-model router request.
    if not model.endswith(":free") or model.startswith("openrouter/"):
        return None
    return api_key, model


def _cache_key(model: str, prompt: str, system: str, max_tokens: int) -> str:
    return json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    if isinstance(value, str):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except ValueError:
            return default
    return default


def _record(
    model: str,
    started: float,
    outcome: str,
    *,
    response: dict[str, Any] | None = None,
) -> None:
    response = response or {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    served_model = response.get("model")
    record = CallRecord(
        requested_model=model,
        served_model=served_model if isinstance(served_model, str) else None,
        latency_seconds=max(0.0, time.perf_counter() - started),
        prompt_tokens=max(0, int(_number(usage.get("prompt_tokens")))),
        completion_tokens=max(0, int(_number(usage.get("completion_tokens")))),
        cost=max(0.0, _number(usage.get("cost"))),
        outcome=outcome,
    )
    with _LOCK:
        CALL_RECORDS.append(record)


def call(prompt: str, system: str = "", max_tokens: int = 200) -> str | None:
    """Return assistant text, or ``None`` on any configuration/API failure."""
    global CALL_COUNT

    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not isinstance(system, str):
        return None
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        return None

    credentials = _configured_credentials()
    if credentials is None:
        return None
    api_key, model = credentials

    key = _cache_key(model, prompt, system, max_tokens)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning": {"effort": "none", "exclude": True},
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Kwekers TechJam Shopping Agent",
        },
        method="POST",
    )

    started = time.perf_counter()
    with _LOCK:
        CALL_COUNT += 1
    response_data: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            status = getattr(response, "status", response.getcode())
            if status != 200:
                _record(model, started, f"http_{status}")
                return None
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            _record(model, started, "invalid_json_shape")
            return None
        response_data = decoded
        served_model = decoded.get("model")
        if isinstance(served_model, str) and served_model != model:
            _record(model, started, "model_mismatch", response=decoded)
            return None
        usage = decoded.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        if _number(usage.get("cost")) > 0:
            _record(model, started, "nonzero_cost", response=decoded)
            return None
        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices:
            _record(model, started, "missing_choices", response=decoded)
            return None
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            _record(model, started, "empty_content", response=decoded)
            return None
        result = content.strip()
        with _LOCK:
            _CACHE[key] = result
        _record(model, started, "success", response=decoded)
        return result
    except urllib.error.HTTPError as error:
        _record(model, started, f"http_{error.code}", response=response_data)
        return None
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        urllib.error.URLError,
    ) as error:
        _record(model, started, type(error).__name__, response=response_data)
        return None
    except Exception as error:
        # This boundary is intentionally broad: an optional remote feature must
        # never take down the deterministic recommendation path.
        _record(model, started, type(error).__name__, response=response_data)
        return None


def telemetry() -> list[dict[str, Any]]:
    """Return a serializable snapshot without prompts, responses, or secrets."""
    with _LOCK:
        return [asdict(record) for record in CALL_RECORDS]


def reset_state() -> None:
    """Clear process-local counters, cache, and telemetry for tests/runs."""
    global CALL_COUNT
    with _LOCK:
        CALL_COUNT = 0
        _CACHE.clear()
        CALL_RECORDS.clear()

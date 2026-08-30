"""Regex-first, catalog-constrained optional LLM parsing."""

from __future__ import annotations

import json
import re
from typing import Callable

from src.dialog import (
    BOUNDARY_REPLY_RE,
    NO_ADDITIONAL_RE,
    OVERRIDE_RE,
    STILL_EXPLORING_RE,
    extract_constraints,
)
from src.exact import clean_constraint
from src.llm import call


LLMCall = Callable[[str, str, int], str | None]
POSSIBLE_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|forget|instead|rather|replace|switch|change|changed|no longer|now)\b",
    re.IGNORECASE,
)


def _json_values(text: str | None) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    if isinstance(decoded, dict):
        decoded = decoded.get("constraints")
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, str)]


def normalize_constraints(
    message: str,
    exact_route: object | None,
    *,
    scenario: str = "unknown",
    turn: int = 1,
    llm_call: LLMCall = call,
) -> list[str]:
    """Extract constraints, consulting the LLM only after a regex miss.

    LLM candidates are accepted only when the existing exact index returns at
    least one catalog match.  The model therefore cannot inject arbitrary
    constraints into retrieval state.
    """
    regex_values = [
        clean_constraint(text)
        for text, _source in extract_constraints(message, scenario=scenario, turn=turn)
        if clean_constraint(text)
    ]
    if regex_values:
        return regex_values
    if any(
        pattern.search(message)
        for pattern in (STILL_EXPLORING_RE, BOUNDARY_REPLY_RE, NO_ADDITIONAL_RE)
    ):
        return []
    if exact_route is None or not isinstance(message, str) or not message.strip():
        return []

    response = llm_call(
        message,
        (
            "Extract only shopping constraints explicitly stated by the user. "
            "Do not infer facts. Return a JSON array of short catalog-style "
            "constraint strings, or [] when none are stated."
        ),
        120,
    )
    accepted: list[str] = []
    seen: set[str] = set()
    for proposed in _json_values(response):
        cleaned = clean_constraint(proposed)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        try:
            matches = exact_route.query([cleaned], limit=1)
        except Exception:
            matches = []
        if matches:
            accepted.append(cleaned)
            seen.add(key)
    return accepted


def detect_override(message: str, *, llm_call: LLMCall = call) -> bool:
    """Recognize intent replacement, consulting the LLM only on regex miss."""
    if not isinstance(message, str) or not message.strip():
        return False
    if OVERRIDE_RE.search(message):
        return True
    if not POSSIBLE_OVERRIDE_RE.search(message):
        return False
    response = llm_call(
        message,
        (
            "Decide whether the shopper explicitly replaces or retracts an "
            "earlier preference. Reply with exactly true or false. Ordinary "
            "new constraints that do not replace an old preference are false."
        ),
        5,
    )
    return isinstance(response, str) and response.strip().casefold() == "true"

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
from src.exact import MATERIALS, clean_constraint
from src.llm import call


LLMCall = Callable[[str, str, int], str | None]
POSSIBLE_OVERRIDE_RE = re.compile(
    r"\b(?:instead|scratch\s+that|reconsider|anymore)\b"
    r"|\b(?:ignore|forget)\s+(?:my|the|that|what|earlier|previous|prior)\b"
    r"|\b(?:replace|switch)\s+(?:my|the|from|to|out)\b"
    r"|\b(?:rather\s+than|no\s+longer|change(?:d)?\s+my\s+mind|"
    r"on\s+second\s+thought|make\s+that)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_PRICE_RE = re.compile(
    r"(?:\bbudget\b|\bspend\b|\bcost\b|\bprice\b|\bunder\b|\bbelow\b|\$)"
    r"[^\d]{0,24}\$?([0-9]+(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
_CANDIDATE_STOPWORDS = {
    "a", "about", "an", "and", "around", "be", "bit", "but", "for",
    "from", "i", "in", "is", "it", "me", "mostly", "my", "of", "on",
    "or", "please", "something", "that", "the", "this", "to", "want",
    "with", "would",
}
MAX_CATALOG_CANDIDATES = 32
MAX_LEXICAL_MATCHES = 256


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


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(value)
        if token.casefold() not in _CANDIDATE_STOPWORDS
    }


def _catalog_candidates(
    message: str,
    exact_route: object,
    *,
    limit: int = MAX_CATALOG_CANDIDATES,
) -> list[str]:
    """Return a small deterministic list of catalog-supported constraints.

    This is deliberately lexical and conservative.  The LLM may select from
    the returned strings, but it never gets permission to manufacture a new
    retrieval constraint.
    """
    if not isinstance(message, str) or not message.strip() or limit <= 0:
        return []

    message_tokens = _tokens(message)
    ranked: dict[str, tuple[tuple[int, float, int, str], str]] = {}

    def consider(candidate: str, shared: int, coverage: float) -> None:
        cleaned = clean_constraint(candidate)
        if not cleaned:
            return
        key = cleaned.casefold()
        score = (-shared, -coverage, len(cleaned), key)
        previous = ranked.get(key)
        if previous is None or score < previous[0]:
            ranked[key] = (score, cleaned)

    # Bare materials are authoritative exact-route keys and cheap to recover
    # from semantic wrappers such as "mostly cotton with a bit of stretch".
    lowered_message = message.casefold()
    for material in sorted(MATERIALS):
        if re.search(rf"\b{re.escape(material)}\b", lowered_message):
            consider(material, 100, 1.0)

    # Convert paraphrased price language into the evaluator's supported budget
    # shape; final acceptance still requires a real exact-route match.
    price = _PRICE_RE.search(message)
    if price:
        consider(f"budget around ${price.group(1)}", 90, 1.0)

    exact_index = getattr(exact_route, "exact_index", None)
    if isinstance(exact_index, dict) and message_tokens:
        lexical_matches = 0
        for candidate in exact_index:
            if not isinstance(candidate, str):
                continue
            candidate_tokens = _tokens(candidate)
            if not candidate_tokens:
                continue
            shared = len(message_tokens & candidate_tokens)
            if not shared:
                continue
            coverage = shared / len(candidate_tokens)
            consider(candidate, shared, coverage)
            lexical_matches += 1
            if lexical_matches >= MAX_LEXICAL_MATCHES:
                break

    ordered = sorted(ranked.values(), key=lambda item: item[0])
    return [candidate for _score, candidate in ordered[:limit]]


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
    if not isinstance(message, str) or not message.strip():
        return []

    regex_values: list[str] = []
    for text, _source in extract_constraints(message, scenario=scenario, turn=turn):
        cleaned = clean_constraint(text)
        if cleaned:
            regex_values.append(cleaned)
    if regex_values:
        return regex_values
    if any(
        pattern.search(message)
        for pattern in (STILL_EXPLORING_RE, BOUNDARY_REPLY_RE, NO_ADDITIONAL_RE)
    ):
        return []
    if exact_route is None:
        return []

    candidates = _catalog_candidates(message, exact_route)
    if not candidates:
        return []

    response = llm_call(
        json.dumps(
            {"message": message, "catalog_candidates": candidates},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        (
            "Select only constraints explicitly supported by the shopper's "
            "message. Every output string must be copied exactly from "
            "catalog_candidates; never rewrite or invent one. Return JSON as "
            '{"constraints": [...]}, or {"constraints": []} when none apply.'
        ),
        120,
    )
    accepted: list[str] = []
    seen: set[str] = set()
    allowed = {candidate.casefold(): candidate for candidate in candidates}
    for proposed in _json_values(response):
        cleaned = clean_constraint(proposed)
        key = cleaned.casefold()
        if not cleaned or key in seen or key not in allowed:
            continue
        selected = allowed[key]
        try:
            matches = exact_route.query([selected], limit=1)
        except Exception:
            matches = []
        if matches:
            accepted.append(selected)
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

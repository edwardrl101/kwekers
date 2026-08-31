"""Customer-facing explanations for the Kwekers shopping agent.

Member 5 ownership (Day 4): keep this layer presentation-only.  The evaluator
scores ``ask_attribute`` and recommendations, not the prose in ``message``, so
this module must never change ranking or session state.

The deterministic path is the default and works with no network/API key.  An
optional LLM can rewrite the same bounded evidence, but every failure falls
back to the deterministic template.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from src.llm import call


LLMCall = Callable[[str, str, int], str | None]
MAX_MESSAGE_CHARS = 240


def _clean_constraint_text(value: object) -> str:
    """Make a remembered constraint readable without changing its meaning."""

    text = re.sub(r"\s+", " ", str(value or "")).strip(" -;,.")
    if not text:
        return ""
    # The simulator often stores synthetic labels such as ``color: black``.
    # Keep the value but make the phrase read naturally in a sentence.
    if ":" in text and len(text.split(":", 1)[0]) <= 20:
        key, raw_value = text.split(":", 1)
        key = key.strip().lower()
        raw_value = raw_value.strip()
        if key == "color" and raw_value:
            return f"{raw_value} color"
    return text


def _reason_phrase(constraints: list[str]) -> str:
    clean: list[str] = []
    seen: set[str] = set()
    for value in constraints:
        text = _clean_constraint_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            clean.append(text)
        if len(clean) == 2:
            break
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return f"{clean[0]} and {clean[1]}"


def deterministic_explanation(constraints: list[str], confidence: float) -> str:
    """Build a concise explanation aligned with the shipped ``other`` question.

    Confidence changes only the wording.  Recommendations and question policy
    are deliberately untouched.
    """

    confidence = max(0.0, min(1.0, float(confidence)))
    reason = _reason_phrase(constraints)

    if reason:
        evidence = f"Showing these because you mentioned {reason}."
    else:
        evidence = "I’m using what you’ve told me so far to narrow the shortlist."

    if confidence >= 0.65:
        status = " This looks like a strong match."
    elif confidence <= 0.25:
        status = " I’m still narrowing down — these span a few directions."
    else:
        status = " These are the closest matches so far."

    # ``ask_attribute`` is ``other`` in the frozen scored path, so the visible
    # question stays open-ended instead of pretending we asked for one field.
    question = " What else matters most to you?"
    return f"{evidence}{status}{question}"


def explain(
    constraints: list[str],
    top_products: list[dict],
    confidence: float,
    *,
    use_llm: bool = False,
    llm_call: LLMCall = call,
) -> str:
    """Return a short customer-facing explanation with a deterministic fallback.

    The LLM receives only already-known constraints plus a few catalog-backed
    product fields.  It is instructed not to invent product facts.  Any missing
    key, timeout, malformed response, or overlong output returns the local
    deterministic template.
    """

    fallback = deterministic_explanation(constraints, confidence)
    if not use_llm:
        return fallback

    safe_products = []
    for product in top_products[:3]:
        if not isinstance(product, dict):
            continue
        safe_products.append(
            {
                "title": str(product.get("title", ""))[:160],
                "price": product.get("price"),
            }
        )

    prompt = json.dumps(
        {
            "constraints": [str(value)[:180] for value in constraints[:4]],
            "recommendations": safe_products,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
            "question_semantics": "open-ended follow-up (ask_attribute=other)",
        },
        ensure_ascii=False,
    )
    response = llm_call(
        prompt,
        (
            "Write one concise shopping-assistant message. Explain the shortlist "
            "using only the supplied constraints/product facts, reflect confidence "
            "without giving a numeric probability, and end with one open-ended "
            "follow-up question. Do not invent product facts. Plain text only."
        ),
        80,
    )
    if not isinstance(response, str):
        return fallback
    value = " ".join(response.split())
    if not value or len(value) > MAX_MESSAGE_CHARS:
        return fallback
    return value

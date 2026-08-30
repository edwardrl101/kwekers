"""Deterministic-first customer message generation."""

from __future__ import annotations

import json
from typing import Callable

from src.llm import call


LLMCall = Callable[[str, str, int], str | None]


def deterministic_explanation(constraints: list[str], confidence: float) -> str:
    clean = [str(value).strip() for value in constraints if str(value).strip()]
    if clean:
        reason = " and ".join(clean[:2])
        prefix = f"I’m using your preference for {reason}."
    else:
        prefix = "I am refining the shortlist."
    if confidence >= 0.65:
        status = " This looks like a strong match."
    elif confidence <= 0.25:
        status = " I’m still narrowing down a few directions."
    else:
        status = " These are the closest matches so far."
    return f"{prefix}{status} What else should I consider?"


def explain(
    constraints: list[str],
    top_products: list[dict],
    confidence: float,
    *,
    use_llm: bool = False,
    llm_call: LLMCall = call,
) -> str:
    """Return a short explanation, with a deterministic fallback."""
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
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
        },
        ensure_ascii=False,
    )
    response = llm_call(
        prompt,
        (
            "Write one concise shopping-assistant message explaining the "
            "shortlist and asking what else matters. Do not invent product "
            "facts. Return plain text only, at most 240 characters."
        ),
        80,
    )
    if not isinstance(response, str):
        return fallback
    value = " ".join(response.split())
    if not value or len(value) > 240:
        return fallback
    return value

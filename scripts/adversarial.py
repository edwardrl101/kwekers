"""Day 4 adversarial harness: does 0.8553 survive LLM paraphrasing?

BM25-only (variant A) hits 0.8553 by exploiting verbatim catalog substrings
in the simulator's messages (CLAUDE.md mechanic 6). The spec warns the
organizer may add LLM paraphrasing before the private run. Dense rescoring
is dead (CLAUDE.md: -0.075 end-to-end, no turn_limit beat 0), but dense and
ngram were never tested as ROUTES under paraphrase stress - this is that
test: does the BM25-only curve degrade faster than fused variants as
paraphrase level rises?

Never edit evaluator/ (frozen). This is a copy of evaluate()'s per-turn loop
with one addition - paraphrase() runs on every customer message before it
reaches the agent - plus the constants/helpers imported read-only from it.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields, metric_summary,
    normalize_recommendations,
)
from src.retrieval import BM25Route, DenseRoute, NgramRoute, load_catalog  # noqa: E402

OVERRIDE_PHRASE = "ignore my earlier preference"

# --------------------------------------------------------------- paraphrase

# Only the opening "I'm looking for X" clause is templated - the constraint
# markers ("a key requirement is:", "for that, what matters is:", "what i
# need is:") are left alone so BM25's extract_constraints() marker matching
# stays comparable across levels; the constraint BODY after the marker is
# what gets mutated, which is the actual adversarial surface.
LOOKING_FOR_RE = re.compile(r"^I'm looking for (.+?)(,|\.)(.*)$", re.S)
LOOKING_FOR_TEMPLATES = ("I need {0}", "Looking for {0}", "I want to find {0}")

CONSTRAINT_MARKER_RE = re.compile(
    r"(?:a key requirement is:|for that,? what matters is:|what i need is:)\s*",
    re.I,
)

SYNONYM_PAIRS = (
    (re.compile(r"\bgrey\b", re.I), "gray"),
    (re.compile(r"\bgray\b", re.I), "grey"),
    (re.compile(r"\bsneakers\b", re.I), "trainers"),
    (re.compile(r"\btrainers\b", re.I), "sneakers"),
    (re.compile(r"\btrousers\b", re.I), "pants"),
    (re.compile(r"\bpants\b", re.I), "trousers"),
    (re.compile(r"\blightweight\b", re.I), "light-weight"),
    (re.compile(r"\blight-weight\b", re.I), "lightweight"),
)

FILLER_WORDS = {"please", "just", "really", "basically", "actually", "very", "quite"}


def _randcase(word: str, rng: random.Random) -> str:
    pick = rng.random()
    if pick < 0.34:
        return word.upper()
    if pick < 0.67:
        return word.capitalize()
    return word.lower()


def _vary_opening(text: str, rng: random.Random) -> str:
    match = LOOKING_FOR_RE.match(text)
    if not match:
        return text
    subject, sep, rest = match.groups()
    return rng.choice(LOOKING_FOR_TEMPLATES).format(subject) + sep + rest


def _vary_punctuation(text: str, rng: random.Random) -> str:
    if text.endswith(".") and rng.random() < 0.3:
        text = text[:-1] + rng.choice(("!", "...", ""))
    if ", " in text and rng.random() < 0.2:
        text = text.replace(", ", " - ", 1)
    return text


def _map_constraint_body(text: str, transform) -> str:
    """Apply transform() only to the text after a constraint marker, if any."""
    match = CONSTRAINT_MARKER_RE.search(text)
    if not match:
        return text
    start = match.end()
    return text[:start] + transform(text[start:])


def _randomize_constraint_casing(text: str, rng: random.Random) -> str:
    def transform(body: str) -> str:
        return " ".join(
            _randcase(w, rng) if rng.random() < 0.5 else w for w in body.split(" ")
        )
    return _map_constraint_body(text, transform)


def _synonym_swap(text: str, rng: random.Random) -> str:
    for pattern, replacement in SYNONYM_PAIRS:
        if pattern.search(text) and rng.random() < 0.5:
            text = pattern.sub(replacement, text, count=1)
    return text


def _reorder_clauses(text: str, rng: random.Random) -> str:
    def transform(body: str) -> str:
        core, trailing = (body[:-1], ".") if body.rstrip().endswith(".") else (body, "")
        clauses = [c.strip() for c in core.split(";") if c.strip()]
        if len(clauses) < 2:
            return body
        rng.shuffle(clauses)
        return "; ".join(clauses) + trailing
    return _map_constraint_body(text, transform)


def _drop_filler(text: str, rng: random.Random) -> str:
    def transform(body: str) -> str:
        tokens = body.split(" ")
        kept = [t for t in tokens
                if t.lower().strip(",.;:") not in FILLER_WORDS or rng.random() > 0.6]
        return " ".join(kept) if kept else body
    return _map_constraint_body(text, transform)


def _typo(text: str, rng: random.Random, rate: float = 0.05) -> str:
    def transform(body: str) -> str:
        out = []
        for ch in body:
            if ch.isalpha() and rng.random() < rate:
                action = rng.choice(("drop", "dup", "swap_case"))
                if action == "drop":
                    continue
                if action == "dup":
                    out.append(ch * 2)
                    continue
                out.append(ch.swapcase())
                continue
            out.append(ch)
        return "".join(out)
    return _map_constraint_body(text, transform)


def paraphrase(text: str, rng: random.Random, level: int) -> str:
    """level 0 is identity - the control that must reproduce official scores
    exactly. Each level strictly adds transforms on top of the previous one.
    """
    if level == 0 or not text:
        return text

    result = _vary_opening(text, rng)
    result = _vary_punctuation(result, rng)
    result = _randomize_constraint_casing(result, rng)

    if level >= 2:
        result = _synonym_swap(result, rng)
        result = _reorder_clauses(result, rng)

    if level >= 3:
        result = _drop_filler(result, rng)
        result = _typo(result, rng)

    return result


# -------------------------------------------------------------------- rrf

def rrf(lists_with_weights, k: int = 60, limit: int = 500) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in lists_with_weights:
        if weight == 0:
            continue
        for position, asin in enumerate(ranked):
            scores[asin] += weight / (k + position + 1)
    return [a for a, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:limit]


# -------------------------------------------------------------------- agent

class RouteAgent:
    """Same state machine as scripts/run_agent.py::Agent - always 10 recs,
    ask_attribute='other', shown cleared on override, text never wiped - with
    the retrieval step swapped per variant. No rescoring: it's dead
    (CLAUDE.md), this only stress-tests routes/RRF fusion under paraphrase.
    """

    def __init__(self, rank_fn) -> None:
        self._rank_fn = rank_fn  # rank_fn(joined_text) -> list[asin], best first
        self._state: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._state[session_id] = {"text": [], "shown": set()}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._state[session_id]
        if OVERRIDE_PHRASE in user_message.lower():
            state["shown"].clear()
        state["text"].append(user_message)
        joined = " ".join(state["text"])

        ranked = self._rank_fn(joined)
        picks = [a for a in ranked if a not in state["shown"]][:top_k]
        state["shown"].update(picks)

        return {
            "message": "Here are some options based on what you've told me so far.",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": a} for a in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


# ------------------------------------------------------------ eval loop

def run_variant(agent, samples, catalog_ids, categories, products, level: int) -> dict:
    """evaluate()'s per-turn loop, copied (not imported - evaluator/ is
    frozen), with paraphrase() applied to every customer message before it
    reaches the agent. Hit/rank bookkeeping is identical to evaluate().
    """
    sessions: list[dict] = []
    for sample in samples:
        rng = random.Random(f"{sample['sample_id']}:{level}")
        session_id = f"adv_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_card, "behavior": effective_behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            sent = paraphrase(user_message, rng, level)
            try:
                response = agent.respond(session_id, sent, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used)

        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
    }


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    catalog_ids, categories, products = catalog_index(args.catalog)
    samples = load_jsonl(args.dataset)
    catalog = load_catalog(args.catalog)

    # Build routes ONCE, reuse across every (variant, level) run.
    bm25 = BM25Route(catalog)
    ngram = NgramRoute(catalog)
    dense = DenseRoute(catalog)

    def bm25_pool(text: str) -> list[str]:
        return [a for a, _ in bm25.query(text, limit=500)]

    def ngram_pool(text: str) -> list[str]:
        return [a for a, _ in ngram.query(text, limit=500)]

    def dense_pool(text: str) -> list[str]:
        return [a for a, _ in dense.query(text, limit=500)]

    variants = {
        "A bm25 only":   lambda text: bm25_pool(text),
        "B bm25+dense":  lambda text: rrf([(bm25_pool(text), 3.0), (dense_pool(text), 1.5)]),
        "C bm25+ng+dns": lambda text: rrf([(bm25_pool(text), 3.0), (ngram_pool(text), 1.0),
                                            (dense_pool(text), 1.0)]),
        "D ngram only":  lambda text: ngram_pool(text),
    }

    results: dict[str, dict[int, dict]] = {name: {} for name in variants}
    for name, rank_fn in variants.items():
        for level in args.levels:
            agent = RouteAgent(rank_fn)
            results[name][level] = run_variant(agent, samples, catalog_ids, categories, products, level)

    print(f"{len(samples)} sessions, levels {args.levels}\n")
    print("TechnicalScore")
    print("-" * 60)
    print(f"{'variant':16s}" + "".join(f"L{lv:<9d}" for lv in args.levels))
    for name in variants:
        row = "".join(f"{results[name][lv]['recommended_technical_score']:<10.4f}"
                       for lv in args.levels)
        print(f"{name:16s}{row}")

    print("\nHitRate@10")
    print("-" * 60)
    print(f"{'variant':16s}" + "".join(f"L{lv:<9d}" for lv in args.levels))
    for name in variants:
        row = "".join(f"{results[name][lv]['hit_rate_at_10']:<10.4f}"
                       for lv in args.levels)
        print(f"{name:16s}{row}")


if __name__ == "__main__":
    main()

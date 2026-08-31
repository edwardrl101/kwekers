"""Wires BM25Route + DenseRoute + ConditionalRescorer into the evaluator's
Agent interface and runs the official local evaluator against it.

Reference: this architecture WITHOUT rescoring (pool returned as-is) scored
TechnicalScore 0.8553 (Hit 0.995, MRR 0.634, MTTC 2.62). Two lines carry most
of that score and are easy to break:

  1. state["shown"].clear() on override - without it, override sessions
     collapse from 0.900 to 0.133. Hits before the override turn are ignored
     by the evaluator (override_applied gates the hit check - see CLAUDE.md
     mechanic 5), but the target still gets added to state["shown"], so once
     override_applied flips true the agent can never recommend it again.
  2. NOT wiping state["text"] on override - the target product never changes
     across an override, only which constraints have been disclosed does.
     Wiping loses every constraint disclosed pre-override; keeping the full
     text reaches 1.000 on override, wiping recovers only 0.333.

If a run comes in far below 0.8553, the loop is broken, not the rescoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.rescore import ConditionalRescorer  # noqa: E402
from src.buckets import extract_category_phrase  # noqa: E402
from src.retrieval import (  # noqa: E402
    BM25Route,
    DenseRoute,
    load_catalog,
    messages_to_bm25_query,
)

TOP_K = 10
OVERRIDE_PHRASE = "ignore my earlier preference"
SWEEP_TURN_LIMITS = (0, 1, 2, 3, 10)
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


class Agent:
    def __init__(
        self,
        catalog_path: str = "data/catalog.jsonl",
        turn_limit: int = 1,
        catalog: dict | None = None,
        bm25: BM25Route | None = None,
        dense: DenseRoute | None = None,
    ) -> None:
        self.catalog = catalog if catalog is not None else load_catalog(catalog_path)
        self.bm25 = bm25 if bm25 is not None else BM25Route(self.catalog)
        self.dense = dense if dense is not None else DenseRoute(self.catalog)
        self.rescorer = ConditionalRescorer(self.dense, turn_limit=turn_limit)
        self._state: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.rescorer.reset(session_id)
        self._state[session_id] = {"text": [], "shown": set(), "category": None}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._state[session_id]

        if OVERRIDE_PHRASE in user_message.lower():
            state["shown"].clear()  # override blackout is over, allow re-recommendation

        state["text"].append(user_message)  # never wipe - target doesn't change across override
        category = extract_category_phrase(user_message)
        if category:
            state["category"] = category
        joined = messages_to_bm25_query(
            state.get("category"), state["text"], fallback=user_message
        )

        pool = [asin for asin, _ in self.bm25.query(joined, limit=500)]
        ranked = self.rescorer.rescore(session_id, pool, joined, turn)

        picks = [asin for asin in ranked if asin not in state["shown"]][:top_k]
        state["shown"].update(picks)

        return {
            "message": "Here are some options based on what you've told me so far.",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": asin} for asin in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def print_result(label: str, result: dict) -> None:
    per = {name: m["hit_rate_at_10"] for name, m in result["scenario_metrics"].items()}
    tail = "  ".join(f"{s[:5]} {per.get(s, 0.0):.3f}" for s in SCENARIOS)
    print(f"{label:>10s}  {result['recommended_technical_score']:.4f}  "
          f"{result['hit_rate_at_10']:.4f}  {result['mrr']:.4f}  "
          f"{result['mttc']:.4f}   {tail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                         help="sweep turn_limit in (0, 1, 2, 3, 10)")
    args = parser.parse_args()

    catalog_path = "data/catalog.jsonl"
    dataset_path = "data/public_set.jsonl"

    catalog_ids, categories, products = catalog_index(catalog_path)
    samples = load_jsonl(dataset_path)

    # Build routes ONCE - BM25 build is a couple seconds, the dense cache load
    # is a couple more, and neither depends on turn_limit.
    catalog = load_catalog(catalog_path)
    bm25 = BM25Route(catalog)
    dense = DenseRoute(catalog)

    if args.sweep:
        print(f"{'limit':>10s}  {'Tech':>6s}  {'Hit':>6s}  {'MRR':>6s}  {'MTTC':>6s}")
        for turn_limit in SWEEP_TURN_LIMITS:
            agent = Agent(catalog_path, turn_limit=turn_limit,
                          catalog=catalog, bm25=bm25, dense=dense)
            result = evaluate(agent, samples, catalog_ids, categories, products)
            print_result(str(turn_limit), result)
        return

    agent = Agent(catalog_path, catalog=catalog, bm25=bm25, dense=dense)
    result = evaluate(agent, samples, catalog_ids, categories, products)

    print(f"TechnicalScore {result['recommended_technical_score']:.4f}")
    print(f"HitRate@10     {result['hit_rate_at_10']:.4f}")
    print(f"MRR            {result['mrr']:.4f}")
    print(f"MTTC           {result['mttc']:.4f}")
    print()
    print("Per scenario")
    print("-" * 60)
    for name, metrics in sorted(result["scenario_metrics"].items()):
        print(f"{name:18s} hit={metrics['hit_rate_at_10']:.3f}  "
              f"mrr={metrics['mrr']:.3f}  mttc={metrics['mttc']:.3f}")


if __name__ == "__main__":
    main()

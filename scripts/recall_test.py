"""Recall@K harness.

Your metric is NOT precision. Reranking downstream can reorder 100 candidates
but cannot conjure a product you failed to retrieve. So the question is only:
how often is the target somewhere in my top K?

This replays the evaluator's own message generation so you are testing against
the exact strings the real harness will send. It reads evaluator code, never
modifies it.

    python3 scripts/recall_test.py                 # bm25 + ngram
    python3 scripts/recall_test.py --dense         # add the dense route
    python3 scripts/recall_test.py --turns 3       # simulate a 3-turn drain
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields,
)
from src.retrieval import BM25Route, NgramRoute, load_catalog  # noqa: E402


def build_queries(samples, categories, products, turns: int):
    """Reconstruct what the customer actually says, turn by turn.

    We simulate the agent always asking ask_attribute='other', which bypasses
    classify_constraint() and drains up to 2 undisclosed constraints per turn.
    That is the information ceiling, so recall measured here is the best your
    route can do given perfect questioning.
    """
    cases = []
    for sample in samples:
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        target = str(sample["ground_truth"]["parent_asin"])
        disclosed: set[str] = set()
        boundary_used = False
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed)
        accumulated = [message]
        for _ in range(turns - 1):
            message, boundary_used = customer_reply(
                effective, "other", disclosed, boundary_used)
            accumulated.append(message)
        cases.append({
            "target": target,
            "scenario": sample["scenario_type"],
            "query": " ".join(accumulated),
        })
    return cases


def score_route(route, cases, ks=(10, 50, 100)) -> dict:
    limit = max(ks)
    hits = {k: 0 for k in ks}
    per_scenario = defaultdict(lambda: {"n": 0, "hit": 0})
    latencies = []
    for case in cases:
        start = time.perf_counter()
        ranked = [asin for asin, _ in route.query(case["query"], limit)]
        latencies.append((time.perf_counter() - start) * 1000)
        position = ranked.index(case["target"]) + 1 if case["target"] in ranked else None
        for k in ks:
            if position is not None and position <= k:
                hits[k] += 1
        bucket = per_scenario[case["scenario"]]
        bucket["n"] += 1
        bucket["hit"] += int(position is not None and position <= limit)
    total = len(cases)
    return {
        "recall": {k: hits[k] / total for k in ks},
        "p50_ms": sorted(latencies)[len(latencies) // 2],
        "scenarios": {
            name: value["hit"] / value["n"] for name, value in sorted(per_scenario.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    catalog = load_catalog(args.catalog)
    cases = build_queries(samples, categories, products, args.turns)
    print(f"{len(cases)} sessions, simulating {args.turns} turn(s) of disclosure\n")

    routes = [("bm25", BM25Route), ("ngram", NgramRoute)]
    if args.dense:
        from src.retrieval import DenseRoute
        routes.append(("dense", DenseRoute))

    print(f"{'route':8s} {'R@10':>7s} {'R@50':>7s} {'R@100':>7s} "
          f"{'build':>8s} {'p50':>8s}")
    print("-" * 50)
    results = {}
    for name, cls in routes:
        start = time.perf_counter()
        route = cls(catalog)
        build = time.perf_counter() - start
        stats = score_route(route, cases)
        results[name] = stats
        r = stats["recall"]
        print(f"{name:8s} {r[10]:7.3f} {r[50]:7.3f} {r[100]:7.3f} "
              f"{build:7.1f}s {stats['p50_ms']:7.1f}ms")

    print("\nRecall@100 by scenario")
    print("-" * 50)
    names = list(results)
    scenarios = sorted({s for stats in results.values() for s in stats["scenarios"]})
    print(f"{'scenario':16s}" + "".join(f"{n:>10s}" for n in names))
    for scenario in scenarios:
        row = "".join(f"{results[n]['scenarios'].get(scenario, 0):10.3f}" for n in names)
        print(f"{scenario:16s}{row}")


if __name__ == "__main__":
    main()
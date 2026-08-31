"""One harness, four subcommands. Replaces recall_test.py + day2.py.

Shared: one catalog load, one query-construction path (build_queries), one
recall function. Routes are cached across subcommands within a run since
ngram build takes ~20s and dense loads a 38MB cache.

    python3 scripts/experiments.py routes    # R@10/50/100 per route
    python3 scripts/experiments.py depth     # K sweep: 10/100/500/2000
    python3 scripts/experiments.py fusion    # RRF weightings + union ceiling
    python3 scripts/experiments.py rescore   # reorder a pool, measure top-10

All subcommands print a per-scenario breakdown (CLAUDE.md rule 3: the average
hides everything). --split {tune,holdout,all} selects 140/60/200 sessions,
default tune. Never tune decisions on the holdout split.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields,
)
from src.dialog import route_scenario  # noqa: E402
from src.retrieval import (  # noqa: E402
    BM25Route, DenseRoute, NgramRoute, extract_constraints, load_catalog,
)

TUNE_SIZE = 140  # of 200 public sessions; CLAUDE.md: 140 tune / 60 holdout


# ------------------------------------------------------------------- dataset

def load_split(split: str, dataset: str = "data/public_set.jsonl") -> list[dict]:
    """Stratified 140/60 split preserving the official scenario mix.

    A contiguous samples[:140] split is NOT balanced: it puts 46.7% buying in
    the holdout vs 37.1% in tune. Buying is the easiest scenario and browsing
    the hardest, so the holdout would score higher for compositional reasons
    and mask real overfitting. Stratify within each scenario instead.
    """
    samples = load_jsonl(dataset)
    if split == "all":
        return samples
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_scenario[sample["scenario_type"]].append(sample)
    fraction = TUNE_SIZE / len(samples)
    tune, holdout = [], []
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        cut = round(len(group) * fraction)
        tune.extend(group[:cut])
        holdout.extend(group[cut:])
    return tune if split == "tune" else holdout


# --------------------------------------------------------- query construction

def build_queries(samples, categories, products, turns: int) -> list[dict]:
    """Reconstruct what the customer actually says, turn by turn.

    Simulates the agent always asking ask_attribute='other', which bypasses
    classify_constraint() and drains up to 2 undisclosed constraints per turn
    - the information ceiling, so recall measured here is the best a route can
    do given perfect questioning.
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


# --------------------------------------------------------------------- recall

def recall_at_k(ranked_per_case: list[list[str]], cases: list[dict], k: int):
    """Fraction of sessions where target is in top k, overall and per scenario."""
    hit = 0
    per = defaultdict(lambda: [0, 0])
    for ranked, case in zip(ranked_per_case, cases):
        found = case["target"] in ranked[:k]
        hit += found
        bucket = per[case["scenario"]]
        bucket[0] += 1
        bucket[1] += found
    overall = hit / len(cases) if cases else 0.0
    per_scenario = {s: v[1] / v[0] for s, v in sorted(per.items())}
    return overall, per_scenario


def print_row(label: str, ranked_per_case, cases, k: int) -> None:
    overall, per = recall_at_k(ranked_per_case, cases, k)
    scenarios = ("buying", "browsing", "intent_override", "boundary")
    tail = "  ".join(f"{s[:5]} {per.get(s, 0):.3f}" for s in scenarios)
    print(f"{label:26s} R@{k:<4d} {overall:.3f}   {tail}")


# ---------------------------------------------------------------- route cache

class RouteCache:
    """Builds each route lazily, once per run, shared across subcommands."""

    def __init__(self, catalog: dict[str, dict]) -> None:
        self.catalog = catalog
        self._routes: dict[str, object] = {}
        self._build_times: dict[str, float] = {}

    def get(self, name: str):
        if name not in self._routes:
            cls = {"bm25": BM25Route, "ngram": NgramRoute, "dense": DenseRoute}[name]
            start = time.perf_counter()
            self._routes[name] = cls(self.catalog)
            self._build_times[name] = time.perf_counter() - start
        return self._routes[name]

    def build_time(self, name: str) -> float:
        return self._build_times.get(name, 0.0)


def pooled(route, cases: list[dict], limit: int) -> list[list[str]]:
    return [[a for a, _ in route.query(c["query"], limit)] for c in cases]


# ------------------------------------------------------------------- rrf

def rrf(lists_with_weights, k: int = 60, limit: int = 500) -> list[str]:
    """Reciprocal Rank Fusion. Combines by RANK, so route scores need no scaling."""
    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in lists_with_weights:
        if weight == 0:
            continue
        for position, asin in enumerate(ranked):
            scores[asin] += weight / (k + position + 1)
    return [a for a, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:limit]


# ------------------------------------------------------------------ commands

def cmd_routes(cases: list[dict], routes: RouteCache, args) -> None:
    """R@10/50/100 per route, latency, per-scenario at R@100."""
    names = ["bm25", "ngram"] + (["dense"] if args.dense else [])
    print(f"{'route':8s} {'R@10':>7s} {'R@50':>7s} {'R@100':>7s} "
          f"{'build':>8s} {'p50':>8s}")
    print("-" * 55)
    results = {}
    for name in names:
        route = routes.get(name)
        latencies = []
        limit = 100
        ranked_per_case = []
        for case in cases:
            start = time.perf_counter()
            ranked = [a for a, _ in route.query(case["query"], limit)]
            latencies.append((time.perf_counter() - start) * 1000)
            ranked_per_case.append(ranked)
        r10, _ = recall_at_k(ranked_per_case, cases, 10)
        r50, _ = recall_at_k(ranked_per_case, cases, 50)
        r100, per100 = recall_at_k(ranked_per_case, cases, 100)
        results[name] = (ranked_per_case, per100)
        p50 = sorted(latencies)[len(latencies) // 2]
        print(f"{name:8s} {r10:7.3f} {r50:7.3f} {r100:7.3f} "
              f"{routes.build_time(name):7.1f}s {p50:7.1f}ms")

    print("\nRecall@100 by scenario")
    print("-" * 55)
    scenarios = sorted({s for _, per in results.values() for s in per})
    print(f"{'scenario':16s}" + "".join(f"{n:>10s}" for n in names))
    for scenario in scenarios:
        row = "".join(f"{results[n][1].get(scenario, 0):10.3f}" for n in names)
        print(f"{scenario:16s}{row}")


def cmd_depth(cases: list[dict], routes: RouteCache, args) -> None:
    """K sweep 10/100/500/2000 for bm25 (rule 1: separate 'find' from 'rank')."""
    ks = (10, 100, 500, 2000)
    route = routes.get("bm25")
    pool = pooled(route, cases, max(ks))
    print(f"BM25 depth sweep ({len(cases)} sessions)")
    print("-" * 70)
    for k in ks:
        print_row(f"K={k}", pool, cases, k)


def cmd_fusion(cases: list[dict], routes: RouteCache, args) -> None:
    """RRF weightings + union ceiling (rule 2: measure a ceiling before tuning)."""
    limit = 100
    bm_lists = pooled(routes.get("bm25"), cases, 500)
    ng_lists = pooled(routes.get("ngram"), cases, 500)
    lists = {"bm25": bm_lists, "ngram": ng_lists}
    if args.dense:
        lists["dense"] = pooled(routes.get("dense"), cases, 500)

    print(f"Route recall @100 ({len(cases)} sessions)")
    print("-" * 70)
    for name, lst in lists.items():
        print_row(f"{name} alone", lst, cases, limit)
    print()

    dn_lists = lists.get("dense", [[] for _ in cases])
    combos = {
        "rrf bm25+ngram equal": (1.0, 1.0, 0.0),
        "rrf bm25+ngram heavy": (3.0, 1.0, 0.0),
        "rrf equal (all 3)":    (1.0, 1.0, 1.0),
        "rrf bm25-heavy":       (3.0, 1.0, 1.0),
        "rrf bm25+dense":       (3.0, 0.0, 1.5),
        "rrf bm25 only":        (1.0, 0.0, 0.0),
    }
    print("Fused")
    print("-" * 70)
    for label, (wb, wn, wd) in combos.items():
        if wd and "dense" not in lists:
            continue
        fused = [rrf([(b, wb), (n, wn), (d, wd)])
                 for b, n, d in zip(bm_lists, ng_lists, dn_lists)]
        print_row(label, fused, cases, limit)

    all_lists = [bm_lists, ng_lists] + ([dn_lists] if "dense" in lists else [])
    union = [list(dict.fromkeys(sum((lst[i][:limit] for lst in all_lists), [])))
             for i in range(len(cases))]
    print()
    print_row("UNION ceiling", union, cases, 10 ** 6)


def cmd_rescore(cases: list[dict], routes: RouteCache, args) -> None:
    """Reorder a BM25 pool, measure whether top-10 improves (rule 4: verify,
    don't assume rescoring helps - char n-gram rescoring HURTS, see CLAUDE.md)."""
    catalog = routes.catalog
    bm = routes.get("bm25")
    bm_lists = pooled(bm, cases, 500)

    print(f"Rescoring a BM25 top-500 pool ({len(cases)} sessions)")
    print("-" * 70)
    print_row("pool only (baseline)", bm_lists, cases, 10)

    if args.dense:
        dn = routes.get("dense")
        dense_pos = {a: i for i, a in enumerate(dn.asins)}
        rescored = []
        for pool, case in zip(bm_lists, cases):
            idx = [dense_pos[a] for a in pool if a in dense_pos]
            keep = [a for a in pool if a in dense_pos]
            if not idx:
                rescored.append(pool)
                continue
            q = dn.model.encode([case["query"]], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                s = dn.embeddings[idx] @ q
            rescored.append([keep[i] for i in np.argsort(-s)])
        print_row("dense rescored", rescored, cases, 10)

    def features(asin: str, constraints: list[str], rank: int) -> float:
        p = catalog[asin]
        title = str(p.get("title") or "").lower()
        score = -rank * 0.01
        tokens = {t for c in constraints for t in c.lower().split() if len(t) > 3}
        if tokens:
            score += 2.0 * sum(t in title for t in tokens) / len(tokens)
        score -= 0.002 * len(title.split())
        try:
            score += 0.1 * float(p.get("average_rating") or 0) / 5.0
        except (TypeError, ValueError):
            pass
        return score

    feat = []
    for pool, case in zip(bm_lists, cases):
        cons = extract_constraints(case["query"])
        rank_of = {a: i for i, a in enumerate(pool)}
        feat.append(sorted(pool, key=lambda a, c=cons, r=rank_of: -features(
            a, c, r[a])))
    print_row("feature rescored", feat, cases, 10)

    if args.dense:
        # Route from the message text, not case["scenario"] - the label isn't
        # available at inference time. route_scenario(turn=1) on a single-turn
        # query never returns "boundary"/"browsing" (those need turn>=2 replay
        # against prior state) or "override" (the function returns
        # "intent_override"); at turn 1 it only distinguishes "buying",
        # "intent_override", and the provisional "browsing_or_boundary".
        cond = []
        for pool, dense_ordered, case in zip(bm_lists, rescored, cases):
            detected = route_scenario(case["query"], turn=1)
            cond.append(pool if detected in ("buying", "intent_override")
                        else dense_ordered)
        print_row("conditional (vague only)", cond, cases, 10)


COMMANDS = {
    "routes": cmd_routes,
    "depth": cmd_depth,
    "fusion": cmd_fusion,
    "rescore": cmd_rescore,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    samples = load_split(args.split, args.dataset)
    _, categories, products = catalog_index(args.catalog)
    catalog = load_catalog(args.catalog)
    cases = build_queries(samples, categories, products, args.turns)
    print(f"{len(cases)} sessions ({args.split}), {args.turns} turn(s) of disclosure\n")

    routes = RouteCache(catalog)
    COMMANDS[args.command](cases, routes, args)


if __name__ == "__main__":
    main()

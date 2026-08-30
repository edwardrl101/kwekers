"""Reversible near-duplicate suppression experiment for ranked product pools."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SCOPES = ("top10", "full")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "gold", "silver",
}
_GENDERS = {"men", "mens", "women", "womens", "boys", "girls", "unisex"}
_SIZES = {"xxs", "xs", "small", "medium", "large", "xl", "xxl", "xxxl"}


@lru_cache(maxsize=100_000)
def normalize_title(title: str) -> str:
    """Conservatively normalize title punctuation, case, and whitespace."""
    if not isinstance(title, str):
        return ""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(_NON_WORD.sub(" ", normalized).split())


@lru_cache(maxsize=100_000)
def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(normalize_title(title).split())


def title_similarity(title_a: str, title_b: str) -> float:
    """Return token Jaccard similarity in [0, 1]; empty titles never match."""
    tokens_a = _title_tokens(title_a)
    tokens_b = _title_tokens(title_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@lru_cache(maxsize=100_000)
def variant_tokens(title: str) -> frozenset[str]:
    """Extract a small, conservative set of variant-distinguishing tokens."""
    tokens = _title_tokens(title)
    result = set(tokens & (_COLORS | _GENDERS | _SIZES))
    result.update(token for token in tokens if any(character.isdigit() for character in token))
    return frozenset(result)


def _is_duplicate(
    title_a: str,
    title_b: str,
    threshold: float,
    variant_sensitive: bool,
) -> bool:
    if title_similarity(title_a, title_b) < threshold:
        return False
    if variant_sensitive and variant_tokens(title_a) != variant_tokens(title_b):
        return False
    return True


def _title(catalog: dict[str, dict], asin: str) -> str:
    product = catalog.get(str(asin))
    if not isinstance(product, dict):
        return ""
    value = product.get("title")
    return value if isinstance(value, str) else ""


def suppress_duplicates(
    pool: list[str],
    catalog: dict[str, dict],
    threshold: float = 0.8,
    variant_sensitive: bool = False,
) -> list[str]:
    """Greedily suppress similar titles while preserving rank 1 and order."""
    if not pool:
        return []
    kept: list[str] = []
    kept_titles: list[str] = []
    for asin in pool:
        candidate = str(asin)
        title = _title(catalog, candidate)
        if not kept:
            kept.append(candidate)
            kept_titles.append(title)
            continue
        duplicate = bool(title) and any(
            _is_duplicate(title, prior_title, threshold, variant_sensitive)
            for prior_title in kept_titles
            if prior_title
        )
        if not duplicate:
            kept.append(candidate)
            kept_titles.append(title)
    return kept


def diverse_top_k(
    pool: list[str],
    catalog: dict[str, dict],
    k: int = 10,
    threshold: float = 0.8,
    scope: str = "full",
    variant_sensitive: bool = False,
) -> list[str]:
    """Build a diverse Top-K and backfill suppressed candidates when needed.

    ``top10`` greedily selects the final slots directly from the ranked pool.
    ``full`` suppresses the wider pool first and then selects its first K.
    Both retain original ordering and backfill from the original pool.
    """
    if k <= 0 or not pool:
        return []
    if scope not in SCOPES:
        raise ValueError(f"unsupported suppression scope: {scope!r}")

    if scope == "full":
        preferred = suppress_duplicates(
            pool, catalog, threshold, variant_sensitive=variant_sensitive
        )
    else:
        preferred = []
        preferred_titles: list[str] = []
        for asin in pool:
            candidate = str(asin)
            title = _title(catalog, candidate)
            duplicate = bool(title) and any(
                _is_duplicate(title, prior, threshold, variant_sensitive)
                for prior in preferred_titles
                if prior
            )
            if not duplicate:
                preferred.append(candidate)
                preferred_titles.append(title)
                if len(preferred) == k:
                    break

    result: list[str] = []
    seen: set[str] = set()
    for asin in preferred:
        candidate = str(asin)
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
            if len(result) == k:
                return result
    for asin in pool:
        candidate = str(asin)
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
            if len(result) == k:
                break
    return result


# Backwards-compatible name used by the first version of the experiment.
deduplicated_top_k = diverse_top_k


def duplicate_pairs(
    pool: list[str],
    catalog: dict[str, dict],
    threshold: float = 0.8,
    variant_sensitive: bool = False,
) -> list[tuple[str, str, float]]:
    """Return every near-duplicate pair in a pool for offline diagnostics."""
    pairs: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(pool):
        left_title = _title(catalog, left)
        if not left_title:
            continue
        for right in pool[left_index + 1 :]:
            right_title = _title(catalog, right)
            if not right_title:
                continue
            similarity = title_similarity(left_title, right_title)
            if similarity >= threshold and (
                not variant_sensitive
                or variant_tokens(left_title) == variant_tokens(right_title)
            ):
                pairs.append((str(left), str(right), similarity))
    return pairs


def count_duplicate_pairs(
    pool: list[str], catalog: dict[str, dict], threshold: float = 0.8
) -> int:
    return len(duplicate_pairs(pool, catalog, threshold))


def load_catalog(path: str | Path = "data/catalog.jsonl") -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product.get("parent_asin", "")).strip()
            if asin and asin not in catalog:
                catalog[asin] = product
    return catalog


class _EvaluationAgent:
    """Diagnostic adapter; it does not modify the production Agent class."""

    def __init__(
        self,
        base_agent: object,
        samples: list[dict],
        threshold: float | None = None,
        scope: str = "top10",
        candidate_cache: dict[str, list[str]] | None = None,
        variant_sensitive: bool = False,
    ) -> None:
        self.base = base_agent
        self.samples = samples
        self.threshold = threshold
        self.scope = scope
        self.candidate_cache = candidate_cache if candidate_cache is not None else {}
        self.variant_sensitive = variant_sensitive
        self.sample_index = 0
        self.session_samples: dict[str, dict] = {}
        self.turn_one_lists: list[list[str]] = []
        self.turn_one_outputs: list[list[str]] = []
        self.target_suppression_events: set[str] = set()
        self.target_suppression_cases: list[dict] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        sample = self.samples[self.sample_index]
        self.sample_index += 1
        self.session_samples[str(session_id)] = sample
        self.base.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        key = str(session_id)
        session = self.base._sessions.get(
            key, {"seed_key": "missing-session", "user_profile": {}}
        )
        message = str(user_message or "")
        cached = self.candidate_cache.get(message)
        if cached is None:
            cached = self.base._route_candidates(session, message, turn)
            self.candidate_cache[message] = list(cached)
        candidates = list(cached)
        original = self.base._random_fill(candidates, session, turn)
        if turn == 1:
            self.turn_one_lists.append(original)

        if self.threshold is None:
            selected = original
        else:
            diverse = diverse_top_k(
                candidates,
                self.base._catalog,
                k=top_k,
                threshold=self.threshold,
                scope=self.scope,
                variant_sensitive=self.variant_sensitive,
            )
            selected = self.base._random_fill(diverse, session, turn)
            sample = self.session_samples.get(key, {})
            target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
            if target in original and target not in selected:
                sample_id = str(sample.get("sample_id", key))
                self.target_suppression_events.add(sample_id)
                target_title = _title(self.base._catalog, target)
                suppressor = next(
                    (
                        asin
                        for asin in selected
                        if _is_duplicate(
                            target_title,
                            _title(self.base._catalog, asin),
                            self.threshold,
                            self.variant_sensitive,
                        )
                    ),
                    None,
                )
                if suppressor is not None and not any(
                    case["sample_id"] == sample_id
                    for case in self.target_suppression_cases
                ):
                    other_title = _title(self.base._catalog, suppressor)
                    self.target_suppression_cases.append(
                        {
                            "sample_id": sample_id,
                            "query": message,
                            "target_asin": target,
                            "target_title": target_title,
                            "suppressor_asin": suppressor,
                            "suppressor_title": other_title,
                            "similarity": title_similarity(target_title, other_title),
                        }
                    )

        if turn == 1:
            self.turn_one_outputs.append(selected[:top_k])

        return {
            "message": "I am refining the shortlist. What else should I consider?",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": asin} for asin in selected[:top_k]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def _near_unique_count(pool: list[str], catalog: dict[str, dict], threshold: float) -> int:
    return len(suppress_duplicates(pool, catalog, threshold))


def prevalence_report(
    pools: list[list[str]], catalog: dict[str, dict], threshold: float = 0.8
) -> dict:
    counts = [count_duplicate_pairs(pool[:10], catalog, threshold) for pool in pools]
    unique_counts = [_near_unique_count(pool[:10], catalog, threshold) for pool in pools]
    return {
        "sessions": len(pools),
        "average_pairs": statistics.fmean(counts) if counts else 0.0,
        "median_pairs": statistics.median(counts) if counts else 0.0,
        "sessions_ge_1": sum(count >= 1 for count in counts),
        "sessions_ge_2": sum(count >= 2 for count in counts),
        "maximum_pairs": max(counts, default=0),
        "average_unique": statistics.fmean(unique_counts) if unique_counts else 0.0,
    }


def _movement_counts(before: dict, after: dict) -> dict[str, int | float]:
    counts: dict[str, int | float] = {
        "target_suppressed": 0,
        "moved_up": 0,
        "moved_down": 0,
        "unchanged": 0,
    }
    improvements: list[int] = []
    losses: list[int] = []
    after_by_id = {row["sample_id"]: row for row in after["sessions"]}
    for prior in before["sessions"]:
        current = after_by_id[prior["sample_id"]]
        if prior["hit"] and not current["hit"]:
            counts["target_suppressed"] += 1
        elif prior["best_rank"] is not None and current["best_rank"] is not None:
            if current["best_rank"] < prior["best_rank"]:
                counts["moved_up"] += 1
                improvements.append(prior["best_rank"] - current["best_rank"])
            elif current["best_rank"] > prior["best_rank"]:
                counts["moved_down"] += 1
                losses.append(current["best_rank"] - prior["best_rank"])
            else:
                counts["unchanged"] += 1
        else:
            counts["unchanged"] += 1
    counts["average_rank_improvement"] = (
        statistics.fmean(improvements) if improvements else 0.0
    )
    counts["average_rank_loss"] = statistics.fmean(losses) if losses else 0.0
    return counts


def evaluate_threshold(
    base_agent: object,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    threshold: float | None,
    scope: str,
    candidate_cache: dict[str, list[str]] | None = None,
    variant_sensitive: bool = False,
) -> tuple[dict, _EvaluationAgent]:
    from evaluator.local_evaluator import evaluate

    adapter = _EvaluationAgent(
        base_agent,
        samples,
        threshold=threshold,
        scope=scope,
        candidate_cache=candidate_cache,
        variant_sensitive=variant_sensitive,
    )
    return evaluate(adapter, samples, catalog_ids, categories, products), adapter


def _metric_row(
    scope: str,
    threshold: float,
    result: dict,
    unique: float,
    target_suppressed: int,
) -> dict:
    return {
        "scope": scope,
        "threshold": threshold,
        "score": result["recommended_technical_score"],
        "hit": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "average_unique": unique,
        "target_suppressed": target_suppressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate title near-duplicate suppression")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", default="data/public_set.jsonl")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    from evaluator.local_evaluator import catalog_index, load_jsonl
    from starter.agent import Agent

    samples = load_jsonl(args.sessions)
    catalog_ids, categories, products = catalog_index(args.catalog)
    base_agent = Agent(args.catalog)
    candidate_cache: dict[str, list[str]] = {}
    baseline, baseline_adapter = evaluate_threshold(
        base_agent,
        samples,
        catalog_ids,
        categories,
        products,
        None,
        "top10",
        candidate_cache,
    )
    prevalence = prevalence_report(baseline_adapter.turn_one_lists, products, 0.80)

    print("=== Near-Duplicate Suppression ===\n")
    print(f"Public sessions: {len(samples):,}")
    print("\nDuplicate prevalence @ threshold 0.80 (turn-1 Top 10)")
    print(f"Sessions with >=1 pair: {prevalence['sessions_ge_1']:,} / {len(samples):,}")
    print(f"Sessions with >=2 pairs: {prevalence['sessions_ge_2']:,} / {len(samples):,}")
    print(f"Average duplicate pairs: {prevalence['average_pairs']:.3f}")
    print(f"Median duplicate pairs:  {prevalence['median_pairs']:.1f}")
    print(f"Maximum duplicate pairs: {prevalence['maximum_pairs']:,}")
    print(f"Average unique titles:   {prevalence['average_unique']:.3f}")

    examples: list[tuple[str, str, float]] = []
    for pool in baseline_adapter.turn_one_lists:
        examples.extend(duplicate_pairs(pool[:10], products, 0.80))
        if len(examples) >= args.examples:
            break
    if examples:
        print("\nExample pairs")
        for left, right, similarity in examples[: args.examples]:
            print(f"- {similarity:.3f}: {_title(products, left)!r} <> {_title(products, right)!r}")

    print("\nBaseline")
    print(f"Score: {baseline['recommended_technical_score']:.6f}")
    print(f"Hit:   {baseline['hit_rate_at_10']:.6f}")
    print(f"MRR:   {baseline['mrr']:.6f}")
    print(f"MTTC:  {baseline['mttc']:.6f}")

    rows: list[dict] = []
    results: dict[tuple[str, float], tuple[dict, _EvaluationAgent]] = {}
    for scope in SCOPES:
        for threshold in THRESHOLDS:
            result, adapter = evaluate_threshold(
                base_agent,
                samples,
                catalog_ids,
                categories,
                products,
                threshold,
                scope,
                candidate_cache,
            )
            unique = prevalence_report(adapter.turn_one_outputs, products, threshold)["average_unique"]
            rows.append(
                _metric_row(
                    scope,
                    threshold,
                    result,
                    unique,
                    len(adapter.target_suppression_events),
                )
            )
            results[(scope, threshold)] = (result, adapter)

    print("\nThreshold sweep")
    print(
        "Scope   Threshold  Score     Hit       MRR       MTTC    "
        "AvgUnique@10  TargetSuppressed"
    )
    for row in rows:
        print(
            f"{row['scope']:<7} {row['threshold']:<10.2f} {row['score']:<9.6f} "
            f"{row['hit']:<9.6f} {row['mrr']:<9.6f} {row['mttc']:<7.3f} "
            f"{row['average_unique']:<13.3f} {row['target_suppressed']}"
        )

    best = max(rows, key=lambda row: (row["score"], row["mrr"], row["threshold"]))
    best_result, best_adapter = results[(best["scope"], best["threshold"])]
    movement = _movement_counts(baseline, best_result)
    print("\nBest measured configuration")
    print(f"Scope: {best['scope']}")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Score delta: {best['score'] - baseline['recommended_technical_score']:+.6f}")
    print(f"MRR delta:   {best['mrr'] - baseline['mrr']:+.6f}")
    print(f"Direct target-suppression events: {len(best_adapter.target_suppression_events):,}")
    for label, count in movement.items():
        rendered = f"{count:.3f}" if isinstance(count, float) else f"{count:,}"
        print(f"{label.replace('_', ' ').title()}: {rendered}")

    harmed_result, harmed_adapter = results[("top10", 0.80)]
    harmed_movement = _movement_counts(baseline, harmed_result)
    print("\nThreshold 0.80 failure analysis")
    print(f"Direct target-suppression events: {len(harmed_adapter.target_suppression_events):,}")
    for label, count in harmed_movement.items():
        rendered = f"{count:.3f}" if isinstance(count, float) else f"{count:,}"
        print(f"{label.replace('_', ' ').title()}: {rendered}")
    for case in harmed_adapter.target_suppression_cases[: args.examples]:
        print(
            f"- {case['sample_id']} query={case['query']!r}\n"
            f"  target {case['target_asin']}: {case['target_title']!r}\n"
            f"  suppressed by {case['suppressor_asin']} "
            f"(similarity={case['similarity']:.3f}): "
            f"{case['suppressor_title']!r}"
        )

    variant_result, variant_adapter = evaluate_threshold(
        base_agent,
        samples,
        catalog_ids,
        categories,
        products,
        0.80,
        "top10",
        candidate_cache,
        variant_sensitive=True,
    )
    variant_movement = _movement_counts(baseline, variant_result)
    print("\nVariant-sensitive threshold 0.80")
    print(f"Score: {variant_result['recommended_technical_score']:.6f}")
    print(f"Hit:   {variant_result['hit_rate_at_10']:.6f}")
    print(f"MRR:   {variant_result['mrr']:.6f}")
    print(f"MTTC:  {variant_result['mttc']:.6f}")
    print(f"Direct target-suppression events: {len(variant_adapter.target_suppression_events):,}")
    for label, count in variant_movement.items():
        rendered = f"{count:.3f}" if isinstance(count, float) else f"{count:,}"
        print(f"{label.replace('_', ' ').title()}: {rendered}")

    print("\nPer-scenario baseline -> best MRR")
    for scenario, metrics in baseline["scenario_metrics"].items():
        after = best_result["scenario_metrics"][scenario]
        print(f"{scenario}: {metrics['mrr']:.6f} -> {after['mrr']:.6f}")


if __name__ == "__main__":
    main()

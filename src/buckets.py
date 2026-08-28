from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    behavior_for,
    coarse_category as evaluator_coarse_category,
    initial_message,
    intent_card,
)


BUCKET_BANDS = ("<=10", "11-50", "51-500", ">500") #bucket-size groups
COVERAGE_CUTOFFS = (10, 50, 200)


def _load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL with a useful filename and line number on malformed input."""
    records: list[dict] = []
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {source} at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {source} at line {line_number}")
            records.append(record)
    return records


def load_catalog(path: str = "data/catalog.jsonl") -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for product in _load_jsonl(path):
        if "parent_asin" not in product or not str(product["parent_asin"]).strip():
            raise ValueError("Every catalog product must have a non-empty parent_asin")
        parent_asin = str(product["parent_asin"])
        if parent_asin in catalog:
            raise ValueError(f"Duplicate parent_asin in catalog: {parent_asin}")
        catalog[parent_asin] = product
    return catalog


def normalize_phrase(text: str) -> str:
    """Return the case- and whitespace-insensitive bucket lookup key."""
    return " ".join(text.lower().strip().split())


def coarse_category(product: dict) -> str:
    """Return the exact coarse category produced by the official evaluator."""
    values = [str(value) for value in product.get("categories") or []]
    return evaluator_coarse_category(values)


OPENING_PATTERNS = (
    re.compile(
        r"^\s*I'm\s+looking\s+for\s+(.+?)\.\s+"
        r"A\s+key\s+requirement\s+is:\s+.+?\.\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*I'm\s+looking\s+for\s+(.+?),\s+"
        r"but\s+I'm\s+still\s+exploring\.\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*I'm\s+looking\s+for\s+(.+?)\.\s+.+?\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
)


def extract_category_phrase(message: str) -> str | None:
    """Extract a category from an evaluator opening message, if recognized."""
    if not isinstance(message, str):
        return None
    for pattern in OPENING_PATTERNS:
        match = pattern.fullmatch(message)
        if match:
            category = " ".join(match.group(1).strip().split())
            return category or None
    return None


class BucketRoute:
    def __init__(self, catalog: dict[str, dict]):
        buckets: defaultdict[str, list[str]] = defaultdict(list)
        for asin, product in catalog.items():
            buckets[normalize_phrase(coarse_category(product))].append(str(asin))
        self.buckets: dict[str, list[str]] = dict(buckets)

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        """Return deterministic, catalog-ordered exact bucket matches."""
        if not isinstance(limit, int) or limit <= 0:
            return []
        category = extract_category_phrase(text)
        if category is None:
            return []
        asins = self.buckets.get(normalize_phrase(category), ())
        return [(asin, 1.0) for asin in asins[:limit]]

    def bucket_for_message(self, text: str) -> tuple[str, ...]:
        """Return the full immutable bucket for a recognized opening message."""
        category = extract_category_phrase(text)
        if category is None:
            return ()
        return tuple(self.buckets.get(normalize_phrase(category), ()))


def bucket_band(size: int) -> str:
    if size <= 10:
        return "<=10"
    if size <= 50:
        return "11-50"
    if size <= 500:
        return "51-500"
    return ">500"


def analyze_catalog(route: BucketRoute) -> dict:
    sizes = [len(asins) for asins in route.buckets.values()]
    bands = dict.fromkeys(BUCKET_BANDS, 0)
    for size in sizes:
        bands[bucket_band(size)] += 1
    return {
        "bucket_count": len(sizes),
        "bands": bands,
        "minimum": min(sizes) if sizes else 0,
        "median": statistics.median(sizes) if sizes else 0,
        "mean": statistics.fmean(sizes) if sizes else 0,
        "maximum": max(sizes) if sizes else 0,
    }


def load_public_sessions(path: str = "data/public_set.jsonl") -> list[dict]:
    return _load_jsonl(path)


def _opening_message(sample: dict, product: dict) -> str:
    scenario = str(sample.get("scenario_type", ""))
    card = sample.get("intent_card") or intent_card(product)
    behavior = sample.get("behavior")
    if behavior is None:
        seed_source = f"{sample.get('sample_id', '')}\0{scenario}"
        behavior = behavior_for(scenario, card, random.Random(seed_source))
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    category = coarse_category(product)
    return initial_message(effective_sample, category, set())


def _empty_session_stats() -> dict:
    return {
        "session_count": 0,
        "target_in_bucket": 0,
        "coverage_at": {cutoff: 0 for cutoff in COVERAGE_CUTOFFS},
    }


def _coverage_percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def analyze_sessions(
    sessions: list[dict], catalog: dict[str, dict], route: BucketRoute
) -> dict:
    bands = dict.fromkeys(BUCKET_BANDS, 0)
    parse_failures = 0
    missing_products = 0
    missing_buckets = 0
    target_coverage = 0
    coverage_at = {cutoff: 0 for cutoff in COVERAGE_CUTOFFS}
    target_positions: list[int] = []
    scenario_stats: defaultdict[str, dict] = defaultdict(_empty_session_stats)
    first_message: str | None = None
    for sample in sessions:
        scenario = str(sample.get("scenario_type") or "unknown")
        scenario_stats[scenario]["session_count"] += 1
        target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
        product = catalog.get(target)
        if product is None:
            missing_products += 1
            continue
        message = _opening_message(sample, product)
        first_message = first_message or message
        category = extract_category_phrase(message)
        if category is None:
            parse_failures += 1
            continue
        bucket = route.buckets.get(normalize_phrase(category))
        if bucket is None:
            missing_buckets += 1
            continue
        bands[bucket_band(len(bucket))] += 1
        if target in bucket:
            target_coverage += 1
            scenario_stats[scenario]["target_in_bucket"] += 1
            position = bucket.index(target) + 1
            target_positions.append(position)
            for cutoff in COVERAGE_CUTOFFS:
                if position <= cutoff:
                    coverage_at[cutoff] += 1
                    scenario_stats[scenario]["coverage_at"][cutoff] += 1
    return {
        "bands": bands,
        "parse_failures": parse_failures,
        "missing_products": missing_products,
        "missing_buckets": missing_buckets,
        "target_coverage": target_coverage,
        "coverage_at": coverage_at,
        "target_position_median": statistics.median(target_positions) if target_positions else None,
        "target_position_mean": statistics.fmean(target_positions) if target_positions else None,
        "scenario_stats": dict(scenario_stats),
        "first_message": first_message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze deterministic coarse-category buckets")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", default="data/public_set.jsonl")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    route = BucketRoute(catalog)
    sessions = load_public_sessions(args.sessions)
    catalog_stats = analyze_catalog(route)
    session_stats = analyze_sessions(sessions, catalog, route)

    print("=== BucketRoute Analysis ===\n")
    print(f"Catalog products: {len(catalog):,}")
    print(f"Buckets: {catalog_stats['bucket_count']:,}\n")
    print("Bucket size distribution")
    for band in BUCKET_BANDS:
        print(f"{band}: {catalog_stats['bands'][band]:>6,} buckets")
    print("\nBucket sizes")
    print(f"min: {catalog_stats['minimum']:,}")
    print(f"median: {catalog_stats['median']:,.1f}")
    print(f"mean: {catalog_stats['mean']:,.2f}")
    print(f"max: {catalog_stats['maximum']:,}\n")

    total = len(sessions)
    print(f"Public sessions: {total:,}\n")
    print("Session bucket distribution")
    for band in BUCKET_BANDS:
        count = session_stats["bands"][band]
        percentage = 100.0 * count / total if total else 0.0
        print(f"{band}: {count:>6,} / {total:,} ({percentage:.2f}%)")
    print(f"\nCategory parse failures: {session_stats['parse_failures']:,}")
    print(f"Missing target products: {session_stats['missing_products']:,}")
    print(f"Missing category buckets: {session_stats['missing_buckets']:,}")
    coverage = session_stats["target_coverage"]
    percentage = _coverage_percent(coverage, total)
    print(f"Target-in-bucket coverage: {coverage:,} / {total:,} ({percentage:.2f}%)")
    print("Catalog-order retrieval coverage")
    for cutoff in COVERAGE_CUTOFFS:
        count = session_stats["coverage_at"][cutoff]
        print(f"@{cutoff}: {count:,} / {total:,} ({_coverage_percent(count, total):.2f}%)")
    median = session_stats["target_position_median"]
    mean = session_stats["target_position_mean"]
    if median is not None and mean is not None:
        print(f"Target bucket position median: {median:,.1f}")
        print(f"Target bucket position mean: {mean:,.2f}")

    print("\nCoverage@200 by scenario")
    for scenario, stats in sorted(session_stats["scenario_stats"].items()):
        count = stats["coverage_at"][200]
        scenario_total = stats["session_count"]
        print(
            f"{scenario}: {count:,} / {scenario_total:,} "
            f"({_coverage_percent(count, scenario_total):.2f}%)"
        )


if __name__ == "__main__":
    main()

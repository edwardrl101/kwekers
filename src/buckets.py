"""Coarse-category precision filter and offline diagnostics.

Day 2 role: preserve BM25 ordering while removing candidates outside the
coarse category in an evaluator-style opening message. Uncertain or empty
filtering results fall back to the original pool.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


BUCKET_BANDS = ("<=10", "11-50", "51-500", ">500")
BM25_POOL_LIMIT = 500 #receives up to 500 BM25 candidates
_SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store") #searchable metadata
_MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


def _load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL and include the source line in validation errors."""
    source = Path(path)
    records: list[dict] = []
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


def load_catalog(path: str | Path = "data/catalog.jsonl") -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for product in _load_jsonl(path):
        parent_asin = str(product.get("parent_asin", "")).strip()
        if not parent_asin:
            raise ValueError("Every catalog product must have a non-empty parent_asin")
        if parent_asin in catalog:
            raise ValueError(f"Duplicate parent_asin in catalog: {parent_asin}")
        catalog[parent_asin] = product
    return catalog


def load_public_sessions(path: str | Path = "data/public_set.jsonl") -> list[dict]:
    """Loads the 200 public sessions using the same safe loader."""
    return _load_jsonl(path)


def normalize_phrase(text: str) -> str:
    return " ".join(text.lower().strip().split())


def coarse_category(product: dict) -> str:
    """Local copy aligned with evaluator.coarse_category."""
    values = [str(value) for value in product.get("categories") or []]
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


# Buying: I'm looking for {category}. A key requirement is: {constraint}.
# Browsing/Boundary: I'm looking for {category}, but I'm still exploring.
# Override: I'm looking for {category}. {old preference}
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
        self.bucket_sets: dict[str, set[str]] = {
            category: set(asins) for category, asins in self.buckets.items()
        }

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        if not isinstance(limit, int) or limit <= 0:
            return []
        category = extract_category_phrase(text)
        if category is None:
            return []
        asins = self.buckets.get(normalize_phrase(category), ())
        return [(asin, 1.0) for asin in asins[:limit]]

    def bucket_for_message(self, text: str) -> tuple[str, ...]:
        category = extract_category_phrase(text)
        if category is None:
            return ()
        return tuple(self.buckets.get(normalize_phrase(category), ()))

    def filter_by_category(self, pool: list[str], message: str) -> list[str]:
        """Filter without reordering and never turn a usable pool into empty."""
        if not pool:
            return pool
        category = extract_category_phrase(message)
        if category is None:
            return pool
        bucket = self.bucket_sets.get(normalize_phrase(category))
        if not bucket:
            return pool
        filtered = [asin for asin in pool if asin in bucket]
        return filtered if filtered else pool


def bucket_band(size: int) -> str:
    if size <= 10:
        return "<=10"
    if size <= 50:
        return "11-50"
    if size <= 500:
        return "51-500"
    return ">500"


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in _SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _intent_card(product: dict, limit: int = 180) -> dict:
    """Local evaluator-aligned intent-card reconstruction for diagnostics."""
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = _MATERIAL_RE.search(corpus)
    color = _COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(
            _clean_constraint(item, limit)
            for item in candidates
            if _clean_constraint(item, limit)
        )
    )
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def _opening_message(sample: dict, product: dict) -> str:
    scenario = str(sample.get("scenario_type", ""))
    card = sample.get("intent_card") or _intent_card(product)
    category = coarse_category(product)
    if scenario == "buying" and card.get("hard_constraints"):
        constraint = str(card["hard_constraints"][0])
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        behavior = sample.get("behavior")
        if behavior is not None:
            old_value = behavior["override"]["old_value"]
        else:
            soft = card["soft_preferences"]
            old_value = soft[-1] if soft else "I prefer a different style."
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


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


def analyze_sessions(sessions: list[dict], catalog: dict[str, dict], route: BucketRoute) -> dict:
    bands = dict.fromkeys(BUCKET_BANDS, 0)
    browsing_bands = dict.fromkeys(BUCKET_BANDS, 0)
    parse_successes = parse_failures = missing_products = missing_buckets = 0
    small_bucket_sessions = small_bucket_targets = 0
    for sample in sessions:
        target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
        product = catalog.get(target)
        if product is None:
            missing_products += 1
            continue
        category = extract_category_phrase(_opening_message(sample, product))
        if category is None:
            parse_failures += 1
            continue
        parse_successes += 1
        bucket = route.buckets.get(normalize_phrase(category))
        if not bucket:
            missing_buckets += 1
            continue
        band = bucket_band(len(bucket))
        bands[band] += 1
        if sample.get("scenario_type") == "browsing":
            browsing_bands[band] += 1
        if len(bucket) <= 10:
            small_bucket_sessions += 1
            small_bucket_targets += int(target in bucket)
    return {
        "bands": bands,
        "browsing_bands": browsing_bands,
        "parse_successes": parse_successes,
        "parse_failures": parse_failures,
        "missing_products": missing_products,
        "missing_buckets": missing_buckets,
        "small_bucket_sessions": small_bucket_sessions,
        "small_bucket_targets": small_bucket_targets,
    }


def evaluate_browsing_filter(
    sessions: list[dict],
    catalog: dict[str, dict],
    route: BucketRoute,
    bm25_route: object,
    pool_limit: int = BM25_POOL_LIMIT,
) -> dict:
    """Measure browsing R@10 and filter safety over the existing BM25 route."""
    stats = {
        "sessions": 0, "parse_successes": 0, "parse_failures": 0,
        "matching_buckets": 0, "filter_reduced": 0, "fallbacks": 0,
        "before_hits_at_10": 0, "after_hits_at_10": 0,
        "targets_in_pool": 0, "targets_preserved": 0, "targets_removed": 0,
    }
    before_sizes: list[int] = []
    after_sizes: list[int] = []
    for sample in sessions:
        if sample.get("scenario_type") != "browsing":
            continue
        target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
        product = catalog.get(target)
        if product is None:
            continue
        stats["sessions"] += 1
        message = _opening_message(sample, product)
        pool = [asin for asin, _score in bm25_route.query(message, limit=pool_limit)]
        filtered = route.filter_by_category(pool, message)
        before_sizes.append(len(pool))
        after_sizes.append(len(filtered))
        category = extract_category_phrase(message)
        bucket = route.bucket_sets.get(normalize_phrase(category)) if category else None
        if category is None:
            stats["parse_failures"] += 1
        else:
            stats["parse_successes"] += 1
        if bucket:
            stats["matching_buckets"] += 1
        if category is None or not bucket or not any(asin in bucket for asin in pool):
            stats["fallbacks"] += 1
        elif len(filtered) < len(pool):
            stats["filter_reduced"] += 1
        stats["before_hits_at_10"] += int(target in pool[:10])
        stats["after_hits_at_10"] += int(target in filtered[:10])
        if target in pool:
            stats["targets_in_pool"] += 1
            if target in filtered:
                stats["targets_preserved"] += 1
            else:
                stats["targets_removed"] += 1
    total = stats["sessions"]
    before = stats["before_hits_at_10"] / total if total else 0.0
    after = stats["after_hits_at_10"] / total if total else 0.0
    stats.update({
        "recall_at_10_before": before,
        "recall_at_10_after": after,
        "absolute_change": after - before,
        "relative_improvement": ((after - before) / before if before else None),
        "average_pool_before": statistics.fmean(before_sizes) if before_sizes else 0.0,
        "average_pool_after": statistics.fmean(after_sizes) if after_sizes else 0.0,
        "median_pool_after": statistics.median(after_sizes) if after_sizes else 0.0,
    })
    return stats


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Day 2 category filter")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", default="data/public_set.jsonl")
    parser.add_argument("--pool-limit", type=int, default=BM25_POOL_LIMIT)
    args = parser.parse_args()
    try:
        from src.retrieval import BM25Route
    except ModuleNotFoundError:  # direct: python3 src/buckets.py
        from retrieval import BM25Route

    catalog = load_catalog(args.catalog)
    sessions = load_public_sessions(args.sessions)
    route = BucketRoute(catalog)
    catalog_stats = analyze_catalog(route)
    session_stats = analyze_sessions(sessions, catalog, route)
    filter_stats = evaluate_browsing_filter(
        sessions, catalog, route, BM25Route(catalog), args.pool_limit
    )

    print("=== Category Bucket Day 2 ===\n")
    print(f"Catalog products: {len(catalog):,}")
    print(f"Total buckets: {catalog_stats['bucket_count']:,}\n")
    print("Catalog bucket histogram")
    for band in BUCKET_BANDS:
        print(f"{band}: {catalog_stats['bands'][band]:>6,} buckets")
    total = len(sessions)
    print("\nAll-session bucket distribution")
    for band in BUCKET_BANDS:
        count = session_stats["bands"][band]
        print(f"{band}: {count:>6,} / {total:,} ({_percent(count, total):.2f}%)")
    browsing_total = filter_stats["sessions"]
    print("\nBrowsing-only bucket distribution")
    for band in BUCKET_BANDS:
        count = session_stats["browsing_bands"][band]
        print(f"{band}: {count:>6,} / {browsing_total:,} ({_percent(count, browsing_total):.2f}%)")
    print("\nCategory parsing")
    print(f"Success: {session_stats['parse_successes']:,} / {total:,}")
    print(f"Failed:  {session_stats['parse_failures']:,} / {total:,}")
    print(f"Missing target products: {session_stats['missing_products']:,}")
    print(f"Missing category buckets: {session_stats['missing_buckets']:,}")
    print(
        f"Small-bucket target containment: {session_stats['small_bucket_targets']:,} / "
        f"{session_stats['small_bucket_sessions']:,}"
    )
    print("\nBrowsing filter evaluation")
    print(f"Sessions:                 {browsing_total:,}")
    print(f"BM25 R@10 before:         {filter_stats['recall_at_10_before']:.3f}")
    print(f"Filtered R@10 after:      {filter_stats['recall_at_10_after']:.3f}")
    print(f"Absolute change:          {filter_stats['absolute_change']:+.3f}")
    relative = filter_stats["relative_improvement"]
    print(f"Relative improvement:     {'n/a' if relative is None else f'{100.0 * relative:+.2f}%'}")
    print(f"Parsing succeeded:        {filter_stats['parse_successes']:,}")
    print(f"Parsing failed:           {filter_stats['parse_failures']:,}")
    print(f"Matching bucket exists:   {filter_stats['matching_buckets']:,}")
    print(f"Pools reduced:            {filter_stats['filter_reduced']:,}")
    print(f"Safety fallbacks:         {filter_stats['fallbacks']:,}")
    print(f"Targets in BM25@{args.pool_limit}:      {filter_stats['targets_in_pool']:,}")
    print(f"Targets preserved:        {filter_stats['targets_preserved']:,}")
    print(f"Targets removed:          {filter_stats['targets_removed']:,}")
    print(f"Average pool before:      {filter_stats['average_pool_before']:.2f}")
    print(f"Average pool after:       {filter_stats['average_pool_after']:.2f}")
    print(f"Median pool after:        {filter_stats['median_pool_after']:.2f}")
    print("\nTune/holdout: no committed 140/60 split found; not reported.")


if __name__ == "__main__":
    main()

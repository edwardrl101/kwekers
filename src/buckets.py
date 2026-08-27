from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load_catalog(
    path: str = "data/catalog.jsonl"
) -> dict[str, dict]:
    return {
        str(p["parent_asin"]): p
        for p in (
            json.loads(line)
            for line in open(path, encoding="utf-8")
        )
    }


def normalize_phrase(text: str) -> str:
    """Return the case- and whitespace-insensitive bucket lookup key."""
    return " ".join(text.lower().strip().split())


def coarse_category(product: dict) -> str:
    """Match the evaluator category built solely from ``product['categories']``."""
    values = [str(value) for value in product.get("categories") or []]
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


# The evaluator generates exactly these three opening shapes:
#   I'm looking for {category}. A key requirement is: {constraint}.
#   I'm looking for {category}. {old preference}
#   I'm looking for {category}, but I'm still exploring.
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
    bands = {"<=10": 0, "11-50": 0, "51-500": 0, ">500": 0}
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
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# The public records omit their generated opening text. These small helpers mirror
# the relevant evaluator path so standalone analysis reads the same opening that
# each public session receives. They do not affect BucketRoute retrieval.
_SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)


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


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _intent_card(product: dict, limit: int = 180) -> dict:
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
    card = sample.get("intent_card") or _intent_card(product)
    scenario = sample.get("scenario_type")
    category = coarse_category(product)
    if scenario == "buying" and card.get("hard_constraints"):
        constraint = str(card["hard_constraints"][0])
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        behavior = sample.get("behavior")
        if behavior is None:
            seed_source = f"{sample.get('sample_id', '')}\0{scenario}"
            rng = random.Random(seed_source)
            soft = card["soft_preferences"]
            hard = card["hard_constraints"]
            old_value = soft[-1] if soft else "I prefer a different style."
            # Consume the same random choice made while constructing evaluator behavior.
            rng.choice([3, 4])
        else:
            old_value = behavior["override"]["old_value"]
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def analyze_sessions(
    sessions: list[dict], catalog: dict[str, dict], route: BucketRoute
) -> dict:
    bands = {"<=10": 0, "11-50": 0, "51-500": 0, ">500": 0}
    parse_failures = 0
    missing_buckets = 0
    target_coverage = 0
    first_message: str | None = None
    for sample in sessions:
        target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
        product = catalog.get(target)
        if product is None:
            missing_buckets += 1
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
    return {
        "bands": bands,
        "parse_failures": parse_failures,
        "missing_buckets": missing_buckets,
        "target_coverage": target_coverage,
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

    assert route.query("", 10) == []
    assert route.query("nonsense", 10) == []
    if session_stats["first_message"] is not None:
        assert extract_category_phrase(session_stats["first_message"]) is not None
        assert route.query(session_stats["first_message"], 0) == []

    print("=== BucketRoute Analysis ===\n")
    print(f"Catalog products: {len(catalog):,}")
    print(f"Buckets: {catalog_stats['bucket_count']:,}\n")
    print("Bucket size distribution")
    for band in ("<=10", "11-50", "51-500", ">500"):
        print(f"{band}: {catalog_stats['bands'][band]:>6,} buckets")
    print("\nBucket sizes")
    print(f"min: {catalog_stats['minimum']:,}")
    print(f"median: {catalog_stats['median']:,.1f}")
    print(f"mean: {catalog_stats['mean']:,.2f}")
    print(f"max: {catalog_stats['maximum']:,}\n")

    total = len(sessions)
    print(f"Public sessions: {total:,}\n")
    print("Session bucket distribution")
    for band in ("<=10", "11-50", "51-500", ">500"):
        count = session_stats["bands"][band]
        percentage = 100.0 * count / total if total else 0.0
        print(f"{band}: {count:>6,} / {total:,} ({percentage:.2f}%)")
    print(f"\nCategory parse failures: {session_stats['parse_failures']:,}")
    print(f"Missing buckets: {session_stats['missing_buckets']:,}")
    coverage = session_stats["target_coverage"]
    percentage = 100.0 * coverage / total if total else 0.0
    print(f"Target-in-bucket coverage: {coverage:,} / {total:,} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()

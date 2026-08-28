"""Build and validate the local embedding cache used by DenseRoute."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import DenseRoute, load_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 50k dense retrieval cache")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/dense_cache.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"Loading catalog: {args.catalog}", flush=True)
    catalog = load_catalog(args.catalog)
    if not catalog:
        raise SystemExit("Catalog is empty; cannot build dense cache")
    print(f"Products: {len(catalog):,}", flush=True)
    print(f"Cache: {args.cache}", flush=True)
    print(f"Device: {args.device}", flush=True)

    route = DenseRoute(catalog, cache=args.cache, device=args.device)
    if len(route.asins) != len(catalog):
        raise RuntimeError(
            f"Dense cache contains {len(route.asins):,} IDs for a {len(catalog):,}-item catalog"
        )
    if route.embeddings.shape[0] != len(catalog):
        raise RuntimeError("Dense embedding row count does not match the catalog")

    first_product = next(iter(catalog.values()))
    smoke_query = str(first_product.get("title") or "clothing item")
    smoke_results = route.query(smoke_query, limit=3)
    elapsed = time.perf_counter() - started

    print(f"Embedding shape: {route.embeddings.shape}")
    print(f"Cache size: {args.cache.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"Smoke query: {smoke_query[:100]}")
    print(f"Smoke results: {smoke_results}")
    print(f"Completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()

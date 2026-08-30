from __future__ import annotations

from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_baseline() -> None:
    catalog = ROOT / "data/catalog.jsonl"
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog)
    result = evaluate(Agent(catalog), samples, catalog_ids, categories, products)

    assert result["sample_count"] == 200
    assert result["recommended_technical_score"] == 0.877011
    assert result["reported_token_usage"]["total_tokens"] == 0

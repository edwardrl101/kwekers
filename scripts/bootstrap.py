"""Day 4: paired bootstrap on the three contested Day 3 decisions.

CLAUDE.md's own statistical-discipline note: TechnicalScore's unpaired
bootstrap SE at n=200 is ~0.0096 (95% CI width ~0.019), and configs run on
the SAME 200 sessions are correlated, so an unpaired comparison is too
conservative. This resamples per-session outcomes with a SHARED resample
index across the two configs in each pair (paired bootstrap), which cancels
the shared session-to-session variance and answers "would config A beat
config B on a resample of this population" directly, rather than "do their
independent score distributions overlap."

Five configs are evaluated once each over all 200 public sessions (full
evaluate() runs, not a recall proxy - CLAUDE.md rule 7), then resampled
B=10,000 times from cached per-session outcomes (hit / reciprocal_rank /
first_hit_turn), which is pure array arithmetic and takes well under a
second per pair.

Pairs, matching Day_4_Plan.txt's wording plus one added clean single-axis
framing for the bucket question (see PAIRS below for why both are reported):

    no-dense vs full-freshness   (dense axis only, clean)
    full-freshness vs no-bucket  (bucket axis only, clean - the actual
                                   comparison CLAUDE.md's "+0.007" already
                                   cites, both configs carry dense=0.20)
    no-dense vs no-bucket(dense=0) (bucket axis only, shipped as the
                                   baseline - the literal Day-plan wording)
    no-dense vs bm25-only        (whole fusion stack vs the null model -
                                   three axes at once, deliberately holistic)

Usage:
    python3 scripts/day4_bootstrap.py [--iterations 10000] [--seed 20260831]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.eval import ABLATION_CONFIGS  # noqa: E402
from starter.agent import Agent  # noqa: E402

MAX_TURNS = 10
MISS_TURN = MAX_TURNS + 1

CONFIGS: dict[str, dict[str, bool | float]] = {
    "no-dense (shipped)": ABLATION_CONFIGS["no-dense"],
    "full-freshness": ABLATION_CONFIGS["freshness"],
    "no-bucket (dense on)": ABLATION_CONFIGS["no-bucket"],
    "no-bucket (dense off)": {
        "enable_freshness": True,
        "enable_dense": False,
        "bucket_match_boost": 0.0,
        "dense_similarity_weight": 0.0,
    },
    "bm25-only": ABLATION_CONFIGS["bm25-only"],
}

# (label, config_a, config_b, axis being isolated)
PAIRS = [
    ("Disable dense", "no-dense (shipped)", "full-freshness", "dense only"),
    (
        "Ship bucket (clean, dense=0.20 both sides)",
        "full-freshness",
        "no-bucket (dense on)",
        "bucket only",
    ),
    (
        "Ship bucket (literal Day-plan wording: vs shipped)",
        "no-dense (shipped)",
        "no-bucket (dense off)",
        "bucket only, shipped baseline",
    ),
    (
        "Full fusion stack beats BM25-only",
        "no-dense (shipped)",
        "bm25-only",
        "exact + bucket + dense together (holistic, not single-axis)",
    ),
]


def technical_score(
    hit: np.ndarray, reciprocal_rank: np.ndarray, first_hit_turn: np.ndarray
) -> np.ndarray:
    """Vectorized TechnicalScore over the last axis. Matches
    evaluator.local_evaluator.evaluate()'s formula exactly."""
    hit_rate = hit.mean(axis=-1)
    mrr = reciprocal_rank.mean(axis=-1)
    mttc = first_hit_turn.mean(axis=-1)
    efficiency = np.clip((11.0 - mttc) / 10.0, 0.0, 1.0)
    return 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency


def run_config(catalog_path: Path, samples: list[dict], catalog_ids, categories, products,
                options: dict) -> dict:
    agent = Agent(catalog_path, **options)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return {session["sample_id"]: session for session in result["sessions"]}, result


def as_arrays(sessions_by_id: dict[str, dict], sample_ids: list[str]) -> dict[str, np.ndarray]:
    hit = np.array([1.0 if sessions_by_id[sid]["hit"] else 0.0 for sid in sample_ids])
    reciprocal_rank = np.array([sessions_by_id[sid]["reciprocal_rank"] for sid in sample_ids])
    first_hit_turn = np.array(
        [
            sessions_by_id[sid]["first_hit_turn"]
            if sessions_by_id[sid]["first_hit_turn"] is not None
            else MISS_TURN
            for sid in sample_ids
        ],
        dtype=float,
    )
    return {"hit": hit, "reciprocal_rank": reciprocal_rank, "first_hit_turn": first_hit_turn}


def paired_bootstrap(
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    idx: np.ndarray,
) -> dict:
    n = arrays_a["hit"].shape[0]
    scores_a = technical_score(
        arrays_a["hit"][idx], arrays_a["reciprocal_rank"][idx], arrays_a["first_hit_turn"][idx]
    )
    scores_b = technical_score(
        arrays_b["hit"][idx], arrays_b["reciprocal_rank"][idx], arrays_b["first_hit_turn"][idx]
    )
    deltas = scores_a - scores_b
    point_a = technical_score(arrays_a["hit"], arrays_a["reciprocal_rank"], arrays_a["first_hit_turn"])
    point_b = technical_score(arrays_b["hit"], arrays_b["reciprocal_rank"], arrays_b["first_hit_turn"])
    return {
        "n_sessions": n,
        "point_estimate_a": round(float(point_a), 6),
        "point_estimate_b": round(float(point_b), 6),
        "point_delta": round(float(point_a - point_b), 6),
        "paired_mean_delta": round(float(deltas.mean()), 6),
        "paired_se_delta": round(float(deltas.std(ddof=1)), 6),
        "paired_ci95": [round(float(np.percentile(deltas, 2.5)), 6),
                         round(float(np.percentile(deltas, 97.5)), 6)],
        "excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
        "unpaired_se_a": round(float(scores_a.std(ddof=1)), 6),
        "unpaired_se_b": round(float(scores_b.std(ddof=1)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "day4_bootstrap.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    catalog_ids, categories, products = catalog_index(args.catalog)

    print(f"Running {len(CONFIGS)} configs over {len(samples)} public sessions "
          f"(full evaluate(), not a recall proxy)...\n")
    sessions_by_config: dict[str, dict[str, np.ndarray]] = {}
    point_results: dict[str, dict] = {}
    for name, options in CONFIGS.items():
        sessions_by_id, result = run_config(
            args.catalog, samples, catalog_ids, categories, products, options
        )
        sessions_by_config[name] = as_arrays(sessions_by_id, sample_ids)
        point_results[name] = {
            "options": options,
            "technical_score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
        }
        print(f"  {name:32s} score={result['recommended_technical_score']:.6f}  "
              f"hit={result['hit_rate_at_10']:.6f}  mrr={result['mrr']:.6f}  "
              f"mttc={result['mttc']:.4f}")

    rng = np.random.default_rng(args.seed)
    n = len(sample_ids)
    idx = rng.integers(0, n, size=(args.iterations, n))

    print(f"\nPaired bootstrap, {args.iterations:,} resamples, seed={args.seed}\n")
    pair_results = []
    for label, key_a, key_b, axis in PAIRS:
        stats = paired_bootstrap(sessions_by_config[key_a], sessions_by_config[key_b], idx)
        pair_results.append({"label": label, "a": key_a, "b": key_b, "axis": axis, **stats})
        verdict = "SIGNIFICANT" if stats["excludes_zero"] else "NOT SIGNIFICANT (CI straddles zero)"
        print(f"{label}")
        print(f"  axis isolated: {axis}")
        print(f"  {key_a} ({stats['point_estimate_a']:.6f}) vs {key_b} ({stats['point_estimate_b']:.6f})")
        print(f"  point delta:        {stats['point_delta']:+.6f}")
        print(f"  paired mean delta:  {stats['paired_mean_delta']:+.6f}  (SE {stats['paired_se_delta']:.6f})")
        print(f"  paired 95% CI:      [{stats['paired_ci95'][0]:+.6f}, {stats['paired_ci95'][1]:+.6f}]")
        print(f"  unpaired SE (a,b):  ({stats['unpaired_se_a']:.6f}, {stats['unpaired_se_b']:.6f})")
        print(f"  verdict:            {verdict}\n")

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iterations": args.iterations,
        "seed": args.seed,
        "sample_count": n,
        "configs": point_results,
        "pairs": pair_results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

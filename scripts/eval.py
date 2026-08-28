from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def _score_fields(metrics: dict) -> dict[str, float | None]:
    mttc = metrics.get("mttc")
    if mttc is None:
        return {"efficiency": None, "technical_score": None}
    efficiency = max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    score = (
        0.50 * float(metrics["hit_rate_at_10"])
        + 0.30 * float(metrics["mrr"])
        + 0.20 * efficiency
    )
    return {"efficiency": round(efficiency, 6), "technical_score": round(score, 6)}


def _select_samples(samples: list[dict], split_name: str, split_path: Path) -> list[dict]:
    if split_name == "all":
        return samples
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    key = f"{split_name}_sample_ids"
    selected_ids = manifest.get(key)
    if not isinstance(selected_ids, list):
        raise ValueError(f"Split manifest is missing a list named {key!r}")
    selected_set = {str(value) for value in selected_ids}
    selected = [sample for sample in samples if str(sample.get("sample_id")) in selected_set]
    found_ids = {str(sample.get("sample_id")) for sample in selected}
    missing = selected_set - found_ids
    if missing:
        raise ValueError(f"Split contains {len(missing)} IDs absent from the dataset")
    if len(selected) != len(selected_ids):
        raise ValueError("Split or dataset contains duplicate sample IDs")
    return selected


def _print_metrics(result: dict, split_name: str) -> None:
    print(f"\nEvaluation split: {split_name}")
    print(
        f"Overall: n={result['sample_count']}  "
        f"score={result['recommended_technical_score']:.6f}  "
        f"hit@10={result['hit_rate_at_10']:.6f}  "
        f"mrr={result['mrr']:.6f}  mttc={result['mttc']:.6f}  "
        f"efficiency={result['efficiency']:.6f}"
    )
    print("\nPer-scenario:")
    print("scenario          n   score     hit@10    mrr       mttc      efficiency")
    for scenario in SCENARIOS:
        metrics = result["scenario_metrics"].get(scenario)
        if not metrics:
            continue
        derived = _score_fields(metrics)
        print(
            f"{scenario:<16} {metrics['sample_count']:>3} "
            f"{derived['technical_score']:>9.6f} "
            f"{metrics['hit_rate_at_10']:>9.6f} "
            f"{metrics['mrr']:>9.6f} "
            f"{metrics['mttc']:>9.6f} "
            f"{derived['efficiency']:>11.6f}"
        )


def _csv_row(result: dict, split_name: str, label: str) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "split": split_name,
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "prompt_tokens": result["reported_token_usage"]["prompt_tokens"],
        "completion_tokens": result["reported_token_usage"]["completion_tokens"],
    }
    for scenario in SCENARIOS:
        metrics = result["scenario_metrics"].get(scenario, {})
        derived = _score_fields(metrics) if metrics else {}
        prefix = f"{scenario}_"
        row[prefix + "sample_count"] = metrics.get("sample_count", 0)
        row[prefix + "hit_rate_at_10"] = metrics.get("hit_rate_at_10", "")
        row[prefix + "mrr"] = metrics.get("mrr", "")
        row[prefix + "mttc"] = metrics.get("mttc", "")
        row[prefix + "efficiency"] = derived.get("efficiency", "")
        row[prefix + "technical_score"] = derived.get("technical_score", "")
    return row


def _append_run(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and log the Kwekers local evaluation")
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "data/catalog.jsonl"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/public_set.jsonl"
    )
    parser.add_argument(
        "--split-file", type=Path, default=ROOT / "data/eval_split.json"
    )
    parser.add_argument("--split", choices=("tune", "holdout", "all"), default="tune")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs" / "runs.csv")
    parser.add_argument("--label", default="day1-random-fill-v1")
    args = parser.parse_args()

    samples = _select_samples(load_jsonl(args.dataset), args.split, args.split_file)
    if not samples:
        parser.error(f"The {args.split!r} split contains no samples")
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)

    output = args.output or ROOT / f"results_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_metrics(result, args.split)
    _append_run(args.runs, _csv_row(result, args.split, args.label))
    print(f"\nWrote results: {output}")
    print(f"Appended run:  {args.runs}")


if __name__ == "__main__":
    main()

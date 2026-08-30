"""Evaluate fixed Agent configurations over grouped cross-validation folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary  # noqa: E402
from scripts.build_cv_folds import SCENARIOS, validate_manifest  # noqa: E402
from starter.agent import Agent  # noqa: E402


CONFIGS: dict[str, dict[str, bool | float]] = {
    "no-dense": {
        "enable_freshness": True,
        "enable_dense": False,
        "exact_match_boost": 0.35,
        "bucket_match_boost": 0.10,
        "dense_similarity_weight": 0.0,
    },
    "bm25-only": {
        "enable_freshness": True,
        "enable_dense": False,
        "exact_match_boost": 0.0,
        "bucket_match_boost": 0.0,
        "dense_similarity_weight": 0.0,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, str | None]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current")}


def summarize_sessions(sessions: list[dict]) -> dict:
    overall = metric_summary(sessions)
    if overall["mttc"] is None:
        efficiency = 0.0
    else:
        efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
        "scenario_metrics": {
            scenario: metric_summary(grouped[scenario]) for scenario in SCENARIOS if grouped[scenario]
        },
    }


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    if not fold_results:
        raise ValueError("at least one fold result is required")
    scores = [float(item["metrics"]["recommended_technical_score"]) for item in fold_results]
    sessions = [session for item in fold_results for session in item["sessions"]]
    aggregate = summarize_sessions(sessions)
    return {
        "fold_count": len(fold_results),
        "mean_technical_score": round(statistics.fmean(scores), 6),
        "technical_score_population_sd": round(statistics.pstdev(scores), 6),
        "worst_fold_technical_score": round(min(scores), 6),
        "best_fold_technical_score": round(max(scores), 6),
        "out_of_fold": aggregate,
    }


def _select_folds(manifest: dict, requested: str) -> list[dict]:
    folds = manifest["folds"]
    if requested == "all":
        return folds
    names = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [fold for fold in folds if fold["name"] in names]
    missing = names - {fold["name"] for fold in selected}
    if missing:
        raise ValueError(f"unknown folds: {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError("no folds selected")
    return selected


def _print_report(report: dict) -> None:
    print(f"Configuration: {report['configuration']['name']} {report['configuration']['options']}")
    for fold in report["folds"]:
        metrics = fold["metrics"]
        print(
            f"{fold['name']}: n={metrics['sample_count']} "
            f"score={metrics['recommended_technical_score']:.6f} "
            f"hit@10={metrics['hit_rate_at_10']:.6f} mrr={metrics['mrr']:.6f} "
            f"mttc={metrics['mttc']:.6f} runtime={fold['runtime_seconds']:.3f}s"
        )
    summary = report["summary"]
    oof = summary["out_of_fold"]
    print(
        f"CV mean={summary['mean_technical_score']:.6f} "
        f"sd={summary['technical_score_population_sd']:.6f} "
        f"worst={summary['worst_fold_technical_score']:.6f}"
    )
    print(
        f"OOF: n={oof['sample_count']} score={oof['recommended_technical_score']:.6f} "
        f"hit@10={oof['hit_rate_at_10']:.6f} mrr={oof['mrr']:.6f} mttc={oof['mttc']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/cv_folds.json")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="no-dense")
    parser.add_argument("--folds", default="all", help="all, one fold, or comma-separated folds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(samples, manifest)
    if args.validate_only:
        print(f"validated {args.manifest} ({len(samples)} samples, {len(manifest['folds'])} folds)")
        return

    selected_folds = _select_folds(manifest, args.folds)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    catalog_ids, categories, products = catalog_index(args.catalog)
    options = CONFIGS[args.config]
    started = time.perf_counter()
    agent = Agent(args.catalog, **options)
    fold_results: list[dict] = []
    for fold in selected_folds:
        fold_samples = [sample_by_id[sample_id] for sample_id in fold["validation_sample_ids"]]
        fold_started = time.perf_counter()
        result = evaluate(agent, fold_samples, catalog_ids, categories, products)
        runtime = time.perf_counter() - fold_started
        fold_results.append(
            {
                "name": fold["name"],
                "runtime_seconds": round(runtime, 6),
                "metrics": {key: value for key, value in result.items() if key != "sessions"},
                "sessions": result["sessions"],
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": {"name": args.config, "options": options},
        "metadata": {
            "git": _git_metadata(),
            "python": sys.version,
            "platform": platform.platform(),
            "dataset_sha256": _sha256(args.dataset),
            "catalog_sha256": _sha256(args.catalog),
            "manifest_sha256": manifest["manifest_sha256"],
            "runtime_seconds": round(time.perf_counter() - started, 6),
        },
        "folds": fold_results,
        "summary": aggregate_fold_results(fold_results),
    }
    output = args.output or ROOT / f"results_cv_{args.config}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

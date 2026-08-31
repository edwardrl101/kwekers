from __future__ import annotations

"""Sheng Yan Day-3 full-evaluator comparison for clarification policies.

Runs the *real* 200-session evaluator twice on the same current Agent pipeline:
1) always ask ``other``;
2) ask ``other`` twice, then rotate feature/material/color/style.

The official evaluator/data are read-only. Results are written under results/.
Tune/holdout numbers are derived from the same full run so we do not waste time
re-running identical sessions.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)
from src.dialog import (  # noqa: E402
    POLICY_ALWAYS_OTHER,
    POLICY_OTHER_TWICE_ROTATE,
)
from starter.agent import Agent  # noqa: E402

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
POLICIES = (POLICY_ALWAYS_OTHER, POLICY_OTHER_TWICE_ROTATE)


def _with_score(summary: dict) -> dict:
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )
    return {
        **summary,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _summarize_sessions(sessions: list[dict]) -> dict:
    overall = _with_score(metric_summary(sessions))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    return {
        **overall,
        "scenario_metrics": {
            name: _with_score(metric_summary(grouped[name]))
            for name in SCENARIOS
            if grouped.get(name)
        },
    }


def _split_ids(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "tune": {str(value) for value in payload["tune_sample_ids"]},
        "holdout": {str(value) for value in payload["holdout_sample_ids"]},
    }


def _print_summary(policy: str, split: str, result: dict) -> None:
    print(
        f"{policy:20s} {split:7s} "
        f"score={result['recommended_technical_score']:.6f} "
        f"hit={result['hit_rate_at_10']:.6f} "
        f"mrr={result['mrr']:.6f} mttc={result['mttc']:.6f}"
    )
    for scenario in SCENARIOS:
        row = result["scenario_metrics"].get(scenario)
        if row:
            print(
                f"  {scenario:16s} n={row['sample_count']:3d} "
                f"hit={row['hit_rate_at_10']:.6f} "
                f"mrr={row['mrr']:.6f} mttc={row['mttc']:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Sheng Yan's two question policies on the full evaluator"
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data/eval_split.json")
    parser.add_argument(
        "--policy",
        choices=("all", *POLICIES),
        default="all",
        help="run both policies or only one policy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "dialog_policy_comparison.json",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    split_ids = _split_ids(args.split_file)

    # Build the routes once, then change only the question-policy mode between
    # full runs. That makes the comparison controlled and avoids a second index build.
    agent = Agent(
        args.catalog,
        enable_dense=False,
        question_policy_mode=POLICY_ALWAYS_OTHER,
    )

    report: dict[str, dict] = {}
    policies = POLICIES if args.policy == "all" else (args.policy,)
    for policy in policies:
        print(f"\nRunning full evaluator: {policy}", flush=True)
        agent.question_policy_mode = policy
        full = evaluate(agent, samples, catalog_ids, categories, products)
        sessions = full["sessions"]

        policy_report: dict[str, dict] = {
            "all": {key: value for key, value in full.items() if key != "sessions"}
        }
        for split_name, ids in split_ids.items():
            subset = [
                row for row in sessions if str(row.get("sample_id")) in ids
            ]
            if not subset:
                raise ValueError(
                    f"Split '{split_name}' matched 0 sessions. "
                    "Check that --splits IDs match the dataset sample_ids."
                )
            policy_report[split_name] = _summarize_sessions(subset)
        report[policy] = policy_report

        for split_name in ("all", "tune", "holdout"):
            _print_summary(policy, split_name, policy_report[split_name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote comparison: {args.output}")


if __name__ == "__main__":
    main()

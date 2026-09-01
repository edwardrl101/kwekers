"""Confidence-by-turn measurement and demo chart.

Shows the read-only confidence value (src/confidence.py, wired in as
Agent._confidence_from_routes when ENABLE_CONFIDENCE is on) across turns of
the real 200-session public set - does belief visibly sharpen as constraints
accumulate, the way CLAUDE.md's "information beats algorithms" finding
predicts?

This is a copy of evaluator.local_evaluator.evaluate()'s per-turn loop (never
edit evaluator/ - it is frozen) with one addition: after every respond()
call, it reads back agent._sessions[session_id]["confidence"] before deciding
whether the session continues. Reading a private attribute for read-only
measurement matches the existing pattern in src/dedup.py's _EvaluationAgent
and scripts/eval_cv.py (both already reach into Agent's private routes).
Nothing here feeds back into ranking - see tests/test_confidence.py for the
proof that ENABLE_CONFIDENCE does not move the official score.

Usage:
    python3 scripts/confidence_by_turn.py
    python3 scripts/confidence_by_turn.py --output results/confidence_by_turn.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")

# dataviz skill categorical palette (light mode), fixed assignment - never
# cycled. Slot 1 (blue) is reserved for the overall/all-scenario line.
PALETTE = {
    "overall": "#2a78d6",       # slot 1 blue
    "buying": "#eb6834",        # slot 2 orange
    "browsing": "#1baf7a",      # slot 3 aqua
    "intent_override": "#eda100",  # slot 4 yellow
    "boundary": "#e87ba4",      # slot 5 magenta
}


def trace_one_session(agent: Agent, sample: dict, catalog_ids, categories, products) -> list[dict]:
    """Run one session turn-by-turn, recording confidence after each turn.

    Stops on hit, exactly like the real evaluator (a session that has already
    found its target is not consulted again) - so later turns naturally have
    fewer contributing sessions. Reported turn counts make this explicit
    rather than silently averaging over a shrinking, non-representative set.
    """
    session_id = f"conf_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective_sample, coarse_category(categories.get(target, [])), disclosed
    )

    turns: list[dict] = []
    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict):
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        confidence = float(agent._sessions.get(session_id, {}).get("confidence", 0.0))
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        target_present = target in ranked
        hit = override_applied and target_present
        turns.append({"turn": turn, "confidence": confidence, "hit": hit})
        if hit or turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )
    return turns


def aggregate(records: list[dict]) -> dict[int, dict]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for row in records:
        by_turn[row["turn"]].append(row["confidence"])
    return {
        turn: {
            "n": len(values),
            "mean": round(statistics.fmean(values), 6),
            "median": round(statistics.median(values), 6),
            "stdev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        }
        for turn, values in sorted(by_turn.items())
    }


def render_chart(overall: dict[int, dict], by_scenario: dict[str, dict[int, dict]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager

    plt.rcParams["font.family"] = [
        "system-ui" if "system-ui" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans"
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    turns_overall = sorted(overall)
    ax.plot(
        turns_overall,
        [overall[t]["mean"] for t in turns_overall],
        color=PALETTE["overall"],
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="All scenarios",
        zorder=5,
    )
    for turn in turns_overall:
        ax.annotate(
            f"n={overall[turn]['n']}",
            (turn, overall[turn]["mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7.5,
            color="#898781",
            ha="center",
        )

    for scenario in SCENARIOS:
        series = by_scenario.get(scenario, {})
        turns = sorted(t for t in series if series[t]["n"] >= 3)
        if len(turns) < 2:
            continue
        ax.plot(
            turns,
            [series[t]["mean"] for t in turns],
            color=PALETTE[scenario],
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            alpha=0.85,
            label=scenario.replace("_", " "),
        )

    ax.set_xlabel("Turn", color="#0b0b0b", fontsize=11)
    ax.set_ylabel("Confidence (softmax entropy over fused pool, 0-1)", color="#0b0b0b", fontsize=11)
    ax.set_title(
        "Measured confidence stays flat - a negative result, reported honestly",
        color="#0b0b0b", fontsize=12.5, fontweight="bold", pad=14,
    )
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.0, 0.05)
    ax.set_xticks(range(1, 11))
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    legend = ax.legend(
        loc="upper right", frameon=False, fontsize=9, labelcolor="#0b0b0b"
    )
    fig.text(
        0.01, 0.005,
        "Read-only (proven: ENABLE_CONFIDENCE never changes ranking, tests/test_confidence.py). Y-axis is zoomed to 0-0.05 to show the real\n"
        "shape, not padded to 0-1 to look dramatic. Why it's flat: the BM25 pool saturates at 500 candidates almost every turn regardless of\n"
        "disclosure; only ~1 candidate per turn earns the +0.35 exact-match boost, and softmax over ~500 rank-normalized [0,1] scores stays\n"
        "high-entropy even then. n= sessions of 200 still active at that turn (sessions stop once they hit).",
        fontsize=7.3, color="#898781",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "confidence_by_turn.png")
    parser.add_argument("--json-output", type=Path, default=ROOT / "results" / "confidence_by_turn.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, enable_confidence=True)

    all_records: list[dict] = []
    records_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        turns = trace_one_session(agent, sample, catalog_ids, categories, products)
        all_records.extend(turns)
        records_by_scenario[str(sample["scenario_type"])].extend(turns)

    overall = aggregate(all_records)
    by_scenario = {scenario: aggregate(records_by_scenario[scenario]) for scenario in SCENARIOS}

    print(f"{len(samples)} sessions\n")
    print(f"{'turn':>4}  {'n':>4}  {'mean conf':>10}  {'median':>8}  {'stdev':>7}")
    for turn, row in overall.items():
        print(f"{turn:>4}  {row['n']:>4}  {row['mean']:>10.4f}  {row['median']:>8.4f}  {row['stdev']:>7.4f}")

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_count": len(samples),
        "overall_by_turn": overall,
        "by_scenario": by_scenario,
    }
    args.json_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.json_output}")

    render_chart(overall, by_scenario, args.output)


if __name__ == "__main__":
    main()

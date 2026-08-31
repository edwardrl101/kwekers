"""Member 2 robustness, latency, and offline-cost evaluation harness.

This module intentionally wraps the shipped Agent without changing production
ranking or retrieval code. Level 0 is an identity control and must reproduce
the current main-branch offline TechnicalScore of 0.891111 before other levels
run. The older 0.877011 score is retained in the Day 3 historical report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from src import llm  # noqa: E402
from starter.agent import Agent  # noqa: E402


FROZEN_SCORE = 0.891111
LEVELS = (0, 1, 2, 3, 4)
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")

_MARKER_RE = re.compile(
    r"(?P<marker>a key requirement is:|for that,?\s*what matters is:|"
    r"what i need is:)\s*(?P<body>.+?)(?P<ending>[.!?]*)$",
    re.IGNORECASE | re.DOTALL,
)
_OPENING_RE = re.compile(r"^I'm looking for (?P<category>.+?)(?P<rest>[,.].*)$", re.DOTALL)
_PERCENT_PAIR_RE = re.compile(
    r"(?P<major>\d+(?:\.\d+)?)\s*%\s*(?P<major_name>[A-Za-z]+)\s*[,/&+ ]+\s*"
    r"(?P<minor>\d+(?:\.\d+)?)\s*%\s*(?P<minor_name>[A-Za-z]+)",
    re.IGNORECASE,
)
_BUDGET_RE = re.compile(r"\bbudget\s+around\s+\$(\d+(?:\.\d+)?)", re.IGNORECASE)

_SEMANTIC_REPLACEMENTS = (
    (re.compile(r"\bwater[ -]?resistant\b", re.I), "able to handle light rain"),
    (re.compile(r"\bwaterproof\b", re.I), "able to keep water out"),
    (re.compile(r"\bnavy blue\b", re.I), "dark blue"),
    (re.compile(r"\bcolor:\s*black\b", re.I), "in the darkest neutral shade"),
    (re.compile(r"\bcolor:\s*white\b", re.I), "in a bright neutral shade"),
    (re.compile(r"\bcolor:\s*blue\b", re.I), "in an azure shade"),
    (re.compile(r"\bcolor:\s*red\b", re.I), "in a crimson shade"),
    (re.compile(r"\bcolor:\s*pink\b", re.I), "in a rose shade"),
    (re.compile(r"\bcolor:\s*green\b", re.I), "in a verdant shade"),
    (re.compile(r"\bcolor:\s*brown\b", re.I), "in an earth-toned shade"),
    (re.compile(r"\bcolor:\s*gr(?:a|e)y\b", re.I), "in a charcoal shade"),
    (re.compile(r"\bcolor:\s*purple\b", re.I), "in a violet shade"),
    (re.compile(r"\bcolor:\s*yellow\b", re.I), "in a sunny golden shade"),
    (re.compile(r"\b100\s*%\s*leather\b", re.I), "made entirely from genuine hide"),
    (re.compile(r"\bleather\b", re.I), "made from genuine hide"),
    (re.compile(r"\b100\s*%\s*cotton\b", re.I), "made entirely from natural fibre"),
    (re.compile(r"\b100\s*%\s*polyester\b", re.I), "made entirely from synthetic fibre"),
    (re.compile(r"\blight[ -]?weight\b", re.I), "easy to carry without much weight"),
    (re.compile(r"\bmachine wash(?: only)?\b", re.I), "safe to clean in a washing machine"),
    (re.compile(r"\bhand wash(?: only)?\b", re.I), "should be cleaned gently by hand"),
    (re.compile(r"\bpull[ -]?on closure\b", re.I), "slips on without fasteners"),
    (re.compile(r"\bzipper closure\b", re.I), "fastens with a zip"),
    (re.compile(r"\bbutton closure\b", re.I), "fastens using buttons"),
    (re.compile(r"\brubber sole\b", re.I), "an outsole made from grippy elastic material"),
    (re.compile(r"\bimported\b", re.I), "produced outside the domestic market"),
)


def _number_words(number: float) -> str:
    """Spell common non-negative prices deterministically; retain unusual values."""
    if not number.is_integer() or not 0 <= number < 1000:
        return f"{number:g}"
    value = int(number)
    ones = (
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if value < 20:
        return ones[value]
    if value < 100:
        return tens[value // 10] + (f"-{ones[value % 10]}" if value % 10 else "")
    remainder = value % 100
    return f"{ones[value // 100]} hundred" + (
        f" {_number_words(float(remainder))}" if remainder else ""
    )


def constraint_body(message: str) -> str | None:
    """Return the evaluator constraint body, excluding its surrounding template."""
    match = _MARKER_RE.search(message)
    return match.group("body").strip() if match else None


def _replace_constraint_body(message: str, body: str, marker: str | None = None) -> str:
    match = _MARKER_RE.search(message)
    if not match:
        return message
    selected_marker = marker if marker is not None else match.group("marker")
    return message[: match.start()] + selected_marker + " " + body.strip() + match.group("ending")


def _semantic_constraint(body: str) -> str:
    """Deterministically weaken literal overlap while preserving intended meaning."""
    result = body

    def composition(match: re.Match[str]) -> str:
        major = match.group("major_name").lower()
        minor = match.group("minor_name").lower()
        return f"mostly {major} with a small amount of {minor}"

    result = _PERCENT_PAIR_RE.sub(composition, result)
    result = _BUDGET_RE.sub(
        lambda match: f"priced near {_number_words(float(match.group(1)))} dollars",
        result,
    )
    for pattern, replacement in _SEMANTIC_REPLACEMENTS:
        result = pattern.sub(replacement, result)

    # Generic fallback for short atomic material constraints not already handled.
    atomic = {
        "cotton": "a breathable natural plant fibre",
        "polyester": "a durable synthetic fibre",
        "nylon": "a strong lightweight synthetic fibre",
        "wool": "a warm natural fleece fibre",
        "spandex": "a stretchy elastic fibre",
        "rayon": "a soft regenerated cellulose fibre",
        "silk": "a smooth natural filament fibre",
        "fabric": "a soft woven textile",
    }
    stripped = result.strip(" .")
    if stripped.casefold() in atomic:
        result = atomic[stripped.casefold()]
    elif result == body:
        # Still alter an unrecognised constraint semantically conservatively by
        # expressing it as a user need; the literal value remains intact.
        result = f"it should offer this property in practical use: {body.strip()}"
    return result


def paraphrase(message: str, level: int, seed: str = "member2-day4") -> str:
    """Apply one deterministic adversarial level to an evaluator user message."""
    if level not in LEVELS:
        raise ValueError(f"unsupported adversarial level: {level}")
    if level == 0 or not message:
        return message

    rng = random.Random(f"{seed}\0{level}\0{message}")
    result = message
    original_body = constraint_body(message)
    opening = _OPENING_RE.match(result)
    if opening:
        templates = (
            "Please help me find {category}{rest}",
            "I'm hoping to get {category}{rest}",
            "Could you show me {category}{rest}",
        )
        result = templates[rng.randrange(len(templates))].format(**opening.groupdict())
    if result.endswith("."):
        result = result[:-1] + rng.choice((".", "!", "..."))

    body = constraint_body(result)
    if level >= 2 and body is not None:
        marker = rng.choice(("My main requirement is:", "The important part is:"))
        result = _replace_constraint_body(result, body, marker)
    if level >= 3:
        if original_body is not None:
            result = re.sub(
                r"(?:My main requirement is:|The important part is:)\s*"
                + re.escape(original_body),
                f"What would suit me is {original_body}",
                result,
                count=1,
                flags=re.I,
            )
        result = re.sub(
            r"Actually,\s*ignore my earlier preference\.\s*",
            "Please replace what I said before; ",
            result,
            flags=re.I,
        )
        result = re.sub(
            r"For that,\s*what matters is:\s*",
            "Another thing I care about is ",
            result,
            flags=re.I,
        )
    if level == 4:
        # Work from the original body so Level 4 changes the actual constraint,
        # not merely the surrounding Level 3 phrasing.
        if original_body is not None:
            semantic = _semantic_constraint(original_body)
            # Level 3 may have removed the known marker, so replace the literal
            # body wherever it occurs rather than relying on marker parsing.
            result = result.replace(original_body, semantic)
    return " ".join(result.split())


def constraint_changed(before: str, after: str) -> bool:
    """Compare actual constraint text, not the surrounding simulator template."""
    before_body = constraint_body(before)
    if before_body is None:
        return False
    after_body = constraint_body(after)
    if after_body is not None:
        return before_body.casefold() != after_body.casefold()
    # Stronger levels intentionally remove markers; locate the original body to
    # distinguish template-only rewriting from semantic body rewriting.
    return before_body.casefold() not in after.casefold()


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    """Calculate token cost only when both explicit prices are supplied."""
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    return (
        (input_tokens / 1_000_000) * input_cost_per_million
        + (output_tokens / 1_000_000) * output_cost_per_million
    )


def summarize_llm_telemetry(
    records: list[dict],
    *,
    call_count: int,
    sample_count: int,
    total_wall_seconds: float | str,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> dict:
    """Aggregate the non-sensitive telemetry exposed by ``src.llm``."""
    input_tokens = sum(int(record.get("prompt_tokens", 0)) for record in records)
    output_tokens = sum(int(record.get("completion_tokens", 0)) for record in records)
    total_tokens = input_tokens + output_tokens
    latencies_ms = [
        max(0.0, float(record.get("latency_seconds", 0.0))) * 1000
        for record in records
    ]
    requested_models = sorted(
        {
            str(record["requested_model"])
            for record in records
            if record.get("requested_model")
        }
    )
    served_models = sorted(
        {
            str(record["served_model"])
            for record in records
            if record.get("served_model")
        }
    )
    estimated_total = estimate_cost(
        input_tokens,
        output_tokens,
        input_cost_per_million,
        output_cost_per_million,
    )
    reported_total = sum(max(0.0, float(record.get("cost", 0.0))) for record in records)
    denominator = sample_count if sample_count > 0 else 1
    return {
        "model_id": ";".join(requested_models) or "none",
        "served_model_id": ";".join(served_models) or "none",
        "llm_calls": call_count,
        "telemetry_records": len(records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_session": total_tokens / denominator,
        "p50_llm_latency_ms": _percentile(latencies_ms, 0.50),
        "p95_llm_latency_ms": _percentile(latencies_ms, 0.95),
        "total_wall_seconds": total_wall_seconds,
        "reported_api_cost_total": reported_total,
        "estimated_cost_per_session": (
            estimated_total / denominator if estimated_total is not None else "not computed"
        ),
        "projected_cost_per_million_sessions": (
            estimated_total / denominator * 1_000_000
            if estimated_total is not None
            else "not computed"
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _score_fields(metrics: dict) -> dict[str, float | None]:
    if metrics.get("mttc") is None:
        return {"efficiency": None, "technical_score": None}
    efficiency = max(0.0, min(1.0, (11.0 - float(metrics["mttc"])) / 10.0))
    score = 0.50 * metrics["hit_rate_at_10"] + 0.30 * metrics["mrr"] + 0.20 * efficiency
    return {"efficiency": round(efficiency, 6), "technical_score": round(score, 6)}


def evaluate_level(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    level: int,
) -> dict:
    """Mirror the official evaluator with message perturbation and timing only."""
    sessions: list[dict] = []
    latencies_ms: list[float] = []
    traces: dict[str, list[dict]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    wall_start = time.perf_counter()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"adversarial_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        original_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed
        )
        hit_turn: int | None = None
        best_rank: int | None = None
        traces[sample_id] = []
        for turn in range(1, MAX_TURNS + 1):
            sent_message = paraphrase(original_message, level, seed=f"{sample_id}:{turn}")
            started = time.perf_counter()
            try:
                response = agent.respond(session_id, sent_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            latencies_ms.append((time.perf_counter() - started) * 1000)
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            usage = response.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                    prompt_tokens += usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                    completion_tokens += usage["completion_tokens"]
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            traces[sample_id].append(
                {
                    "turn": turn,
                    "original_message": original_message,
                    "perturbed_message": sent_message,
                    "constraint_changed": constraint_changed(original_message, sent_message),
                    "ranked": ranked,
                }
            )
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                original_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                original_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        sessions.append(
            {
                "sample_id": sample_id,
                "scenario_type": sample["scenario_type"],
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    wall_seconds = time.perf_counter() - wall_start
    overall = metric_summary(sessions)
    derived = _score_fields(overall)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": derived["efficiency"],
        "recommended_technical_score": derived["technical_score"],
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
        "traces": traces,
        "latency": {
            "turn_count": len(latencies_ms),
            "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "total_wall_seconds": wall_seconds,
        },
    }


def compare_sessions(baseline: dict, current: dict, level: int) -> tuple[dict, list[dict]]:
    baseline_rows = {row["sample_id"]: row for row in baseline["sessions"]}
    deltas: list[dict] = []
    summary = defaultdict(int)
    rank_improvements: list[int] = []
    rank_losses: list[int] = []
    for row in current["sessions"]:
        before = baseline_rows[row["sample_id"]]
        before_rank = before["best_rank"]
        after_rank = row["best_rank"]
        if before_rank is not None and after_rank is None:
            movement = "disappeared"
            summary["target_disappeared"] += 1
        elif before_rank is None and after_rank is not None:
            movement = "improved"
            summary["target_rank_improved"] += 1
        elif before_rank is not None and after_rank is not None and after_rank < before_rank:
            movement = "improved"
            summary["target_rank_improved"] += 1
            rank_improvements.append(before_rank - after_rank)
        elif before_rank is not None and after_rank is not None and after_rank > before_rank:
            movement = "worsened"
            summary["target_rank_worsened"] += 1
            rank_losses.append(after_rank - before_rank)
        else:
            movement = "unchanged"
            summary["target_rank_unchanged"] += 1
        before_quality = (int(before["hit"]), before["reciprocal_rank"], -(before["first_hit_turn"] or 11))
        after_quality = (int(row["hit"]), row["reciprocal_rank"], -(row["first_hit_turn"] or 11))
        outcome = "improved" if after_quality > before_quality else "worsened" if after_quality < before_quality else "unchanged"
        summary[f"sessions_{outcome}"] += 1
        deltas.append(
            {
                "level": level,
                "sample_id": row["sample_id"],
                "scenario_type": row["scenario_type"],
                "outcome": outcome,
                "rank_movement": movement,
                "baseline_rank": before_rank,
                "perturbed_rank": after_rank,
                "baseline_hit_turn": before["first_hit_turn"],
                "perturbed_hit_turn": row["first_hit_turn"],
            }
        )
    summary["average_rank_improvement"] = statistics.fmean(rank_improvements) if rank_improvements else 0.0
    summary["average_rank_loss"] = statistics.fmean(rank_losses) if rank_losses else 0.0
    return dict(summary), deltas


def _failure_reason(original: str, perturbed: str) -> str:
    if "ignore my earlier preference" in original.casefold() and "ignore my earlier preference" not in perturbed.casefold():
        return "override phrase changed, so the agent's freshness reset may not be recognized"
    if constraint_changed(original, perturbed):
        return "constraint wording changed, weakening exact-match and BM25 lexical overlap"
    return "surface/template wording changed, likely affecting category or constraint parsing"


def representative_examples(result: dict, level: int, limit: int) -> list[dict]:
    candidates: list[dict] = []
    for sample_id, turns in result["traces"].items():
        for trace in turns:
            if trace["original_message"] != trace["perturbed_message"]:
                candidates.append({"level": level, "sample_id": sample_id, **{k: trace[k] for k in ("turn", "original_message", "perturbed_message", "constraint_changed")}})
    candidates.sort(key=lambda row: (not row["constraint_changed"], row["sample_id"], row["turn"]))
    return candidates[:limit]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _result_row(level: int, result: dict, baseline_score: float, movement: dict) -> dict:
    score = result["recommended_technical_score"]
    absolute = score - baseline_score
    return {
        "level": level,
        "sample_count": result["sample_count"],
        "technical_score": score,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "absolute_score_delta": absolute,
        "relative_score_delta_percent": (absolute / baseline_score * 100) if baseline_score else 0.0,
        **{key: movement.get(key, 0) for key in (
            "sessions_improved", "sessions_unchanged", "sessions_worsened",
            "target_rank_improved", "target_rank_unchanged", "target_rank_worsened",
            "target_disappeared", "average_rank_improvement", "average_rank_loss",
        )},
    }


def _markdown_report(
    result_rows: list[dict],
    results: dict[int, dict],
    examples: list[dict],
    failures: list[dict],
    cost_rows: list[dict],
) -> str:
    baseline = results[0]
    lines = [
        "# Member 2 Day 4 robustness report", "",
        "## Frozen baseline verification", "",
        f"Level 0 reproduced TechnicalScore **{baseline['recommended_technical_score']:.6f}** "
        f"on {baseline['sample_count']} sessions. The shipped response reported "
        f"{baseline['reported_token_usage']['total_tokens']} tokens.",
        "The assignment's `0.877011` gate is historical; current main's frozen "
        "offline test is `0.891111` after the three-state exact-constraint correction.", "",
        "## Adversarial levels", "",
        "| Level | Score | Hit@10 | MRR | MTTC | Abs. delta | Rel. delta | Improved | Unchanged | Worsened | Disappeared |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['level']} | {row['technical_score']:.6f} | {row['hit_rate_at_10']:.6f} | "
            f"{row['mrr']:.6f} | {row['mttc']:.6f} | {row['absolute_score_delta']:+.6f} | "
            f"{row['relative_score_delta_percent']:+.3f}% | {row['sessions_improved']} | "
            f"{row['sessions_unchanged']} | {row['sessions_worsened']} | {row['target_disappeared']} |"
        )
    lines.extend(["", "## Per-scenario robustness", "", "| Level | Scenario | N | Hit@10 | MRR | MTTC |", "|---:|---|---:|---:|---:|---:|"])
    for level, result in results.items():
        for scenario in SCENARIOS:
            metrics = result["scenario_metrics"].get(scenario)
            if metrics:
                lines.append(f"| {level} | {scenario} | {metrics['sample_count']} | {metrics['hit_rate_at_10']:.6f} | {metrics['mrr']:.6f} | {metrics['mttc']:.6f} |")
    lines.extend(["", "## Representative before/after examples", ""])
    for example in examples:
        lines.extend([
            f"### Level {example['level']} — {example['sample_id']} turn {example['turn']}", "",
            f"- Before: {example['original_message']}",
            f"- After: {example['perturbed_message']}",
            f"- Constraint changed: {'YES' if example['constraint_changed'] else 'NO'}", "",
        ])
    lines.extend(["## Level 4 failure examples", ""])
    if not failures:
        lines.extend(["No worsened Level 4 sessions were found.", ""])
    for failure in failures:
        lines.extend([
            f"### {failure['sample_id']}", "",
            f"- Original message: {failure['original_message']}",
            f"- Perturbed message: {failure['perturbed_message']}",
            f"- Original target rank: {failure['baseline_rank']}",
            f"- Perturbed target rank: {failure['perturbed_rank']}",
            f"- Likely cause: {failure['likely_cause']}", "",
        ])
    lines.extend(["## Latency", "", "Only `Agent.respond()` is included in per-turn latency; Agent/catalog construction is excluded. Total wall time covers the evaluator loop.", "", "| Level | Turns | Mean ms | p50 ms | p95 ms | Wall seconds |", "|---:|---:|---:|---:|---:|---:|"])
    for level, result in results.items():
        latency = result["latency"]
        lines.append(f"| {level} | {latency['turn_count']} | {latency['mean_ms']:.3f} | {latency['p50_ms']:.3f} | {latency['p95_ms']:.3f} | {latency['total_wall_seconds']:.3f} |")
    lines.extend(["", "## LLM on/off cost", "", "| Mode | Available | Score | Model | Calls | Input tokens | Output tokens | p50 ms | p95 ms | Cost/session | Cost/1M sessions |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in cost_rows:
        lines.append(
            f"| {row['mode']} | {row['available']} | {row['technical_score']} | {row['model_id']} | "
            f"{row['llm_calls']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"{row['p50_llm_latency_ms']} | {row['p95_llm_latency_ms']} | "
            f"{row['estimated_cost_per_session']} | "
            f"{row['projected_cost_per_million_sessions']} |"
        )
    lines.extend([
        "", "LLM-off is always measured with explicit false Agent flags and a reset telemetry state. LLM-on is only run when `--measure-llm-on` is supplied; this prevents an ordinary robustness run from making network calls. Token-price projections remain `not computed` unless explicit prices are supplied.",
        "", "## Previous rejected experiments", "",
        "- Day 2 category filtering: browsing R@10 remained 0.0375 before and after. Rejected.",
        "- Day 3 near-duplicate suppression: only 7/200 sessions had a pair at 0.80; best score delta was about +0.00005. Rejected / do not ship.",
        "", "## Limitations", "",
        "- Levels 1–4 are deterministic synthetic perturbations, not an LLM paraphrase distribution.",
        "- Level 4 uses a bounded semantic rule set; unrecognised long constraints retain their literal body inside a natural-language wrapper.",
        "- Session comparison uses the evaluator's first successful target rank, not every per-turn rank after conversion.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Agent under deterministic paraphrases")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument(
        "--measure-llm-on",
        action="store_true",
        help="Explicitly run one network-enabled LLM configuration",
    )
    parser.add_argument(
        "--llm-on-level",
        type=int,
        choices=LEVELS,
        default=4,
        help="Adversarial level used for the optional LLM-on cost run",
    )
    args = parser.parse_args()
    invalid = set(args.levels) - set(LEVELS)
    if invalid or 0 not in args.levels:
        parser.error("levels must include control Level 0 and use only 0,1,2,3,4")

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    results: dict[int, dict] = {}
    llm.reset_state()
    for level in args.levels:
        print(f"Running adversarial Level {level}...", flush=True)
        results[level] = evaluate_level(
            Agent(
                args.catalog,
                enable_llm_normalize=False,
                enable_llm_override=False,
                enable_llm_message=False,
                enable_confidence=False,
            ),
            samples,
            catalog_ids,
            categories,
            products,
            level,
        )
        if level == 0 and results[level]["recommended_technical_score"] != FROZEN_SCORE:
            raise SystemExit(
                f"STOP: Level 0 score {results[level]['recommended_technical_score']:.6f} "
                f"does not equal frozen {FROZEN_SCORE:.6f}"
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = results[0]
    result_rows: list[dict] = []
    delta_rows: list[dict] = []
    movement_by_level: dict[int, dict] = {}
    for level, result in results.items():
        movement, deltas = compare_sessions(baseline, result, level)
        movement_by_level[level] = movement
        result_rows.append(_result_row(level, result, FROZEN_SCORE, movement))
        delta_rows.extend(deltas)

    examples: list[dict] = []
    for level in sorted(results):
        if level:
            examples.extend(representative_examples(results[level], level, args.examples))
    level4_deltas = {row["sample_id"]: row for row in delta_rows if row["level"] == 4 and row["outcome"] == "worsened"}
    failures: list[dict] = []
    if 4 in results:
        for sample_id, delta in list(level4_deltas.items())[: args.examples]:
            traces = results[4]["traces"][sample_id]
            changed = next((trace for trace in traces if trace["constraint_changed"]), traces[-1])
            failures.append({
                **delta,
                "original_message": changed["original_message"],
                "perturbed_message": changed["perturbed_message"],
                "likely_cause": _failure_reason(changed["original_message"], changed["perturbed_message"]),
            })

    latency_rows = [{"level": level, **result["latency"]} for level, result in results.items()]
    offline_summary = summarize_llm_telemetry(
        llm.telemetry(),
        call_count=llm.CALL_COUNT,
        sample_count=len(samples),
        total_wall_seconds=baseline["latency"]["total_wall_seconds"],
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    # Zero inference always has zero cost, even when no price parameters exist.
    if offline_summary["total_tokens"] == 0:
        offline_summary["estimated_cost_per_session"] = 0.0
        offline_summary["projected_cost_per_million_sessions"] = 0.0
    cost_rows = [
        {
            "mode": "llm_off",
            "available": "yes",
            "technical_score": baseline["recommended_technical_score"],
            **offline_summary,
        }
    ]

    if args.measure_llm_on:
        if not llm.setting("OPENROUTER_API_KEY") or not llm.setting("OPENROUTER_MODEL"):
            raise SystemExit(
                "--measure-llm-on requires OPENROUTER_API_KEY and "
                "OPENROUTER_MODEL (a concrete :free model)"
            )
        print(f"Running explicit LLM-on Level {args.llm_on_level}...", flush=True)
        llm.reset_state()
        llm_started = time.perf_counter()
        llm_result = evaluate_level(
            Agent(
                args.catalog,
                enable_llm_normalize=True,
                enable_llm_override=True,
                enable_llm_message=False,
                enable_confidence=False,
            ),
            samples,
            catalog_ids,
            categories,
            products,
            args.llm_on_level,
        )
        llm_wall = time.perf_counter() - llm_started
        online_summary = summarize_llm_telemetry(
            llm.telemetry(),
            call_count=llm.CALL_COUNT,
            sample_count=len(samples),
            total_wall_seconds=llm_wall,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )
        cost_rows.append(
            {
                "mode": f"llm_on_level_{args.llm_on_level}",
                "available": "yes",
                "technical_score": llm_result["recommended_technical_score"],
                **online_summary,
            }
        )
    else:
        cost_rows.append(
            {
                "mode": "llm_on",
                "available": "not run; pass --measure-llm-on",
                "technical_score": "not measured",
                "model_id": llm.setting("OPENROUTER_MODEL") or "not configured",
                "served_model_id": "not measured",
                "llm_calls": "not measured",
                "telemetry_records": "not measured",
                "input_tokens": "not measured",
                "output_tokens": "not measured",
                "total_tokens": "not measured",
                "tokens_per_session": "not measured",
                "p50_llm_latency_ms": "not measured",
                "p95_llm_latency_ms": "not measured",
                "total_wall_seconds": "not measured",
                "reported_api_cost_total": "not measured",
                "estimated_cost_per_session": "not computed",
                "projected_cost_per_million_sessions": "not computed",
            }
        )

    _write_csv(output_dir / "adversarial_results.csv", result_rows)
    _write_csv(output_dir / "adversarial_session_deltas.csv", delta_rows)
    _write_csv(output_dir / "latency_results.csv", latency_rows)
    _write_csv(output_dir / "cost_results.csv", cost_rows)
    (output_dir / "adversarial_examples.json").write_text(
        json.dumps({"examples": examples, "level4_failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "member2_day4_report.md").write_text(
        _markdown_report(result_rows, results, examples, failures, cost_rows), encoding="utf-8"
    )

    print("\nLevel  Score     Hit@10   MRR      MTTC    Delta")
    for row in result_rows:
        print(f"{row['level']:<6} {row['technical_score']:<9.6f} {row['hit_rate_at_10']:<8.6f} {row['mrr']:<8.6f} {row['mttc']:<7.3f} {row['absolute_score_delta']:+.6f}")
    print(f"\nWrote robustness evidence to {output_dir}")


if __name__ == "__main__":
    main()

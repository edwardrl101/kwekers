"""Educational, single-session trace of evaluator.local_evaluator.evaluate.

This file is intentionally separate from the official evaluator. It mirrors
the evaluator's turn loop so intermediate state can be displayed without
adding debug prints to scoring code.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as local_evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402


def heading(title: str, character: str = "=") -> None:
    width = 88
    print()
    print(character * width)
    print(title)
    print(character * width)


def pretty(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def show_evaluate_source() -> None:
    """Print the current evaluate() implementation with source line numbers."""
    source, first_line = inspect.getsourcelines(local_evaluator.evaluate)
    heading("SOURCE: evaluator.local_evaluator.evaluate()")
    for offset, line in enumerate(source):
        print(f"{first_line + offset:4} | {line}", end="")


def choose_sample(
    samples: list[dict],
    sample_id: str | None,
    scenario: str | None,
    index: int,
) -> dict:
    if sample_id:
        for sample in samples:
            if str(sample.get("sample_id")) == sample_id:
                return sample
        raise ValueError(f"No sample has sample_id={sample_id!r}")

    candidates = [
        sample
        for sample in samples
        if scenario is None or sample.get("scenario_type") == scenario
    ]
    if not candidates:
        raise ValueError(f"No samples match scenario={scenario!r}")
    if index < 0 or index >= len(candidates):
        raise ValueError(
            f"Index {index} is outside the selected range 0..{len(candidates) - 1}"
        )
    return candidates[index]


def safe_response(
    agent: Agent,
    session_id: str,
    user_message: str,
    turn: int,
) -> tuple[dict, str | None]:
    """Apply the same response-failure rules as the real evaluator."""
    error: str | None = None
    try:
        response = agent.respond(
            session_id,
            user_message,
            turn,
            local_evaluator.TOP_K,
        )
    except Exception as exception:  # educational trace should expose the error
        error = f"{type(exception).__name__}: {exception}"
        response = {"message": "", "ask_attribute": None, "recommendations": []}

    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        error = error or "Invalid response: expected a dict with a string message"
        response = {"message": "", "ask_attribute": None, "recommendations": []}
    return response, error


def print_recommendations(
    ranked: list[str],
    target: str,
    products: dict[str, dict],
) -> None:
    print("\nNormalized recommendations (the list that is actually scored):")
    if not ranked:
        print("  <none>")
        return
    print(f"  {'Rank':<6}{'Target?':<10}{'parent_asin':<18}Title")
    print(f"  {'-' * 5:<6}{'-' * 7:<10}{'-' * 15:<18}{'-' * 35}")
    for rank, parent_asin in enumerate(ranked, start=1):
        marker = "YES" if parent_asin == target else ""
        title = str(products.get(parent_asin, {}).get("title", "<unknown>"))
        if len(title) > 52:
            title = title[:49] + "..."
        print(f"  {rank:<6}{marker:<10}{parent_asin:<18}{title}")


def trace_sample(
    agent: Agent,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    max_turns: int,
) -> dict:
    """Run one annotated conversation using the evaluator's real helpers."""
    session_id = f"trace_{sample['sample_id']}"
    target = str(sample["ground_truth"]["parent_asin"])
    target_product = products[target]

    heading("1. RAW EVALUATION SAMPLE")
    print(pretty(sample))

    heading("2. SECRET TARGET (visible here only for learning)")
    print(f"parent_asin : {target}")
    print(f"title       : {target_product.get('title', '<missing>')}")
    print(f"categories  : {pretty(categories.get(target, []))}")
    print(f"price       : {target_product.get('price', '<missing>')}")

    intent_card, behavior = local_evaluator.materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}

    heading("3. MATERIALIZED HIDDEN SIMULATOR STATE")
    print("intent_card:")
    print(pretty(intent_card))
    print("\nbehavior:")
    print(pretty(behavior))

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    category = local_evaluator.coarse_category(categories.get(target, []))
    user_message = local_evaluator.initial_message(
        effective_sample,
        category,
        disclosed,
    )
    agent.reset(session_id, sample["user_profile"])

    heading("4. INITIAL CONVERSATION STATE")
    print(f"session_id       : {session_id}")
    print(f"coarse category  : {category}")
    print(f"disclosed        : {sorted(disclosed)}")
    print(f"boundary_used    : {boundary_used}")
    print(f"override_applied : {override_applied}")
    print(f"first message    : {user_message}")

    prompt_tokens = 0
    completion_tokens = 0
    hit_turn: int | None = None
    best_rank: int | None = None

    for turn in range(1, max_turns + 1):
        heading(f"TURN {turn}", "-")
        print("INPUT TO Agent.respond()")
        print(f"  session_id : {session_id}")
        print(f"  turn       : {turn}")
        print(f"  top_k      : {local_evaluator.TOP_K}")
        print(f"  user       : {user_message}")

        response, error = safe_response(agent, session_id, user_message, turn)
        print("\nRAW AGENT RESPONSE")
        print(pretty(response))
        if error:
            print(f"\nEvaluator fallback reason: {error}")

        usage = response.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                prompt_tokens += usage["prompt_tokens"]
            if (
                isinstance(usage.get("completion_tokens"), int)
                and usage["completion_tokens"] >= 0
            ):
                completion_tokens += usage["completion_tokens"]

        ranked = local_evaluator.normalize_recommendations(
            response.get("recommendations"), catalog_ids
        )
        print_recommendations(ranked, target, products)

        target_present = target in ranked
        eligible_hit = override_applied and target_present
        print("\nSCORING DECISION")
        print(f"  target present?   : {target_present}")
        print(f"  override applied? : {override_applied}")
        print(f"  eligible hit?     : {eligible_hit}")

        if eligible_hit:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            print(f"  result            : HIT at rank {best_rank}; session stops")
            break

        if turn == max_turns:
            print("  result            : no more traced turns")
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get(
                    "message",
                    "Actually, please ignore my earlier preference.",
                )
            )
            reply_source = "scheduled intent override"
        else:
            user_message, boundary_used = local_evaluator.customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )
            reply_source = "customer_reply()"

        print("\nNEXT-TURN STATE")
        print(f"  source             : {reply_source}")
        print(f"  disclosed          : {sorted(disclosed)}")
        print(f"  boundary_used      : {boundary_used}")
        print(f"  override_applied   : {override_applied}")
        print(f"  next user message  : {user_message}")

    session_result = {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
    }
    overall = local_evaluator.metric_summary([session_result])
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    result = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "session": session_result,
    }

    heading("5. FINAL ONE-SESSION RESULT")
    print(pretty(result))
    return result


def main() -> None:
    # Windows PowerShell may default to CP-1252 even though catalog text is
    # UTF-8. Reconfigure the stream so symbols in titles/features cannot crash
    # this debugging tool.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Pretty-print one educational evaluation trace"
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--sample-id", help="Trace an exact sample ID, such as public_0001")
    parser.add_argument(
        "--scenario",
        choices=("buying", "browsing", "intent_override", "boundary"),
        help="Filter samples by scenario before applying --index",
    )
    parser.add_argument("--index", type=int, default=0, help="Zero-based sample index")
    parser.add_argument("--max-turns", type=int, choices=range(1, 11), default=10)
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Print the current evaluate() source with line numbers first",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Print evaluate() source and skip the runtime trace",
    )
    args = parser.parse_args()

    if args.show_source or args.source_only:
        show_evaluate_source()
    if args.source_only:
        return

    samples = local_evaluator.load_jsonl(args.dataset)
    sample = choose_sample(samples, args.sample_id, args.scenario, args.index)
    catalog_ids, categories, products = local_evaluator.catalog_index(args.catalog)
    trace_sample(
        Agent(args.catalog),
        sample,
        catalog_ids,
        categories,
        products,
        args.max_turns,
    )


if __name__ == "__main__":
    main()

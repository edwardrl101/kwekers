"""Interactive Member-5 demo CLI and transcript generator.

Default mode is intentionally simple for screen recording:

    python scripts/demo.py

Type one customer message per turn. The CLI prints the assistant explanation,
``ask_attribute``, confidence, and the ranked recommendations with catalog
metadata. No web UI is required.

To generate the three required reproducible example transcripts from the
released simulator:

    python scripts/demo.py --save-examples

This writes buying, browsing, and intent-override transcripts under
``demo_transcripts/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
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
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402


SCENARIOS = ("buying", "browsing", "intent_override")


def _price_text(product: dict) -> str:
    value = product.get("price")
    if isinstance(value, (int, float)):
        return f"${float(value):.2f}"
    return "price n/a"


def _session_confidence(agent: Agent, session_id: str) -> float:
    session = agent._sessions.get(str(session_id), {})
    try:
        return max(0.0, min(1.0, float(session.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _product_rows(agent: Agent, response: dict) -> list[tuple[int, str, dict]]:
    rows: list[tuple[int, str, dict]] = []
    for rank, item in enumerate(response.get("recommendations", []), start=1):
        if not isinstance(item, dict):
            continue
        asin = str(item.get("parent_asin", "")).strip()
        if not asin:
            continue
        rows.append((rank, asin, agent._catalog.get(asin, {})))
    return rows


def print_response(agent: Agent, session_id: str, response: dict) -> None:
    """Pretty-print one agent turn for a human-facing demo."""

    print("\nAGENT")
    print(f"  {response.get('message', '')}")
    print(f"  ask_attribute: {response.get('ask_attribute')}")
    print(f"  confidence: {_session_confidence(agent, session_id):.3f}")
    print("  recommendations:")
    for rank, asin, product in _product_rows(agent, response):
        title = str(product.get("title") or "(title unavailable)")
        print(f"    {rank:>2}. {title[:88]} | {_price_text(product)} | {asin}")


def interactive(agent: Agent, session_id: str = "demo") -> int:
    """Run a manual multi-turn session suitable for screen recording."""

    agent.reset(session_id, {})
    print("Kwekers interactive demo")
    print("Type the shopper message for each turn. Type 'quit' to stop.\n")

    for turn in range(1, MAX_TURNS + 1):
        try:
            message = input(f"CUSTOMER turn {turn}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.casefold() in {"quit", "exit", "q"}:
            return 0
        if not message:
            print("Please enter a non-empty customer message.")
            continue

        response = agent.respond(session_id, message, turn, TOP_K)
        print_response(agent, session_id, response)

    return 0


def _choose_samples(samples: Iterable[dict]) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for sample in samples:
        scenario = str(sample.get("scenario_type", ""))
        if scenario in SCENARIOS and scenario not in chosen:
            chosen[scenario] = sample
        if len(chosen) == len(SCENARIOS):
            break
    missing = [scenario for scenario in SCENARIOS if scenario not in chosen]
    if missing:
        raise RuntimeError(f"public set is missing required demo scenarios: {missing}")
    return chosen


def _render_recommendations(
    agent: Agent,
    response: dict,
    target: str,
    *,
    limit: int = 5,
) -> list[str]:
    lines: list[str] = []
    for rank, asin, product in _product_rows(agent, response)[:limit]:
        marker = " **← target**" if asin == target else ""
        title = str(product.get("title") or "(title unavailable)")
        lines.append(
            f"{rank}. {title[:120]} — {_price_text(product)} — `{asin}`{marker}"
        )
    return lines


def simulate_sample(
    agent: Agent,
    sample: dict,
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> str:
    """Run one released simulator sample and return a Markdown transcript."""

    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample.get("user_profile", {}))

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    out = [
        f"# {str(sample['scenario_type']).replace('_', ' ').title()} demo transcript",
        "",
        f"- Public sample: `{sample['sample_id']}`",
        f"- Target ASIN (for evaluator/demo annotation only): `{target}`",
        "- Runtime mode: deterministic/offline; confidence enabled for display",
        "",
    ]

    for turn in range(1, MAX_TURNS + 1):
        out.extend([f"## Turn {turn}", "", f"**Customer:** {user_message}", ""])
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), set(agent._catalog))
        target_rank = ranked.index(target) + 1 if target in ranked else None
        confidence = _session_confidence(agent, session_id)

        out.extend(
            [
                f"**Agent:** {response.get('message', '')}",
                "",
                f"- ask_attribute: `{response.get('ask_attribute')}`",
                f"- confidence: `{confidence:.3f}`",
                f"- target rank this turn: `{target_rank if target_rank is not None else 'not in top 10'}`",
                "- top recommendations:",
            ]
        )
        rec_lines = _render_recommendations(agent, response, target)
        out.extend([f"  {line}" for line in rec_lines] or ["  (none)"])
        out.append("")

        # Match the evaluator's special rule: pre-override hits do not end an
        # intent-override session.
        if override_applied and target_rank is not None:
            out.append(f"**Evaluator outcome:** hit at turn {turn}, rank {target_rank}.")
            break
        if turn == MAX_TURNS:
            out.append("**Evaluator outcome:** no valid hit within 10 turns.")
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get(
                    "message", "Actually, please ignore my earlier preference."
                )
            )
        else:
            user_message, boundary_used = customer_reply(
                effective,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )

    return "\n".join(out).rstrip() + "\n"


def save_examples(
    agent: Agent,
    catalog_path: Path,
    public_set_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate the required buying/browsing/override saved transcripts."""

    samples = load_jsonl(public_set_path)
    _, categories, products = catalog_index(catalog_path)
    chosen = _choose_samples(samples)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for scenario in SCENARIOS:
        path = output_dir / f"{scenario}.md"
        path.write_text(
            simulate_sample(agent, chosen[scenario], categories, products),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Kwekers interactive demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="write buying/browsing/intent-override transcripts and exit",
    )
    parser.add_argument("--output-dir", default="demo_transcripts")
    parser.add_argument("--session-id", default="demo")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        parser.error(f"catalog not found: {catalog_path}")

    # Confidence only changes an unscored display signal/message; the ranked
    # recommendation path remains unchanged.
    agent = Agent(catalog_path, enable_confidence=True, enable_llm_message=False)

    if args.save_examples:
        public_set_path = Path(args.public_set)
        if not public_set_path.exists():
            parser.error(f"public set not found: {public_set_path}")
        written = save_examples(
            agent,
            catalog_path,
            public_set_path,
            Path(args.output_dir),
        )
        print("Saved demo transcripts:")
        for path in written:
            print(f"  {path}")
        return 0

    return interactive(agent, args.session_id)


if __name__ == "__main__":
    raise SystemExit(main())

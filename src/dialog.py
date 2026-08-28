"""Dialog state, scenario routing, and clarification policy for Kwekers.

Day-1 ownership: Sheng Yan.

This module intentionally has no runtime dependency on the evaluator.  The
``__main__`` diagnostics may import the public evaluator *only* to measure the
question policy against the released 200-session development set.

Public evaluator facts mirrored here:
- ``classify_constraint`` has a strict branch order.
- ``other`` bypasses classification and can reveal any two undisclosed items.
- ``category`` and ``brand`` are never classifier outputs.

The runtime-facing pieces are:
- :func:`classify_constraint`
- :class:`ScenarioRouter`
- :class:`SlotState`
- :class:`QuestionPolicy`
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


# Keep these values synchronized with evaluator/local_evaluator.py.
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)

USEFUL_CLASSIFIED_ATTRIBUTES = (
    "budget",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "feature",
)

# ``category`` and ``brand`` are deliberately absent: classify_constraint()
# never returns them. ``other`` is handled specially by customer_reply().
ALLOWED_ASK_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)

# Day-3 policy comparison names. Keep them stable so Agent and diagnostics agree.
POLICY_ALWAYS_OTHER = "always_other"
POLICY_OTHER_TWICE_ROTATE = "other_twice_rotate"
QUESTION_POLICY_MODES = (POLICY_ALWAYS_OTHER, POLICY_OTHER_TWICE_ROTATE)

# The evaluator's classifier can never produce brand/category. A valid policy
# must therefore stay inside this set (plus ``other``).
SAFE_ASK_ATTRIBUTES = frozenset((*USEFUL_CLASSIFIED_ATTRIBUTES, "other"))


# Message-shape detection.
KEY_REQUIREMENT_RE = re.compile(r"\bA key requirement is:\s*(.+?)(?:\.\s*$|$)", re.I)
MATTERS_RE = re.compile(r"\bFor that, what matters is:\s*(.+?)(?:\.\s*$|$)", re.I)
WHAT_I_NEED_RE = re.compile(r"\bWhat I need is:\s*(.+?)(?:\.\s*$|$)", re.I)
STILL_EXPLORING_RE = re.compile(r"\bstill exploring\b", re.I)
BOUNDARY_REPLY_RE = re.compile(r"\bI don't have a preference for\b", re.I)
NO_ADDITIONAL_RE = re.compile(r"\bI don't have an additional preference for\s+([a-z_]+)\b", re.I)
OVERRIDE_RE = re.compile(
    r"\b(?:ignore|forget)\s+(?:about\s+)?my\s+(?:earlier|previous|prior)\s+preference\b",
    re.I,
)
INITIAL_PREFIX_RE = re.compile(r"^I'm looking for .+?\.\s+(.+?)\s*$", re.I)


def _normalise(value: str) -> str:
    """Compact normalization for de-duplication; does not alter stored text."""

    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n").casefold()


def classify_constraint(value: str) -> str:
    """Mirror the public evaluator's classifier branch-for-branch.

    Ordering is semantically important.  In particular, budget is tested
    before material, so a mixed string such as ``"budget $40, cotton"`` is a
    budget constraint and cannot be retrieved by asking for ``material``.
    """

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


CLASSIFIER_TABLE = (
    {
        "priority": 1,
        "ask_attribute": "budget",
        "matches": 'contains "budget" OR /(?:\\$|<=|under)\\s*\\d/',
        "notes": "Checked first; wins over every later branch, including material.",
    },
    {
        "priority": 2,
        "ask_attribute": "material",
        "matches": "cotton/polyester/nylon/leather/wool/spandex/silk/rayon/fabric",
        "notes": "Only reached if the budget branch did not match.",
    },
    {
        "priority": 3,
        "ask_attribute": "color",
        "matches": "color/black/white/blue/red/pink/green",
        "notes": 'Generated forms like "color: brown" still match because they contain "color".',
    },
    {
        "priority": 4,
        "ask_attribute": "size",
        "matches": "size/sizing/width/wide/narrow",
        "notes": "Sizing and width language routes here.",
    },
    {
        "priority": 5,
        "ask_attribute": "style",
        "matches": "department/style/fit/sleeve/neck",
        "notes": 'A constraint such as "Department: Womens" is classified as style.',
    },
    {
        "priority": 6,
        "ask_attribute": "use_case",
        "matches": "hiking/running/gym/winter/outdoor/work",
        "notes": "Use-case keywords only if no earlier branch matched.",
    },
    {
        "priority": 7,
        "ask_attribute": "feature",
        "matches": "everything else",
        "notes": "Default/fallback classifier output.",
    },
    {
        "priority": "bypass",
        "ask_attribute": "other",
        "matches": "any undisclosed constraint (up to 2)",
        "notes": "customer_reply() bypasses classify_constraint() entirely.",
    },
    {
        "priority": "never",
        "ask_attribute": "brand",
        "matches": "none",
        "notes": "classify_constraint() never returns brand; wastes a normal turn.",
    },
    {
        "priority": "never",
        "ask_attribute": "category",
        "matches": "none",
        "notes": "classify_constraint() never returns category; wastes a normal turn.",
    },
)


@dataclass(slots=True)
class Constraint:
    """One compact remembered preference/constraint."""

    text: str
    attribute: str
    turn: int
    source: str
    active: bool = True
    soft_reason: str | None = None

    @property
    def key(self) -> str:
        return _normalise(self.text)


@dataclass
class SlotState:
    """Compact evolving dialog state.

    Active constraints accumulate instead of replacing each other.  On an
    intent override we demote the earlier preference to ``soft_constraints``
    rather than deleting it: the simulator's target ASIN does not change, so
    the old metadata can remain a weak ranking signal even after the shopper
    says to ignore it as a hard requirement.
    """

    session_id: str | None = None
    scenario: str = "unknown"
    constraints: list[Constraint] = field(default_factory=list)
    soft_constraints: list[Constraint] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    boundary_seen: bool = False
    override_count: int = 0
    other_constraint_replies: int = 0
    other_exhausted: bool = False
    last_user_message: str = ""
    _initial_override_key: str | None = None

    def reset(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.scenario = "unknown"
        self.constraints.clear()
        self.soft_constraints.clear()
        self.asked_attributes.clear()
        self.boundary_seen = False
        self.override_count = 0
        self.other_constraint_replies = 0
        self.other_exhausted = False
        self.last_user_message = ""
        self._initial_override_key = None

    @property
    def last_asked_attribute(self) -> str | None:
        return self.asked_attributes[-1] if self.asked_attributes else None

    def record_ask(self, ask_attribute: str) -> None:
        if ask_attribute not in ALLOWED_ASK_ATTRIBUTES:
            raise ValueError(f"unsupported ask_attribute: {ask_attribute!r}")
        self.asked_attributes.append(ask_attribute)

    def active_by_attribute(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for item in self.constraints:
            if item.active:
                grouped.setdefault(item.attribute, []).append(item.text)
        return grouped

    def to_context(self) -> dict:
        """Small serializable state for prompts/logging instead of raw transcript."""

        return {
            "scenario": self.scenario,
            "active": self.active_by_attribute(),
            "soft": [item.text for item in self.soft_constraints],
            "boundary_seen": self.boundary_seen,
            "override_count": self.override_count,
            "other_constraint_replies": self.other_constraint_replies,
            "other_exhausted": self.other_exhausted,
            "asked_attributes": self.asked_attributes[-5:],
        }

    def update(self, user_message: str, turn: int) -> list[Constraint]:
        """Parse one user message and merge any newly revealed constraints.

        Returns the newly-added active constraints from this message.
        """

        message = str(user_message or "")
        self.last_user_message = message

        # Scenario tracking is stateful so later generic replies do not erase a
        # scenario detected on turn 1.
        self.scenario = route_scenario(message, turn=turn, current=self.scenario)

        if BOUNDARY_REPLY_RE.search(message):
            self.boundary_seen = True

        no_additional = NO_ADDITIONAL_RE.search(message)
        if no_additional and self.last_asked_attribute == "other":
            self.other_exhausted = True

        is_override = bool(OVERRIDE_RE.search(message))
        extracted = extract_constraints(message, scenario=self.scenario, turn=turn)

        # The initial override message contains the old soft preference without
        # a label. Remember which item it is so the later "ignore my earlier
        # preference" can demote the correct preference even if other facts
        # were learned in turns 1-2.
        if turn == 1 and self.scenario == "intent_override" and extracted:
            self._initial_override_key = _normalise(extracted[0][0])

        if is_override:
            self.override_count += 1
            preferred_attr = classify_constraint(extracted[0][0]) if extracted else None
            self._demote_overridden_preference(preferred_attr=preferred_attr, turn=turn)

        new_items: list[Constraint] = []
        existing = {item.key for item in self.constraints if item.active}
        existing.update(item.key for item in self.soft_constraints)
        for text, source in extracted:
            key = _normalise(text)
            if not key or key in existing:
                continue
            item = Constraint(
                text=text,
                attribute=classify_constraint(text),
                turn=turn,
                source=source,
            )
            self.constraints.append(item)
            new_items.append(item)
            existing.add(key)

        # Only a true constraint-bearing reply to an ``other`` question counts
        # toward draining the card. Boundary replies and forced override turns
        # deliberately do not count.
        if (
            self.last_asked_attribute == "other"
            and MATTERS_RE.search(message)
            and new_items
        ):
            self.other_constraint_replies += 1

        return new_items

    def _demote_overridden_preference(self, preferred_attr: str | None, turn: int) -> Constraint | None:
        candidate: Constraint | None = None

        # Best signal: the exact old preference shown in the special turn-1
        # intent-override opening message.
        if self._initial_override_key:
            for item in reversed(self.constraints):
                if item.active and item.key == self._initial_override_key:
                    candidate = item
                    break

        # Robust fallback for paraphrased/private variants: replace only the
        # conflicting slot if we can identify one.
        if candidate is None and preferred_attr:
            for item in reversed(self.constraints):
                if item.active and item.attribute == preferred_attr:
                    candidate = item
                    break

        # Last fallback: demote just the most recent preference, never wipe the
        # entire state.
        if candidate is None:
            candidate = next((item for item in reversed(self.constraints) if item.active), None)

        if candidate is None:
            return None

        candidate.active = False
        candidate.soft_reason = f"demoted by override at turn {turn}"
        self.soft_constraints.append(candidate)
        return candidate


class ScenarioRouter:
    """Small stateful wrapper around :func:`route_scenario`."""

    def __init__(self) -> None:
        self.scenario = "unknown"

    def reset(self) -> None:
        self.scenario = "unknown"

    def update(self, message: str, turn: int) -> str:
        self.scenario = route_scenario(message, turn=turn, current=self.scenario)
        return self.scenario


def route_scenario(message: str, turn: int, current: str = "unknown") -> str:
    """Route the public simulator's message templates.

    Turn-1 contract:
    - ``A key requirement is:`` -> buying
    - ``still exploring`` -> browsing-or-boundary
    - neither -> intent_override

    Boundary is distinguishable on turn 2 when the simulator returns
    ``I don't have a preference for ...``.  A non-boundary reply resolves the
    provisional state to browsing.
    """

    message = str(message or "")

    if OVERRIDE_RE.search(message) or WHAT_I_NEED_RE.search(message):
        return "intent_override"
    if KEY_REQUIREMENT_RE.search(message):
        return "buying"
    if STILL_EXPLORING_RE.search(message):
        return "browsing_or_boundary"

    if current == "browsing_or_boundary" and turn >= 2:
        if BOUNDARY_REPLY_RE.search(message):
            return "boundary"
        return "browsing"

    if current in {"buying", "browsing", "boundary", "intent_override"}:
        return current

    if turn == 1:
        return "intent_override"
    return current


def is_override_message(message: str) -> bool:
    """Return True when the current user message is the simulator's override event."""

    return bool(OVERRIDE_RE.search(str(message or "")) or WHAT_I_NEED_RE.search(str(message or "")))


def extract_constraints(message: str, scenario: str = "unknown", turn: int = 1) -> list[tuple[str, str]]:
    """Extract user-visible constraints from the simulator's message shapes.

    The return shape is ``[(constraint_text, source_label), ...]``.
    """

    message = str(message or "").strip()
    if not message:
        return []

    match = KEY_REQUIREMENT_RE.search(message)
    if match:
        value = match.group(1).strip(" -;,.")
        return [(value, "key_requirement")] if value else []

    match = MATTERS_RE.search(message)
    if match:
        values = [part.strip(" -;,.") for part in re.split(r";\s*", match.group(1))]
        return [(value, "customer_reply") for value in values if value]

    match = WHAT_I_NEED_RE.search(message)
    if match:
        value = match.group(1).strip(" -;,.")
        return [(value, "override_new") ] if value else []

    # Special intent-override opening: "I'm looking for {cat}. {old_value}"
    # The old value is useful to keep as a weak signal after it is demoted.
    if turn == 1 and scenario == "intent_override" and not STILL_EXPLORING_RE.search(message):
        match = INITIAL_PREFIX_RE.search(message)
        if match:
            value = match.group(1).strip(" -;,.")
            if value:
                return [(value, "override_initial")]

    return []


class QuestionPolicy:
    """Evaluator-safe clarification policy with two measured Day-3 variants.

    ``always_other``
        Ask ``other`` every turn. In the released evaluator this weakly
        dominates a specific attribute because it can reveal any two
        undisclosed constraints.

    ``other_twice_rotate``
        Follow the Day-2 fallback exactly: ask ``other`` on the first two
        agent turns, then rotate ``feature -> material -> color -> style``.
        This is the robustness/presentation fallback that Day 3 asks us to
        compare against ``always_other`` on the real evaluator.

    Both modes are guaranteed non-null and never emit ``brand`` or
    ``category``.
    """

    FALLBACK_ORDER = ("feature", "material", "color", "style")

    def __init__(self, mode: str = POLICY_OTHER_TWICE_ROTATE) -> None:
        if mode not in QUESTION_POLICY_MODES:
            raise ValueError(
                f"unknown question-policy mode {mode!r}; expected one of {QUESTION_POLICY_MODES}"
            )
        self.mode = mode

    def next_attribute(
        self,
        state: SlotState,
        candidate_attribute_scores: Mapping[str, float] | None = None,
    ) -> str:
        # The comparison requested in Day 3 is deliberately simple and
        # deterministic. ``candidate_attribute_scores`` remains accepted for
        # interface compatibility with Day 1, but does not override the two
        # policies being measured.
        del candidate_attribute_scores

        if self.mode == POLICY_ALWAYS_OTHER:
            ask_attribute = "other"
        elif len(state.asked_attributes) < 2:
            ask_attribute = "other"
        else:
            index = (len(state.asked_attributes) - 2) % len(self.FALLBACK_ORDER)
            ask_attribute = self.FALLBACK_ORDER[index]

        if ask_attribute not in SAFE_ASK_ATTRIBUTES:
            # Defensive invariant: never let future edits emit a dead or null
            # attribute into the evaluator.
            return "other"
        return ask_attribute


def attribute_constraint_table() -> list[dict]:
    """Return a copy-safe form of the Day-1 classifier map."""

    return [dict(row) for row in CLASSIFIER_TABLE]


def _format_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    headers = ("priority", "ask_attribute", "matches", "notes")
    widths = {
        h: max(len(h), *(len(str(row.get(h, ""))) for row in rows))
        for h in headers
    }
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = [
        " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        for row in rows
    ]
    return "\n".join([line, sep, *body])


def _self_check() -> list[str]:
    failures: list[str] = []

    cases = {
        "budget around $29.99": "budget",
        "budget $40, cotton": "budget",  # ordering trap
        "95% Cotton, 5% Spandex": "material",
        "color: black": "color",
        "Width: Wide": "size",
        "Department: Womens": "style",
        "great for hiking": "use_case",
        "water resistant": "feature",
    }
    for text, expected in cases.items():
        actual = classify_constraint(text)
        if actual != expected:
            failures.append(f"classify_constraint({text!r}) -> {actual!r}, expected {expected!r}")

    if "brand" in {classify_constraint(text) for text in cases}:
        failures.append("classifier unexpectedly returned brand")
    if "category" in {classify_constraint(text) for text in cases}:
        failures.append("classifier unexpectedly returned category")

    router = ScenarioRouter()
    if router.update("I'm looking for shoes. A key requirement is: leather.", 1) != "buying":
        failures.append("buying router failed")
    router.reset()
    if router.update("I'm looking for sandals, but I'm still exploring.", 1) != "browsing_or_boundary":
        failures.append("browsing/boundary provisional router failed")
    if router.update("I don't have a preference for other; please use your judgment.", 2) != "boundary":
        failures.append("boundary turn-2 router failed")
    router.reset()
    if router.update("I'm looking for shirts. Machine washable", 1) != "intent_override":
        failures.append("intent-override turn-1 router failed")

    state = SlotState("self-check")
    state.update("I'm looking for shirts. Department: Womens", 1)
    state.record_ask("other")
    state.update("For that, what matters is: cotton; color: black.", 2)
    state.update("Actually, ignore my earlier preference. What I need is: wool.", 3)
    if not state.soft_constraints or state.soft_constraints[0].text != "Department: Womens":
        failures.append("override did not demote the original preference")
    if not any(item.text == "wool" and item.active for item in state.constraints):
        failures.append("override did not preserve new active constraint")

    return failures


def _public_dialog_metrics(
    catalog_path: Path,
    public_set_path: Path,
    *,
    policy_mode: str = POLICY_OTHER_TWICE_ROTATE,
) -> dict:
    """Measure card-drain speed and scenario-router accuracy on the public set.

    This is diagnostics only. Runtime classes above remain evaluator-independent.
    Boundary is intentionally resolved on the first reply because it is
    indistinguishable from browsing in the opening message.
    """

    from collections import Counter, defaultdict

    from evaluator.local_evaluator import (
        catalog_index,
        coarse_category,
        customer_reply,
        initial_message,
        load_jsonl,
        materialize_hidden_fields,
    )

    _, categories, products = catalog_index(catalog_path)
    samples = load_jsonl(public_set_path)
    by_scenario: dict[str, list[int]] = defaultdict(list)
    routed_correct = 0
    family_correct = 0

    for sample in samples:
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        target = str(sample["ground_truth"]["parent_asin"])
        expected = str(sample["scenario_type"])

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = expected != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )

        state = SlotState(str(sample["sample_id"]))
        state.update(message, turn=1)
        policy = QuestionPolicy(policy_mode)

        # Turn-1 family routing is the strongest possible claim: browsing and
        # boundary share exactly the same opener, so both map to the same
        # provisional family until the first reply.
        expected_family = (
            "browsing_or_boundary" if expected in {"browsing", "boundary"} else expected
        )
        if state.scenario == expected_family:
            family_correct += 1

        card_values = list(
            dict.fromkeys(
                [
                    *[str(v) for v in card.get("hard_constraints", [])],
                    *[str(v) for v in card.get("soft_preferences", [])],
                ]
            )
        )

        drain_turn: int | None = 0 if set(card_values).issubset(disclosed) else None
        router_resolved = state.scenario in {
            "buying", "browsing", "boundary", "intent_override"
        }
        if router_resolved and state.scenario == expected:
            routed_correct += 1

        for turn in range(1, 11):
            if drain_turn is not None and router_resolved:
                break

            ask_attribute = policy.next_attribute(state)
            state.record_ask(ask_attribute)

            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                message, boundary_used = customer_reply(
                    effective,
                    ask_attribute,
                    disclosed,
                    boundary_used,
                )

            state.update(message, turn=turn + 1)

            if not router_resolved and state.scenario in {
                "buying", "browsing", "boundary", "intent_override"
            }:
                router_resolved = True
                if state.scenario == expected:
                    routed_correct += 1

            if drain_turn is None and set(card_values).issubset(disclosed):
                drain_turn = turn

        if drain_turn is None:
            drain_turn = 11
        by_scenario[expected].append(drain_turn)

    summary = {}
    all_values: list[int] = []
    for scenario, values in sorted(by_scenario.items()):
        all_values.extend(values)
        summary[scenario] = {
            "n": len(values),
            "distribution": dict(sorted(Counter(values).items())),
            "mean_question_turns": round(sum(values) / len(values), 3),
        }
    summary["overall"] = {
        "n": len(all_values),
        "distribution": dict(sorted(Counter(all_values).items())),
        "mean_question_turns": round(sum(all_values) / len(all_values), 3),
    }
    summary["router"] = {
        "sample_count": len(samples),
        "turn1_family_correct": family_correct,
        "turn1_family_accuracy": round(family_correct / len(samples), 6) if samples else 0.0,
        "resolved_correct": routed_correct,
        "resolved_accuracy": round(routed_correct / len(samples), 6) if samples else 0.0,
    }
    return summary


def _public_drain_metrics(catalog_path: Path, public_set_path: Path) -> dict:
    """Backward-compatible Day-1 diagnostic alias."""

    return _public_dialog_metrics(
        catalog_path, public_set_path, policy_mode=POLICY_OTHER_TWICE_ROTATE
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sheng Yan dialog diagnostics")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--policy",
        choices=QUESTION_POLICY_MODES,
        default=POLICY_OTHER_TWICE_ROTATE,
        help="policy used for the drain diagnostic",
    )
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="print drain diagnostics for both Day-3 policy variants",
    )
    parser.add_argument(
        "--skip-public-metrics",
        action="store_true",
        help="only print classifier/router self-checks",
    )
    args = parser.parse_args()

    print("Sheng Yan: attribute -> constraint map")
    print(_format_table(attribute_constraint_table()))
    print()
    print("Confirmed: brand and category are never classify_constraint() outputs.")
    print(
        "Ordering trap: budget is checked before material, so mixed "
        "budget+material text routes to budget."
    )

    failures = _self_check()
    print()
    if failures:
        print(f"Self-check: FAIL ({len(failures)})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Self-check: PASS")

    if not args.skip_public_metrics:
        catalog = Path(args.catalog)
        public_set = Path(args.public_set)
        if catalog.exists() and public_set.exists():
            modes = QUESTION_POLICY_MODES if args.compare_policies else (args.policy,)
            first_router = None
            for mode in modes:
                metrics = _public_dialog_metrics(
                    catalog, public_set, policy_mode=mode
                )
                if first_router is None:
                    first_router = metrics["router"]
                    print(
                        "\nScenario-router accuracy on released 200-session set: "
                        f"turn-1 family={first_router['turn1_family_correct']}/"
                        f"{first_router['sample_count']} "
                        f"({first_router['turn1_family_accuracy']:.1%}), "
                        f"resolved={first_router['resolved_correct']}/"
                        f"{first_router['sample_count']} "
                        f"({first_router['resolved_accuracy']:.1%})"
                    )

                print(f"\nQuestion-policy drain metrics [{mode}]:")
                for scenario in (
                    "buying", "browsing", "intent_override", "boundary", "overall"
                ):
                    row = metrics.get(scenario)
                    if row:
                        print(
                            f"  {scenario:16s} n={row['n']:3d} "
                            f"distribution={row['distribution']} "
                            f"mean={row['mean_question_turns']}"
                        )
        else:
            print(
                "\nPublic metrics skipped: expected catalog/public-set files were not found. "
                "Run with --catalog and --public-set after placing the catalog under data/."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

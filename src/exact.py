import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Common materials extracted by the simulator's intent_card()
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "silk", 
    "canvas", "denim", "spandex", "linen", "velvet", "fleece", 
    "suede", "mesh", "cashmere", "rayon", "satin", "acrylic", "fabric"
}

# Pre-compiled regex patterns with word boundaries
CONSTRAINT_PATTERNS = (
    re.compile(r"\bA key requirement is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
    re.compile(r"\bWhat I need is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
    re.compile(r"\bFor that, what matters is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
)

UNSUPPORTED_COLOR_CONSTRAINT = re.compile(r"^color\s*:", re.IGNORECASE)


def flatten_text(value: object) -> list[str]:
    """Flatten a catalog value the same way the evaluator's search corpus does."""
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value is not None else []


def clean_constraint(v: str) -> str:
    """Replicates the evaluator's _clean_constraint() byte-for-byte."""
    return re.sub(r"\s+", " ", str(v)).strip(" -;,.\t\n")[:180].rstrip()


class ExactRoute:
    def __init__(self, catalog: dict[str, dict]):
        self.catalog = catalog
        self.exact_index = defaultdict(set)
        self.material_index = defaultdict(set)
        self.sorted_prices = []  # List of (price_float, parent_asin)
        self.build_index()

    def build_index(self):
        """Indexes feature lines, detail pairs, material keywords, and catalog prices."""
        for asin, p in self.catalog.items():
            # 1. Feature bullet points
            features = p.get("features") or []
            for f in features:
                norm = clean_constraint(f)
                if norm:
                    self.exact_index[norm].add(asin)
                    lowered = norm.lower()
                    if lowered != norm:
                        self.exact_index[lowered].add(asin)

            # 2. Key: Value detail pairs
            details = p.get("details") or {}
            detail_strings = []
            for k, v in details.items():
                detail_str = f"{k}: {v}"
                detail_strings.append(detail_str)
                norm = clean_constraint(detail_str)
                if norm:
                    self.exact_index[norm].add(asin)
                    lowered = norm.lower()
                    if lowered != norm:
                        self.exact_index[lowered].add(asin)

            # 3. Bare material token indexing
            full_text = " ".join(
                part
                for field in (
                    "title", "features", "details", "description", "categories", "store"
                )
                for part in flatten_text(p.get(field))
            ).lower()
            for mat in MATERIALS:
                if re.search(rf"\b{re.escape(mat)}\b", full_text):
                    self.material_index[mat].add(asin)

            # 4. Catalog prices for budget range queries
            price = p.get("price")
            if isinstance(price, (int, float)) and price > 0:
                self.sorted_prices.append((float(price), asin))

        self.sorted_prices.sort(key=lambda x: x[0])

    def parse_budget(self, constraint: str) -> float | None:
        """Parses numerical price values from 'budget around $X' strings."""
        match = re.search(r"budget\s+around\s+\$?([\d.]+)", constraint, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def extract_constraints_from_message(self, message: str) -> list[str]:
        """Extracts individual constraint strings across all three message templates."""
        for pat in CONSTRAINT_PATTERNS:
            match = pat.search(message)
            if match:
                raw_str = match.group(1)
                # Split multi-item templates (e.g. "c1; c2")
                items = [clean_constraint(p) for p in raw_str.split(";") if clean_constraint(p)]
                return items
        cleaned = clean_constraint(message)
        return [cleaned] if cleaned else []

    def _get_single_constraint_matches(self, constraint: str) -> set[str]:
        cleaned = clean_constraint(constraint)
        lowered = cleaned.lower()

        # The current exact index has only sparse explicit colour details.
        # Treating those partial postings as authoritative caused false vetoes;
        # Day 3 explicitly defers a complete colour index.
        if UNSUPPORTED_COLOR_CONSTRAINT.match(cleaned):
            return set()

        # Handler A: Budget ranges ("budget around $29.99")
        budget = self.parse_budget(cleaned)
        if budget is not None:
            # Match items priced up to budget + 15% tolerance
            max_price = budget * 1.15
            return {asin for price, asin in self.sorted_prices if price <= max_price}

        # Handler B: Bare material words ("cotton")
        if lowered in MATERIALS:
            return self.material_index.get(lowered, set())

        # Handler C: Verbatim feature or detail string lookup ("color: black", feature line)
        matches = set(self.exact_index.get(cleaned, [])) | set(self.exact_index.get(lowered, []))
        return matches

    def exact_matches(self, constraints: list[str]) -> list[str]:
        """Intersect every indexed constraint, skipping constraints we cannot match.

        The exact route is supporting evidence, not a catalog-wide validator.  A
        constraint absent from its indexes (notably many colour constraints)
        must therefore contribute no evidence rather than vetoing evidence from
        constraints the route does understand.  If none of the constraints are
        indexed, there is no exact evidence and this returns an empty list.
        """
        if not constraints:
            return []

        constraint_sets: list[set[str]] = []
        for c in constraints:
            asins = self._get_single_constraint_matches(c)
            if asins:
                constraint_sets.append(asins)

        if not constraint_sets:
            return []

        # Catalog order makes limit truncation and benchmarks reproducible.
        intersection = set.intersection(*constraint_sets)
        return [asin for asin in self.catalog if asin in intersection]

    def query(self, text: str | list[str], limit: int = 200) -> list[tuple[str, float]]:
        """Return exact evidence for one message or accumulated constraints."""
        if limit <= 0:
            return []

        if isinstance(text, list):
            constraints = [clean_constraint(c) for c in text if clean_constraint(c)]
        else:
            constraints = self.extract_constraints_from_message(text)

        matched_asins = self.exact_matches(constraints)
        return [(asin, 1.0) for asin in matched_asins[:limit]]

if __name__ == "__main__":
    catalog_path = Path("data/catalog.jsonl")
    public_path = Path("data/public_set.jsonl")

    if not catalog_path.exists():
        raise SystemExit("data/catalog.jsonl not found. Decompress catalog.jsonl.gz first.")

    print("Loading catalog into exact index...")
    catalog = {
        str(p["parent_asin"]): p
        for p in (json.loads(l) for l in catalog_path.open(encoding="utf-8"))
    }
    route = ExactRoute(catalog)
    print(f"Indexed {len(route.exact_index)} unique normalized constraint keys.")

    if public_path.exists():
        buying_sessions = []
        with public_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("scenario_type") == "buying":
                    buying_sessions.append(row)

        # Mirror the evaluator's actual always-"other" disclosure sequence:
        # opening hard constraint, then up to two remaining constraints per reply.
        try:
            from evaluator.local_evaluator import intent_card
        except ModuleNotFoundError:
            # Direct ``python src/exact.py`` execution puts only ``src`` on
            # sys.path; add the repository root for the benchmark import.
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from evaluator.local_evaluator import intent_card

        print("\n--- Actual Disclosure Sequence Benchmark (Buying Sessions) ---")
        for turn, num_constraints in enumerate((1, 3, 4, 4), start=1):
            candidate_sizes = []
            hits = 0
            for sample in buying_sessions:
                target = str(sample["ground_truth"]["parent_asin"])
                prod = catalog.get(target, {})
                card = sample.get("intent_card") or intent_card(prod)
                disclosed = [
                    *card.get("hard_constraints", []),
                    *card.get("soft_preferences", []),
                ]
                test_constraints = disclosed[:num_constraints]
                if test_constraints:
                    matched = route.exact_matches(test_constraints)
                    candidate_sizes.append(len(matched))
                    if target in matched:
                        hits += 1

            if candidate_sizes:
                median_size = statistics.median(candidate_sizes)
                hit_pct = (hits / len(buying_sessions)) * 100
                print(
                    f"Turn: {turn} | Constraints: {num_constraints} | "
                    f"Coverage: {hits}/{len(buying_sessions)} ({hit_pct:.2f}%) | "
                    f"Median Candidate Set Size: {median_size}"
                )

import json
import re
from collections import defaultdict
from pathlib import Path

# Common materials extracted by the simulator's intent_card()
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "silk", 
    "canvas", "denim", "spandex", "linen", "velvet", "fleece", 
    "suede", "mesh", "cashmere", "rayon", "satin", "acrylic"
}

# Pre-compiled regex patterns with word boundaries
CONSTRAINT_PATTERNS = (
    re.compile(r"\bA key requirement is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
    re.compile(r"\bWhat I need is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
    re.compile(r"\bFor that, what matters is:\s*(.+?)(?:\.\s*$|$)", re.IGNORECASE),
)


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
<<<<<<< HEAD
                    self.exact_index[norm].add(asin)
                    lowered = norm.lower()
                    if lowered != norm:
                        self.exact_index[lowered].add(asin)
=======
                    self.exact_index[norm].append(asin)
                    lowered = norm.lower()
                    if lowered != norm:
                        self.exact_index[lowered].append(asin)
>>>>>>> 3b89b37a30f884e56849780ce72fa0cca468d13d

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

<<<<<<< HEAD
            # 3. Bare material token indexing
            full_text = " ".join(features + detail_strings).lower()
            for mat in MATERIALS:
                if mat in full_text:
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
=======
    def extract_constraint(
        self,
        message: str,
        _patterns=(
            re.compile(r"\bA key requirement is:\s*(.+?)(?:\.\s*$|$)", re.I),
            re.compile(r"\bWhat I need is:\s*(.+?)(?:\.\s*$|$)", re.I),
            re.compile(r"\bFor that, what matters is:\s*(.+?)(?:\.\s*$|$)", re.I),
        ),
    ) -> str:
        """Extracts customer constraints following the simulator's exact message templates."""
        for pat in _patterns:
>>>>>>> 3b89b37a30f884e56849780ce72fa0cca468d13d
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
        """
        Accepts accumulated constraints across turns.
        Returns parent ASINs that satisfy ALL disclosed constraints (intersection).
        """
        if not constraints:
            return []

        constraint_sets = []
        for c in constraints:
            asins = self._get_single_constraint_matches(c)
            if asins:
                constraint_sets.append(asins)

        if not constraint_sets:
            return []

        # Intersect matching candidate sets across all constraints
        intersection = set.intersection(*constraint_sets)
        return list(intersection)


if __name__ == "__main__":
    catalog_path = Path("data/catalog.jsonl")
    public_path = Path("data/public_set.jsonl")

    if not catalog_path.exists():
<<<<<<< HEAD
        raise SystemExit("data/catalog.jsonl not found. Decompress catalog.jsonl.gz first.")
=======
        raise SystemExit("data/catalog.jsonl not found. Please extract catalog.jsonl.gz first.")
>>>>>>> 3b89b37a30f884e56849780ce72fa0cca468d13d

    print("Loading catalog into exact index...")
    catalog = {
        str(p["parent_asin"]): p
        for p in (json.loads(l) for l in catalog_path.open(encoding="utf-8"))
    }
    route = ExactRoute(catalog)
    print(f"Indexed {len(route.exact_index)} unique constraint keys.")

    if public_path.exists():
        buying_sessions = []
        with public_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("scenario_type") == "buying":
                    buying_sessions.append(row)
<<<<<<< HEAD

=======
        
>>>>>>> 3b89b37a30f884e56849780ce72fa0cca468d13d
        candidate_sizes = []
        hits = 0

        for sample in buying_sessions:
            target = sample["ground_truth"]["parent_asin"]
            prod = catalog.get(target, {})
            features = prod.get("features", [])
            if features:
                # Test turn-1 exact lookup using first target constraint
                matched = route.exact_matches([features[0]])
                candidate_sizes.append(len(matched))
                if target in matched:
                    hits += 1

        if candidate_sizes:
            candidate_sizes.sort()
            median_size = candidate_sizes[len(candidate_sizes) // 2]
            print(f"\nBuying Sessions Benchmark ({len(buying_sessions)} sessions):")
            print(f"  Turn-1 Target Hit Rate: {hits / len(buying_sessions):.2%}")
            print(f"  Median Candidate Set Size: {median_size}")
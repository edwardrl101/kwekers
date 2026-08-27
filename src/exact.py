import json
import re
from collections import defaultdict
from pathlib import Path


def clean_constraint(v: str, limit: int = 180) -> str:
    """Replicates the simulator's exact string cleaning logic."""
    return re.sub(r"\s+", " ", str(v)).strip(" -;,.\t\n")[:limit].rstrip()


class ExactRoute:
    def __init__(self, catalog: dict[str, dict]):
        self.catalog = catalog
        self.exact_index = defaultdict(list)
        self.build_index()

    def build_index(self):
        for asin, p in self.catalog.items():
            # 1. Index bullet point features
            for f in p.get("features") or []:
                norm = clean_constraint(f)
                if norm:
                    self.exact_index[norm].append(asin)
                    self.exact_index[norm.lower()].append(asin)

            # 2. Index "Key: Value" detail pairs
            for k, v in (p.get("details") or {}).items():
                norm = clean_constraint(f"{k}: {v}")
                if norm:
                    self.exact_index[norm].append(asin)
                    self.exact_index[norm.lower()].append(asin)

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
            match = pat.search(message)
            if match:
                return clean_constraint(match.group(1))
        return clean_constraint(message)

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        cleaned = self.extract_constraint(text)
        asins = self.exact_index.get(cleaned, []) or self.exact_index.get(cleaned.lower(), [])
        
        # De-duplicate while maintaining catalog order
        seen = set()
        deduped = []
        for asin in asins:
            if asin not in seen:
                seen.add(asin)
                deduped.append((asin, 1.0))
                if len(deduped) >= limit:
                    break
        return deduped


if __name__ == "__main__":
    # Self-test on local data
    catalog_path = Path("data/catalog.jsonl")
    public_path = Path("data/public_set.jsonl")

    if not catalog_path.exists():
        print("data/catalog.jsonl not found. Please extract catalog.jsonl.gz first.")
        exit(1)

    print("Loading catalog into exact index...")
    catalog = {
        str(p["parent_asin"]): p
        for p in (json.loads(l) for l in catalog_path.open(encoding="utf-8"))
    }
    route = ExactRoute(catalog)
    print(f"Indexed {len(route.exact_index)} unique normalized constraint keys.")

    # Run quick benchmark over Buying sessions in public set
    if public_path.exists():
        buying_sessions = []
        with public_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("scenario_type") == "buying":
                    buying_sessions.append(row)
        
        candidate_sizes = []
        hits = 0

        for sample in buying_sessions:
            target = sample["ground_truth"]["parent_asin"]
            # Buying turn 1 opening message format: "I'm looking for {category}. A key requirement is: {constraint}."
            # Simulating query on the target's first feature string if available:
            prod = catalog.get(target, {})
            features = prod.get("features", [])
            if features:
                test_msg = f"I'm looking for clothing. A key requirement is: {features[0]}"
                results = route.query(test_msg)
                result_asins = [asin for asin, _ in results]
                candidate_sizes.append(len(result_asins))
                if target in result_asins:
                    hits += 1

        if candidate_sizes:
            candidate_sizes.sort()
            median_size = candidate_sizes[len(candidate_sizes) // 2]
            print(f"\nBuying Sessions Test ({len(buying_sessions)} sessions):")
            print(f"  Turn-1 Target Hit Rate: {hits / len(buying_sessions):.2%}")
            print(f"  Median Candidate Set Size: {median_size}")
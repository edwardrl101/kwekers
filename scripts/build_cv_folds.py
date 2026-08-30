"""Build and validate deterministic grouped, scenario-stratified CV folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dedup import title_similarity, variant_tokens  # noqa: E402


DEFAULT_SEED = "kwekers-cv-v1"
DEFAULT_FOLDS = 5
DEFAULT_THRESHOLD = 0.90
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_fields(samples: list[dict]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    sample_ids: list[str] = []
    scenarios: dict[str, str] = {}
    targets: dict[str, str] = {}
    for sample in samples:
        raw_sample_id = sample.get("sample_id")
        raw_scenario = sample.get("scenario_type")
        ground_truth = sample.get("ground_truth")
        raw_target = ground_truth.get("parent_asin") if isinstance(ground_truth, dict) else None
        if not all(isinstance(value, str) for value in (raw_sample_id, raw_scenario, raw_target)):
            raise ValueError("every sample requires string sample_id, scenario_type, and target ASIN")
        sample_id = raw_sample_id.strip()
        scenario = raw_scenario.strip()
        target = raw_target.strip()
        if not sample_id or not scenario or not target:
            raise ValueError("every sample requires sample_id, scenario_type, and target ASIN")
        if sample_id in scenarios:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        if scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario_type {scenario!r} for {sample_id}")
        sample_ids.append(sample_id)
        scenarios[sample_id] = scenario
        targets[sample_id] = target
    if not sample_ids:
        raise ValueError("dataset contains no samples")
    return sample_ids, scenarios, targets


def load_target_titles(catalog_path: str | Path, targets: set[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product.get("parent_asin", ""))
            if asin in targets:
                titles[asin] = str(product.get("title") or "")
    missing = targets - set(titles)
    if missing:
        raise ValueError(f"catalog is missing {len(missing)} target ASINs")
    return titles


def build_groups(
    samples: list[dict],
    target_titles: dict[str, str],
    threshold: float = DEFAULT_THRESHOLD,
    variant_sensitive: bool = True,
) -> tuple[list[dict], dict[str, str]]:
    """Group repeated targets and conservative near-duplicate target titles."""
    sample_ids, scenarios, targets = _sample_fields(samples)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("title threshold must be within [0, 1]")
    missing = set(targets.values()) - set(target_titles)
    if missing:
        raise ValueError(f"target title mapping is missing {len(missing)} ASINs")

    parent = {sample_id: sample_id for sample_id in sample_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parent[second] = first

    for index, left in enumerate(sample_ids):
        left_asin = targets[left]
        left_title = target_titles[left_asin]
        for right in sample_ids[index + 1 :]:
            right_asin = targets[right]
            if left_asin == right_asin:
                union(left, right)
                continue
            right_title = target_titles[right_asin]
            variants_match = (
                not variant_sensitive or variant_tokens(left_title) == variant_tokens(right_title)
            )
            if variants_match and title_similarity(left_title, right_title) >= threshold:
                union(left, right)

    components: dict[str, list[str]] = {}
    for sample_id in sorted(sample_ids):
        components.setdefault(find(sample_id), []).append(sample_id)

    groups: list[dict] = []
    sample_groups: dict[str, str] = {}
    for members in sorted(components.values(), key=lambda values: tuple(values)):
        digest = hashlib.sha256("\0".join(members).encode()).hexdigest()[:16]
        group_id = f"group_{digest}"
        counts = Counter(scenarios[sample_id] for sample_id in members)
        groups.append(
            {
                "group_id": group_id,
                "sample_ids": members,
                "scenario_counts": {name: counts.get(name, 0) for name in SCENARIOS},
            }
        )
        for sample_id in members:
            sample_groups[sample_id] = group_id
    return groups, sample_groups


def allocate_groups(groups: list[dict], fold_count: int, seed: str) -> list[dict]:
    """Greedily balance whole groups across folds with deterministic tie-breaking."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    totals = Counter()
    for group in groups:
        totals.update(group["scenario_counts"])
    targets = {name: totals[name] / fold_count for name in SCENARIOS}
    folds = [
        {"name": f"fold_{index}", "validation_sample_ids": [], "scenario_counts": Counter()}
        for index in range(fold_count)
    ]

    def group_order(group: dict) -> tuple:
        counts = group["scenario_counts"]
        digest = hashlib.sha256(f"{seed}\0{group['group_id']}".encode()).hexdigest()
        return (-max(counts.values()), -len(group["sample_ids"]), digest)

    for group in sorted(groups, key=group_order):
        additions = group["scenario_counts"]

        def placement_score(index: int) -> tuple:
            current = folds[index]["scenario_counts"]
            error_delta = sum(
                (
                    (current[name] + additions[name] - targets[name]) ** 2
                    - (current[name] - targets[name]) ** 2
                )
                / max(targets[name], 1.0)
                for name in SCENARIOS
            )
            resulting_size = len(folds[index]["validation_sample_ids"]) + len(group["sample_ids"])
            tie = hashlib.sha256(
                f"{seed}\0{group['group_id']}\0{folds[index]['name']}".encode()
            ).hexdigest()
            return (error_delta, resulting_size, tie)

        chosen = min(range(fold_count), key=placement_score)
        folds[chosen]["validation_sample_ids"].extend(group["sample_ids"])
        folds[chosen]["scenario_counts"].update(additions)

    rendered: list[dict] = []
    for fold in folds:
        rendered.append(
            {
                "name": fold["name"],
                "validation_sample_ids": sorted(fold["validation_sample_ids"]),
                "scenario_counts": {
                    name: fold["scenario_counts"].get(name, 0) for name in SCENARIOS
                },
            }
        )
    return rendered


def _manifest_digest(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(
    samples: list[dict],
    target_titles: dict[str, str],
    fold_count: int = DEFAULT_FOLDS,
    seed: str = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
    variant_sensitive: bool = True,
) -> dict:
    groups, sample_groups = build_groups(samples, target_titles, threshold, variant_sensitive)
    folds = allocate_groups(groups, fold_count, seed)
    multi_groups = [group for group in groups if len(group["sample_ids"]) > 1]
    manifest = {
        "version": 1,
        "seed": seed,
        "fold_count": fold_count,
        "grouping": {
            "method": "target-ASIN connected with conservative target-title families",
            "title_similarity": "token-jaccard",
            "title_threshold": threshold,
            "variant_sensitive": variant_sensitive,
        },
        "audit": {
            "sample_count": len(samples),
            "group_count": len(groups),
            "multi_sample_group_count": len(multi_groups),
            "largest_group_size": max((len(group["sample_ids"]) for group in groups), default=0),
        },
        "sample_groups": {key: sample_groups[key] for key in sorted(sample_groups)},
        "folds": folds,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    validate_manifest(samples, manifest)
    return manifest


def validate_manifest(samples: list[dict], manifest: dict) -> None:
    sample_ids, scenarios, targets = _sample_fields(samples)
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != manifest.get("fold_count"):
        raise ValueError("manifest fold count is inconsistent")
    names = [fold.get("name") for fold in folds]
    if len(names) != len(set(names)):
        raise ValueError("fold names must be unique")
    assigned: list[str] = []
    group_folds: dict[str, str] = {}
    sample_groups = manifest.get("sample_groups")
    if not isinstance(sample_groups, dict) or set(sample_groups) != set(sample_ids):
        raise ValueError("sample_groups must cover the dataset exactly")
    target_groups: dict[str, str] = {}
    for sample_id, target in targets.items():
        group_id = str(sample_groups[sample_id])
        prior = target_groups.setdefault(target, group_id)
        if prior != group_id:
            raise ValueError(f"target ASIN {target} is split across declared groups")
    for fold in folds:
        validation_ids = fold.get("validation_sample_ids")
        if not isinstance(validation_ids, list):
            raise ValueError("each fold requires validation_sample_ids")
        counts = Counter()
        for sample_id in validation_ids:
            if sample_id not in scenarios:
                raise ValueError(f"unknown validation sample ID: {sample_id}")
            assigned.append(sample_id)
            counts[scenarios[sample_id]] += 1
            group_id = str(sample_groups[sample_id])
            prior = group_folds.setdefault(group_id, str(fold["name"]))
            if prior != fold["name"]:
                raise ValueError(f"group {group_id} crosses validation folds")
        expected_counts = {name: counts.get(name, 0) for name in SCENARIOS}
        if fold.get("scenario_counts") != expected_counts:
            raise ValueError(f"scenario counts are incorrect for {fold['name']}")
    if len(assigned) != len(set(assigned)):
        raise ValueError("validation folds contain duplicate sample IDs")
    if set(assigned) != set(sample_ids):
        raise ValueError("validation folds do not cover the dataset exactly")
    expected_digest = _manifest_digest(manifest)
    if manifest.get("manifest_sha256") != expected_digest:
        raise ValueError("manifest_sha256 does not match manifest content")


def _print_summary(manifest: dict) -> None:
    audit = manifest["audit"]
    print(
        f"samples={audit['sample_count']} groups={audit['group_count']} "
        f"multi_sample_groups={audit['multi_sample_group_count']} "
        f"largest_group={audit['largest_group_size']}"
    )
    for fold in manifest["folds"]:
        print(
            f"{fold['name']}: n={len(fold['validation_sample_ids'])} "
            + " ".join(f"{name}={fold['scenario_counts'][name]}" for name in SCENARIOS)
        )
    print(f"manifest_sha256={manifest['manifest_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cv_folds.json")
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--title-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.validate_only:
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        validate_manifest(samples, manifest)
    else:
        _, _, targets = _sample_fields(samples)
        titles = load_target_titles(args.catalog, set(targets.values()))
        manifest = build_manifest(
            samples,
            titles,
            fold_count=args.fold_count,
            seed=args.seed,
            threshold=args.title_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    _print_summary(manifest)


if __name__ == "__main__":
    main()

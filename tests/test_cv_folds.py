from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.build_cv_folds import build_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def fixture_samples() -> list[dict]:
    scenarios = ("buying", "browsing", "intent_override", "boundary")
    return [
        {
            "sample_id": f"sample_{index:02d}",
            "scenario_type": scenarios[index % len(scenarios)],
            "ground_truth": {"parent_asin": f"ASIN_{index:02d}"},
        }
        for index in range(20)
    ]


class CrossValidationFoldTest(unittest.TestCase):
    def test_generation_is_deterministic_and_complete(self) -> None:
        samples = fixture_samples()
        titles = {f"ASIN_{index:02d}": f"Distinct product title {index}" for index in range(20)}
        first = build_manifest(samples, titles)
        second = build_manifest(list(reversed(samples)), titles)
        self.assertEqual(first, second)
        assigned = [sample_id for fold in first["folds"] for sample_id in fold["validation_sample_ids"]]
        self.assertCountEqual(assigned, [sample["sample_id"] for sample in samples])
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_repeated_target_group_never_crosses_folds(self) -> None:
        samples = fixture_samples()
        samples[1]["ground_truth"]["parent_asin"] = samples[0]["ground_truth"]["parent_asin"]
        titles = {f"ASIN_{index:02d}": f"Distinct product title {index}" for index in range(20)}
        manifest = build_manifest(samples, titles)
        fold_by_sample = {
            sample_id: fold["name"]
            for fold in manifest["folds"]
            for sample_id in fold["validation_sample_ids"]
        }
        self.assertEqual(fold_by_sample["sample_00"], fold_by_sample["sample_01"])

    def test_duplicate_sample_id_fails_loudly(self) -> None:
        samples = fixture_samples()
        samples[1]["sample_id"] = samples[0]["sample_id"]
        titles = {f"ASIN_{index:02d}": f"Distinct product title {index}" for index in range(20)}
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            build_manifest(samples, titles)

    def test_committed_manifest_is_valid_and_balanced(self) -> None:
        samples = [json.loads(line) for line in (ROOT / "data/public_set.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((ROOT / "data/cv_folds.json").read_text(encoding="utf-8"))
        validate_manifest(samples, manifest)
        expected = {"buying": 16, "browsing": 16, "intent_override": 6, "boundary": 2}
        self.assertTrue(all(fold["scenario_counts"] == expected for fold in manifest["folds"]))


if __name__ == "__main__":
    unittest.main()

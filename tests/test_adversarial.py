from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.adversarial import (
    _write_csv,
    constraint_changed,
    estimate_cost,
    paraphrase,
)


class AdversarialTests(unittest.TestCase):
    def test_level_zero_is_identity(self) -> None:
        message = "I'm looking for Shirts. A key requirement is: 95% Cotton, 5% Spandex."
        self.assertEqual(paraphrase(message, 0), message)

    def test_level_four_changes_constraint_wording(self) -> None:
        message = "I'm looking for Shirts. A key requirement is: 95% Cotton, 5% Spandex."
        changed = paraphrase(message, 4, seed="fixed")
        self.assertTrue(constraint_changed(message, changed))
        self.assertIn("mostly cotton with a small amount of spandex", changed.lower())

    def test_fixed_seed_is_deterministic(self) -> None:
        message = "I'm looking for Shoes. A key requirement is: leather."
        self.assertEqual(
            paraphrase(message, 3, seed="same"),
            paraphrase(message, 3, seed="same"),
        )

    def test_cost_formula_and_zero_cost(self) -> None:
        self.assertEqual(estimate_cost(1_000_000, 2_000_000, 2.0, 3.0), 8.0)
        self.assertEqual(estimate_cost(0, 0, 2.0, 3.0), 0.0)
        self.assertIsNone(estimate_cost(10, 10, None, None))

    def test_metrics_writer_keeps_required_columns_and_rows(self) -> None:
        rows = [
            {"level": 0, "sample_count": 2, "technical_score": 0.5},
            {"level": 4, "sample_count": 2, "technical_score": 0.4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            _write_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(set(loaded[0]), {"level", "sample_count", "technical_score"})
        self.assertEqual({row["level"] for row in loaded}, {"0", "4"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.adversarial import (
    FROZEN_SCORE,
    _preserved_online_rows,
    _write_csv,
    constraint_changed,
    estimate_cost,
    paraphrase,
    summarize_llm_telemetry,
)


class AdversarialTests(unittest.TestCase):
    def test_harness_uses_current_main_frozen_score(self) -> None:
        self.assertEqual(FROZEN_SCORE, 0.891084)

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

    def test_llm_telemetry_summary_uses_exposed_instrumentation(self) -> None:
        records = [
            {
                "requested_model": "vendor/model:free",
                "served_model": "vendor/model:free",
                "latency_seconds": 0.1,
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "cost": 0.0,
            },
            {
                "requested_model": "vendor/model:free",
                "served_model": "vendor/model:free",
                "latency_seconds": 0.3,
                "prompt_tokens": 60,
                "completion_tokens": 20,
                "cost": 0.0,
            },
        ]
        summary = summarize_llm_telemetry(
            records,
            call_count=2,
            sample_count=2,
            total_wall_seconds=1.0,
            input_cost_per_million=2.0,
            output_cost_per_million=4.0,
        )
        self.assertEqual(summary["llm_calls"], 2)
        self.assertEqual(summary["total_tokens"], 130)
        self.assertEqual(summary["tokens_per_session"], 65)
        self.assertAlmostEqual(summary["p50_llm_latency_ms"], 200.0)
        self.assertAlmostEqual(summary["p95_llm_latency_ms"], 290.0)
        self.assertAlmostEqual(summary["estimated_cost_per_session"], 0.00016)

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

    def test_offline_refresh_preserves_successful_online_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost_results.csv"
            path.write_text(
                "mode,available,technical_score\n"
                "llm_off,yes,0.8\n"
                "llm_on_level_4,yes,0.4\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _preserved_online_rows(path),
                [{
                    "mode": "historical_llm_on_level_4",
                    "available": "historical; not rerun",
                    "technical_score": "0.4",
                }],
            )


if __name__ == "__main__":
    unittest.main()

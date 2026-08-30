from __future__ import annotations

from importlib import metadata as importlib_metadata
import unittest
from unittest.mock import patch

from scripts.eval import ABLATION_CONFIGS
from scripts.eval_cv import (
    CONFIGS,
    _dependency_versions,
    aggregate_fold_results,
    summarize_sessions,
)
from starter.agent import BUCKET_MATCH_BOOST, DENSE_SIMILARITY_WEIGHT, EXACT_MATCH_BOOST


class CrossValidationEvaluationTest(unittest.TestCase):
    def test_cv_configurations_match_historical_ablation_runner(self) -> None:
        defaults = {
            "enable_freshness": True,
            "enable_dense": False,
            "exact_match_boost": EXACT_MATCH_BOOST,
            "bucket_match_boost": BUCKET_MATCH_BOOST,
            "dense_similarity_weight": DENSE_SIMILARITY_WEIGHT,
        }
        expected = {
            name: {**defaults, **options} for name, options in ABLATION_CONFIGS.items()
        }
        self.assertEqual(CONFIGS, expected)

    def test_aggregate_uses_all_out_of_fold_sessions(self) -> None:
        sessions = [
            {"scenario_type": "buying", "hit": True, "first_hit_turn": 1, "reciprocal_rank": 1.0},
            {"scenario_type": "boundary", "hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0},
        ]
        fold_results = [
            {"metrics": summarize_sessions([sessions[0]]), "sessions": [sessions[0]]},
            {"metrics": summarize_sessions([sessions[1]]), "sessions": [sessions[1]]},
        ]
        summary = aggregate_fold_results(fold_results)
        self.assertEqual(summary["fold_count"], 2)
        self.assertEqual(summary["out_of_fold"]["sample_count"], 2)
        self.assertEqual(summary["out_of_fold"]["hit_rate_at_10"], 0.5)
        self.assertIn("technical_score_population_sd", summary)

    def test_dependency_versions_include_reproducibility_packages(self) -> None:
        versions = {"numpy": "2.0.0", "scikit-learn": "1.6.0"}

        def fake_version(package: str) -> str:
            if package == "sentence-transformers":
                raise importlib_metadata.PackageNotFoundError
            return versions[package]

        with patch("scripts.eval_cv.importlib_metadata.version", side_effect=fake_version):
            self.assertEqual(
                _dependency_versions(),
                {
                    "numpy": "2.0.0",
                    "scikit-learn": "1.6.0",
                    "sentence-transformers": None,
                },
            )


if __name__ == "__main__":
    unittest.main()

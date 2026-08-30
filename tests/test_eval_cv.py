from __future__ import annotations

import unittest

from scripts.eval_cv import aggregate_fold_results, summarize_sessions


class CrossValidationEvaluationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import llm
from starter.agent import Agent


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCORE = 0.891111


class FrozenOfflineBaselineTest(unittest.TestCase):
    def test_offline_reproduces_exactly_without_llm_calls(self) -> None:
        catalog_path = ROOT / "data" / "catalog.jsonl"
        public_path = ROOT / "data" / "public_set.jsonl"
        if not public_path.exists():
            self.fail("frozen public set is not available")
        if not catalog_path.exists():
            self.skipTest(
                "frozen catalog is not stored in Git; run this test locally with "
                "data/catalog.jsonl present"
            )

        offline = {
            "OPENROUTER_API_KEY": "",
            "OPENROUTER_MODEL": "",
            "ENABLE_LLM_NORMALIZE": "false",
            "ENABLE_LLM_OVERRIDE": "false",
            "ENABLE_LLM_MESSAGE": "false",
            "ENABLE_CONFIDENCE": "false",
        }
        llm.reset_state()
        samples = load_jsonl(public_path)
        catalog_ids, categories, products = catalog_index(catalog_path)
        with patch.dict(os.environ, offline, clear=False), patch.object(
            llm, "PROJECT_ENV_PATH", ROOT / ".missing-offline-env"
        ):
            result = evaluate(
                Agent(catalog_path), samples, catalog_ids, categories, products
            )

        self.assertEqual(llm.CALL_COUNT, 0, "LLM called during offline scoring")
        self.assertAlmostEqual(
            result["recommended_technical_score"], EXPECTED_SCORE, places=9
        )


if __name__ == "__main__":
    unittest.main()

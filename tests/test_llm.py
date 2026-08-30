from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from src import llm


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body.read()


class OpenRouterClientTest(unittest.TestCase):
    def setUp(self) -> None:
        llm.reset_state()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.env_path = patch.object(llm, "PROJECT_ENV_PATH", Path("missing.env"))
        self.env_path.start()

    def tearDown(self) -> None:
        self.env_path.stop()
        self.environment.stop()
        llm.reset_state()

    def _configure(self, model: str = "vendor/model:free") -> None:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_MODEL"] = model

    def test_missing_credentials_are_inert(self) -> None:
        with patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(llm.call("hello"))
        urlopen.assert_not_called()
        self.assertEqual(llm.CALL_COUNT, 0)

    def test_paid_and_router_models_are_rejected_before_network(self) -> None:
        for model in ("vendor/paid-model", "openrouter/free"):
            llm.reset_state()
            self._configure(model)
            with patch("urllib.request.urlopen") as urlopen:
                self.assertIsNone(llm.call("hello"))
            urlopen.assert_not_called()
            self.assertEqual(llm.CALL_COUNT, 0)

    def test_success_is_cached_by_complete_request(self) -> None:
        self._configure()
        payload = {
            "model": "vendor/model:free",
            "choices": [{"message": {"content": "  answer  "}}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "cost": 0,
            },
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as urlopen:
            self.assertEqual(llm.call("hello", "system", 20), "answer")
            self.assertEqual(llm.call("hello", "system", 20), "answer")

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(llm.CALL_COUNT, 1)
        request = urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], "vendor/model:free")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["reasoning"], {"effort": "none", "exclude": True})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(llm.telemetry()[0]["outcome"], "success")

    def test_failures_return_none_and_are_not_cached(self) -> None:
        self._configure()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as urlopen:
            self.assertIsNone(llm.call("hello"))
            self.assertIsNone(llm.call("hello"))
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(llm.CALL_COUNT, 2)

    def test_http_status_is_recorded_without_response_body(self) -> None:
        self._configure()
        error = urllib.error.HTTPError(
            llm.OPENROUTER_URL, 429, "rate limited", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=error):
            self.assertIsNone(llm.call("hello"))
        self.assertEqual(llm.telemetry()[0]["outcome"], "http_429")

    def test_nonzero_reported_cost_is_rejected(self) -> None:
        self._configure()
        payload = {
            "model": "vendor/model:free",
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"cost": 0.01},
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            self.assertIsNone(llm.call("hello"))
        self.assertEqual(llm.telemetry()[0]["outcome"], "nonzero_cost")

    def test_unexpected_served_model_is_rejected(self) -> None:
        self._configure()
        payload = {
            "model": "vendor/different:free",
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"cost": 0},
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            self.assertIsNone(llm.call("hello"))
        self.assertEqual(llm.telemetry()[0]["outcome"], "model_mismatch")

    def test_local_dotenv_is_used_only_when_environment_is_missing(self) -> None:
        fake_env = io.StringIO(
            "OPENROUTER_API_KEY=local-key\n"
            "OPENROUTER_MODEL=vendor/local:free\n"
        )
        with patch.object(Path, "read_text", return_value=fake_env.getvalue()):
            self.assertEqual(
                llm._configured_credentials(),
                ("local-key", "vendor/local:free"),
            )

    def test_environment_flag_parser_is_conservative(self) -> None:
        os.environ["FEATURE"] = "true"
        self.assertTrue(llm.env_flag("FEATURE"))
        os.environ["FEATURE"] = "false"
        self.assertFalse(llm.env_flag("FEATURE", True))
        os.environ["FEATURE"] = "unexpected"
        self.assertTrue(llm.env_flag("FEATURE", True))


if __name__ == "__main__":
    unittest.main()

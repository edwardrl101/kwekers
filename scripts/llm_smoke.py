"""Make one non-sensitive OpenRouter call and print safe telemetry."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import llm  # noqa: E402


def main() -> int:
    llm.reset_state()
    result = llm.call(
        "Reply with exactly OK.",
        "Follow the user instruction exactly.",
        max_tokens=32,
    )
    records = llm.telemetry()
    record = records[-1] if records else {}
    summary = {
        "success": result == "OK",
        "outcome": record.get("outcome", "no_call"),
        "requested_model": record.get("requested_model"),
        "served_model": record.get("served_model"),
        "latency_seconds": round(float(record.get("latency_seconds", 0.0)), 3),
        "prompt_tokens": record.get("prompt_tokens", 0),
        "completion_tokens": record.get("completion_tokens", 0),
        "cost": record.get("cost", 0.0),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] and summary["cost"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

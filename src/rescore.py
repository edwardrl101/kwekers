"""Conditional dense rescoring, scenario-gated.

Verified in scripts/experiments.py cmd_rescore (the "conditional (vague only)"
row): tune-split R@10 0.300, vs 0.221 bm25-only and 0.243 dense-always
(buying 0.286 / browsing 0.179 / override 0.619 / boundary 0.429). Buying and
intent_override messages carry a verbatim catalog substring, so lexical BM25
already ranks them well - dense rescoring only helps when the message is
vague (browsing/boundary), and hurts otherwise (CLAUDE.md finding 4: char
n-gram rescoring hurts for the same surface-overlap-vs-relevance reason; dense
rescoring is the same trap on buying/override, just less severe).

route_scenario (src/dialog.py) at turn 1 never returns "boundary" or
"browsing" (those need turn>=2 replay against prior state) and never returns
"override" (the value is "intent_override"). "vague" here means anything NOT
in {"buying", "intent_override"}.
"""

from __future__ import annotations

import numpy as np

from src.dialog import ScenarioRouter

RAW_SCENARIOS = ("buying", "intent_override")


class ConditionalRescorer:
    """Leaves buying/override pools untouched; dense-rescores everything else,
    but only through turn_limit.

    End-to-end (scripts/run_agent.py), unconditional dense rescoring costs
    -0.075 TechnicalScore (0.7807 vs 0.8553 raw-pool) despite +0.03 R@10 in
    the isolated turn-1 test above: the other-drain question policy loads the
    query with disclosed constraints by turn 2-3, so BM25 alone is already
    accurate by then and rescoring just perturbs a good ranking. turn_limit
    caps rescoring to the early turns where the query is still vague.

    Stateful per session_id: tracks scenario detection state (ScenarioRouter).
    """

    def __init__(self, dense, turn_limit: int = 1) -> None:
        self.dense = dense
        self.turn_limit = turn_limit
        self._dense_pos = {a: i for i, a in enumerate(dense.asins)}
        self._routers: dict[str, ScenarioRouter] = {}

    def reset(self, session_id: str) -> None:
        self._routers[session_id] = ScenarioRouter()

    def rescore(self, session_id: str, pool: list[str], text: str, turn: int = 1) -> list[str]:
        if turn > self.turn_limit:
            return pool

        router = self._routers.setdefault(session_id, ScenarioRouter())
        detected = router.update(text, turn)
        if detected in RAW_SCENARIOS:
            return pool
        return self._dense_rescore(pool, text)

    def _dense_rescore(self, pool: list[str], text: str) -> list[str]:
        idx = [self._dense_pos[a] for a in pool if a in self._dense_pos]
        keep = [a for a in pool if a in self._dense_pos]
        if not idx:
            return pool
        vector = np.asarray(
            self.dense.model.encode([text], normalize_embeddings=True,
                                     convert_to_numpy=True)[0], dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.dense.embeddings[idx] @ vector
        order = np.argsort(-scores)
        rescored = [keep[i] for i in order]
        # products missing dense vectors (shouldn't happen off the full
        # catalog, but keep them rather than silently dropping candidates)
        missing = [a for a in pool if a not in self._dense_pos]
        return rescored + missing

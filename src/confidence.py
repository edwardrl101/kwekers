"""Read-only confidence layer: softmax + normalized entropy over the fused pool.

Day 4. This module is the single implementation used by
``starter.agent.Agent._confidence_from_routes`` and by offline measurement
scripts. Keeping the computation here prevents the production integration and
confidence diagnostics from drifting apart.

Contract: every function here is a pure function of its arguments. Nothing in
this module reads or writes Agent/session state, and nothing it returns is
consumed by the ranking or selection path - see
``tests/test_confidence.py::FrozenBaselineInvarianceTest`` for the empirical
proof that enabling this layer does not move the official score.
"""

from __future__ import annotations

import math
from typing import Sequence

RouteResults = dict[str, list[tuple[str, float]]]


def softmax(scores: Sequence[float]) -> list[float]:
    """Numerically stable softmax. Empty input returns ``[]``."""
    if not scores:
        return []
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    if total <= 0:
        # Every weight underflowed to 0 (e.g. all-(-inf) input). Fall back to
        # uniform rather than dividing by zero; unreachable from
        # fused_pool_scores() since the max-scoring term always contributes
        # exp(0) == 1, but this function is a general-purpose utility.
        return [1.0 / len(scores)] * len(scores)
    return [weight / total for weight in weights]


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats. Non-positive probabilities are skipped (log undefined)."""
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def normalized_confidence(scores: Sequence[float]) -> float:
    """Map a pool of fused scores to a 0-1 confidence value.

    1.0: the pool's probability mass sits on one candidate (certain).
    0.0: probability mass is spread uniformly across the whole pool, or the
    pool is empty (nothing to be confident about).
    A single-candidate pool is defined as 1.0: there is no competing
    evidence, so there is nothing to be uncertain about.
    """
    count = len(scores)
    if count == 0:
        return 0.0
    if count == 1:
        return 1.0
    probabilities = softmax(scores)
    raw_entropy = entropy(probabilities)
    max_entropy = math.log(count)
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    return max(0.0, min(1.0, 1.0 - normalized_entropy))


def _normalize_bm25_rank(rank: int, candidate_count: int) -> float:
    """Mirrors Agent._normalize_bm25_rank: map rank 1..N onto 1..0."""
    if candidate_count <= 1:
        return 1.0
    return 1.0 - ((rank - 1) / (candidate_count - 1))


def _min_max_normalize(results: list[tuple[str, float]]) -> dict[str, float]:
    """Mirrors Agent._normalize_scores: min-max one route's scores into [0, 1]."""
    if not results:
        return {}
    values = [score for _parent_asin, score in results]
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return {parent_asin: 1.0 for parent_asin, _score in results}
    scale = maximum - minimum
    return {parent_asin: (score - minimum) / scale for parent_asin, score in results}


def fused_pool_scores(
    route_results: RouteResults,
    *,
    exact_match_boost: float = 0.35,
    bucket_match_boost: float = 0.10,
    dense_similarity_weight: float = 0.0,
    pool_limit: int = 500,
) -> list[float]:
    """Reconstruct the per-candidate fused scores the ranking path assigns.

    Pure function of ``route_results`` (whatever ``Agent._route_candidates``
    returned for one turn) - no Agent/session state - so it is safe to call
    from offline measurement scripts without touching starter/agent.py.
    Mirrors ``Agent._fuse_bm25_pool`` / ``Agent._confidence_from_routes``
    exactly: same BM25-pool dedup/cap, same rank normalization, same flat
    exact/bucket boosts, same min-max dense normalization.
    """
    bm25_pool: list[str] = []
    seen: set[str] = set()
    for parent_asin, _score in route_results.get("bm25", []):
        if parent_asin and parent_asin not in seen:
            seen.add(parent_asin)
            bm25_pool.append(parent_asin)
        if len(bm25_pool) >= pool_limit:
            break
    if not bm25_pool:
        return []

    exact_ids = {parent_asin for parent_asin, _score in route_results.get("exact", [])}
    bucket_ids = {parent_asin for parent_asin, _score in route_results.get("bucket", [])}
    dense_scores = _min_max_normalize(route_results.get("dense", []))

    count = len(bm25_pool)
    scores: list[float] = []
    for rank, parent_asin in enumerate(bm25_pool, start=1):
        score = _normalize_bm25_rank(rank, count)
        if parent_asin in exact_ids:
            score += exact_match_boost
        if parent_asin in bucket_ids:
            score += bucket_match_boost
        score += dense_similarity_weight * dense_scores.get(parent_asin, 0.0)
        scores.append(score)
    return scores


def confidence_from_route_results(
    route_results: RouteResults,
    *,
    exact_match_boost: float = 0.35,
    bucket_match_boost: float = 0.10,
    dense_similarity_weight: float = 0.0,
    pool_limit: int = 500,
) -> float:
    """One-call convenience: raw route_results -> confidence in [0, 1]."""
    scores = fused_pool_scores(
        route_results,
        exact_match_boost=exact_match_boost,
        bucket_match_boost=bucket_match_boost,
        dense_similarity_weight=dense_similarity_weight,
        pool_limit=pool_limit,
    )
    return normalized_confidence(scores)

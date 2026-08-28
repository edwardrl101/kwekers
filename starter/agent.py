from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Callable, TypeAlias


RECOMMENDATION_COUNT = 10
ROUTE_CANDIDATE_LIMIT = 500
RANDOM_FILL_SEED = "kwekers-day1-random-fill-v1"

ScoredCandidate: TypeAlias = tuple[str, float]
RouteResults: TypeAlias = dict[str, list[ScoredCandidate]]


class Agent:
    """Crash-safe Day 1 router with deterministic random fallback."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        enable_dense: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.enable_dense = enable_dense
        self.dense_cache_path = self.catalog_path.with_name("dense_cache.npz")
        self._catalog = self._load_catalog()
        self._catalog_ids = list(self._catalog)
        self._catalog_id_set = set(self._catalog_ids)
        self._sessions: dict[str, dict] = {}
        self._bucket_route = None
        self._exact_route = None
        self._bm25_route = None
        self._dense_route = None
        self._route_errors: dict[str, str] = {}
        self._initialize_routes()

    def _load_catalog(self) -> dict[str, dict]:
        """Load valid, uniquely identified products without crashing Agent."""
        catalog: dict[str, dict] = {}
        try:
            with self.catalog_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        product = json.loads(line)
                        parent_asin = str(product.get("parent_asin", "")).strip()
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if parent_asin and parent_asin not in catalog:
                        catalog[parent_asin] = product
        except OSError:
            return {}
        return catalog

    def _initialize_routes(self) -> None:
        """Build each independent route, isolating optional dependency failures."""
        if not self._catalog:
            return

        try:
            from src.buckets import BucketRoute

            self._bucket_route = BucketRoute(self._catalog)
        except Exception as error:
            self._bucket_route = None
            self._route_errors["bucket"] = repr(error)

        try:
            from src.exact import ExactRoute

            self._exact_route = ExactRoute(self._catalog)
        except Exception as error:
            self._exact_route = None
            self._route_errors["exact"] = repr(error)

        try:
            from src.retrieval import BM25Route

            self._bm25_route = BM25Route(self._catalog)
        except Exception as error:
            self._bm25_route = None
            self._route_errors["bm25"] = repr(error)

        if self.enable_dense:
            try:
                from src.retrieval import DenseRoute

                self._dense_route = DenseRoute(
                    self._catalog,
                    cache=self.dense_cache_path,
                )
            except Exception as error:
                # Dense retrieval is optional at runtime: the model dependency
                # or its offline cache may not be present during judging.
                self._dense_route = None
                self._route_errors["dense"] = repr(error)

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start fresh state for a session; malformed inputs remain harmless."""
        try:
            key = str(session_id)
            safe_profile = user_profile if isinstance(user_profile, dict) else {}
            encoded_profile = json.dumps(
                safe_profile, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            self._sessions[key] = {
                "seed_key": hashlib.sha256(encoded_profile).hexdigest(),
                "user_profile": safe_profile,
            }
        except Exception:
            return None

    @staticmethod
    def _query_route(route: object, user_message: str) -> list[ScoredCandidate]:
        """Validate a route response without discarding its retrieval scores."""
        if route is None:
            return []
        results = route.query(user_message, limit=ROUTE_CANDIDATE_LIMIT)
        if not isinstance(results, list):
            return []
        candidates: list[ScoredCandidate] = []
        for result in results:
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                continue
            parent_asin = str(result[0]).strip()
            score = result[1]
            if (
                not parent_asin
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                continue
            candidates.append((parent_asin, float(score)))
        return candidates

    def _route_bucket(
        self, session: dict, user_message: str, turn: int
    ) -> list[ScoredCandidate]:
        return self._query_route(self._bucket_route, user_message)

    def _route_exact(
        self, session: dict, user_message: str, turn: int
    ) -> list[ScoredCandidate]:
        return self._query_route(self._exact_route, user_message)

    def _route_bm25(
        self, session: dict, user_message: str, turn: int
    ) -> list[ScoredCandidate]:
        return self._query_route(self._bm25_route, user_message)

    def _route_dense(
        self, session: dict, user_message: str, turn: int
    ) -> list[ScoredCandidate]:
        return self._query_route(self._dense_route, user_message)

    def _route_candidates(
        self, session: dict, user_message: str, turn: int
    ) -> RouteResults:
        """Call every route independently and retain each route's scores."""
        routes: tuple[
            tuple[str, Callable[[dict, str, int], list[ScoredCandidate]]], ...
        ] = (
            ("bucket", self._route_bucket),
            ("exact", self._route_exact),
            ("bm25", self._route_bm25),
            ("dense", self._route_dense),
        )
        route_results: RouteResults = {}
        for name, route in routes:
            try:
                candidates = route(session, user_message, turn)
            except Exception:
                candidates = []
            route_results[name] = candidates if isinstance(candidates, list) else []
        return route_results

    @staticmethod
    def _concatenate_route_ids(route_results: RouteResults) -> list[str]:
        """Keep the current route-priority ordering until fusion is implemented."""
        merged: list[str] = []
        seen: set[str] = set()
        for name in ("bucket", "exact", "bm25", "dense"):
            for parent_asin, _score in route_results.get(name, []):
                if parent_asin and parent_asin not in seen:
                    seen.add(parent_asin)
                    merged.append(parent_asin)
        return merged

    def _random_fill(self, candidates: list[str], session: dict, turn: int) -> list[str]:
        """Pad route results to ten unique recommendations reproducibly."""
        result: list[str] = []
        seen: set[str] = set()
        for parent_asin in candidates:
            if parent_asin in self._catalog_id_set and parent_asin not in seen:
                seen.add(parent_asin)
                result.append(parent_asin)
                if len(result) == RECOMMENDATION_COUNT:
                    return result

        seed_key = (
            session.get("seed_key", "missing-session")
            if isinstance(session, dict)
            else "missing-session"
        )
        seed_text = f"{RANDOM_FILL_SEED}\0{seed_key}\0{turn}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        while (
            len(result) < RECOMMENDATION_COUNT
            and len(seen) < len(self._catalog_ids)
        ):
            parent_asin = self._catalog_ids[rng.randrange(len(self._catalog_ids))]
            if parent_asin not in seen:
                seen.add(parent_asin)
                result.append(parent_asin)
        if len(result) == RECOMMENDATION_COUNT:
            return result

        # The real catalog has 50,000 IDs. Placeholders preserve the response
        # schema and exact length if that file is unavailable or too small.
        placeholder_index = 0
        while len(result) < RECOMMENDATION_COUNT:
            placeholder = f"__missing_catalog_{placeholder_index}"
            placeholder_index += 1
            if placeholder not in seen:
                seen.add(placeholder)
                result.append(placeholder)
        return result

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """Return a valid response even if session state or a route is broken."""
        try:
            key = str(session_id)
            session = self._sessions.get(
                key, {"seed_key": "missing-session", "user_profile": {}}
            )
            safe_message = user_message if isinstance(user_message, str) else ""
            safe_turn = turn if isinstance(turn, int) else 0
            route_results = self._route_candidates(session, safe_message, safe_turn)
            routed_ids = self._concatenate_route_ids(route_results)
            parent_asins = self._random_fill(routed_ids, session, safe_turn)
        except Exception:
            parent_asins = self._random_fill([], {"seed_key": "error"}, 0)

        return {
            "message": "I am refining the shortlist. What else should I consider?",
            "ask_attribute": "other",
            "recommendations": [
                {"parent_asin": parent_asin} for parent_asin in parent_asins
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

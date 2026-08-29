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
EXACT_MATCH_BOOST = 0.35
BUCKET_MATCH_BOOST = 0.10
DENSE_SIMILARITY_WEIGHT = 0.20

ScoredCandidate: TypeAlias = tuple[str, float]
RouteResults: TypeAlias = dict[str, list[ScoredCandidate]]
RouteQuery: TypeAlias = str | list[str]


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

        if self.enable_dense and self.dense_cache_path.exists():
            try:
                from src.retrieval import DenseRoute

                self._dense_route = DenseRoute(
                    self._catalog,
                    cache=self.dense_cache_path,
                    build_if_missing=False,
                )
            except Exception as error:
                # Dense retrieval is optional at runtime: the model dependency
                # or its offline cache may not be present during judging.
                self._dense_route = None
                self._route_errors["dense"] = repr(error)
        elif self.enable_dense:
            self._route_errors["dense"] = (
                f"Dense cache not found at {self.dense_cache_path}; "
                "run scripts/build_dense_cache.py first"
            )

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start fresh state for a session; malformed inputs remain harmless."""
        try:
            key = str(session_id)
            safe_profile = user_profile if isinstance(user_profile, dict) else {}
            encoded_profile = json.dumps(
                safe_profile, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            slot_state = None
            try:
                from src.dialog import SlotState

                slot_state = SlotState(session_id=key)
            except Exception:
                pass
            self._sessions[key] = {
                "seed_key": hashlib.sha256(encoded_profile).hexdigest(),
                "user_profile": safe_profile,
                "slot_state": slot_state,
                "category_message": "",
                "category": "",
                "active_constraints": [],
                "shown": set(),
            }
        except Exception:
            return None

    @staticmethod
    def _query_route(route: object, query: RouteQuery) -> list[ScoredCandidate]:
        """Validate a route response without discarding its retrieval scores."""
        if route is None:
            return []
        results = route.query(query, limit=ROUTE_CANDIDATE_LIMIT)
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
        category_message = session.get("category_message", "")
        query = category_message if isinstance(category_message, str) else ""
        return self._query_route(self._bucket_route, query or user_message)

    def _route_exact(
        self, session: dict, user_message: str, turn: int
    ) -> list[ScoredCandidate]:
        active_constraints = session.get("active_constraints")
        if isinstance(active_constraints, list) and active_constraints:
            return self._query_route(self._exact_route, active_constraints)
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
    def _normalize_scores(results: list[ScoredCandidate]) -> dict[str, float]:
        """Min-max normalize one route's finite scores into the range [0, 1]."""
        if not results:
            return {}
        values = [score for _parent_asin, score in results]
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum):
            return {parent_asin: 1.0 for parent_asin, _score in results}
        scale = maximum - minimum
        return {
            parent_asin: (score - minimum) / scale
            for parent_asin, score in results
        }

    @staticmethod
    def _normalize_bm25_rank(rank: int, candidate_count: int) -> float:
        """Map BM25 rank 1..N onto 1..0 while retaining its base ordering."""
        if candidate_count <= 1:
            return 1.0
        return 1.0 - ((rank - 1) / (candidate_count - 1))

    def _fuse_bm25_pool(self, route_results: RouteResults) -> list[str]:
        """Rerank only BM25's pool using exact, bucket, and dense evidence."""
        bm25_pool: list[ScoredCandidate] = []
        seen: set[str] = set()
        for parent_asin, score in route_results.get("bm25", []):
            if parent_asin and parent_asin not in seen:
                seen.add(parent_asin)
                bm25_pool.append((parent_asin, score))
            if len(bm25_pool) >= ROUTE_CANDIDATE_LIMIT:
                break
        if not bm25_pool:
            return []

        exact_ids = {
            parent_asin for parent_asin, _score in route_results.get("exact", [])
        }
        bucket_ids = {
            parent_asin for parent_asin, _score in route_results.get("bucket", [])
        }
        dense_scores = self._normalize_scores(route_results.get("dense", []))

        fused: list[tuple[str, float, int]] = []
        candidate_count = len(bm25_pool)
        for rank, (parent_asin, _raw_bm25_score) in enumerate(bm25_pool, start=1):
            score = self._normalize_bm25_rank(rank, candidate_count)
            if parent_asin in exact_ids:
                score += EXACT_MATCH_BOOST
            if parent_asin in bucket_ids:
                score += BUCKET_MATCH_BOOST
            score += DENSE_SIMILARITY_WEIGHT * dense_scores.get(parent_asin, 0.0)
            fused.append((parent_asin, score, rank))

        # Original BM25 rank is the deterministic tie-breaker.
        fused.sort(key=lambda item: (-item[1], item[2]))
        return [parent_asin for parent_asin, _score, _rank in fused]

    @staticmethod
    def _update_retrieval_context(
        session: dict, user_message: str, turn: int
    ) -> str:
        """Update dialog slots and produce a compact cumulative search query."""
        slot_state = session.get("slot_state")
        if slot_state is not None:
            try:
                slot_state.update(user_message, turn)
            except Exception:
                slot_state = None

        try:
            from src.buckets import extract_category_phrase

            category = extract_category_phrase(user_message)
        except Exception:
            category = None
        if category:
            session["category"] = category
            session["category_message"] = user_message

        active_constraints: list[str] = []
        if slot_state is not None:
            try:
                active_constraints = [
                    item.text for item in slot_state.constraints if item.active
                ]
            except Exception:
                active_constraints = []
        session["active_constraints"] = active_constraints

        parts: list[str] = []
        stored_category = session.get("category")
        if isinstance(stored_category, str) and stored_category.strip():
            parts.append(stored_category.strip())
        parts.extend(value.strip() for value in active_constraints if value.strip())
        return " ".join(parts) or user_message

    def _random_fill(self, candidates: list[str], session: dict, turn: int) -> list[str]:
        """Pad route results reproducibly without repeating session history."""
        result: list[str] = []
        seen: set[str] = set()
        shown = session.get("shown", set()) if isinstance(session, dict) else set()
        excluded = shown if isinstance(shown, set) else set()
        for parent_asin in candidates:
            if (
                parent_asin in self._catalog_id_set
                and parent_asin not in excluded
                and parent_asin not in seen
            ):
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
        available_count = len(self._catalog_ids) - len(excluded & self._catalog_id_set)
        while (
            len(result) < RECOMMENDATION_COUNT
            and len(seen) < available_count
        ):
            parent_asin = self._catalog_ids[rng.randrange(len(self._catalog_ids))]
            if parent_asin not in excluded and parent_asin not in seen:
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

    @staticmethod
    def _is_override_message(user_message: str) -> bool:
        """Recognize the evaluator override and close private paraphrases."""
        try:
            from src.dialog import OVERRIDE_RE

            return bool(OVERRIDE_RE.search(user_message))
        except Exception:
            return "ignore my earlier preference" in user_message.lower()

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
            retrieval_query = self._update_retrieval_context(
                session, safe_message, safe_turn
            )
            shown = session.get("shown")
            if not isinstance(shown, set):
                shown = set()
                session["shown"] = shown
            if self._is_override_message(safe_message):
                shown.clear()
            route_results = self._route_candidates(
                session, retrieval_query, safe_turn
            )
            routed_ids = self._fuse_bm25_pool(route_results)
            parent_asins = self._random_fill(routed_ids, session, safe_turn)
            shown.update(parent_asins)
            slot_state = session.get("slot_state")
            if slot_state is not None:
                try:
                    slot_state.record_ask("other")
                except Exception:
                    pass
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

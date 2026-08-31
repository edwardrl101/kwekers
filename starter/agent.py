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
DENSE_SIMILARITY_WEIGHT = 0.0
ENABLE_LLM_NORMALIZE = False
ENABLE_LLM_OVERRIDE = False
ENABLE_LLM_MESSAGE = False
ENABLE_CONFIDENCE = False

ScoredCandidate: TypeAlias = tuple[str, float]
RouteResults: TypeAlias = dict[str, list[ScoredCandidate]]
RouteQuery: TypeAlias = str | list[str]


class Agent:
    """Crash-safe Day 1 router with deterministic random fallback."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        enable_dense: bool = False,
        enable_freshness: bool = True,
        exact_match_boost: float = EXACT_MATCH_BOOST,
        bucket_match_boost: float = BUCKET_MATCH_BOOST,
        dense_similarity_weight: float = DENSE_SIMILARITY_WEIGHT,
        enable_llm_normalize: bool | None = None,
        enable_llm_override: bool | None = None,
        enable_llm_message: bool | None = None,
        enable_confidence: bool | None = None,
        question_policy_mode: str = "always_other",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.enable_dense = enable_dense
        self.enable_freshness = enable_freshness
        self.exact_match_boost = float(exact_match_boost)
        self.bucket_match_boost = float(bucket_match_boost)
        self.dense_similarity_weight = float(dense_similarity_weight)
        self.enable_llm_normalize = self._resolve_feature_flag(
            "ENABLE_LLM_NORMALIZE", enable_llm_normalize, ENABLE_LLM_NORMALIZE
        )
        self.enable_llm_override = self._resolve_feature_flag(
            "ENABLE_LLM_OVERRIDE", enable_llm_override, ENABLE_LLM_OVERRIDE
        )
        self.enable_llm_message = self._resolve_feature_flag(
            "ENABLE_LLM_MESSAGE", enable_llm_message, ENABLE_LLM_MESSAGE
        )
        self.enable_confidence = self._resolve_feature_flag(
            "ENABLE_CONFIDENCE", enable_confidence, ENABLE_CONFIDENCE
        )
        self.question_policy_mode = question_policy_mode
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

    @staticmethod
    def _resolve_feature_flag(name: str, explicit: bool | None, default: bool) -> bool:
        if explicit is not None:
            return bool(explicit)
        try:
            from src.llm import env_flag

            return env_flag(name, default)
        except Exception:
            return default

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
                "confidence": 0.0,
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
                score += self.exact_match_boost
            if parent_asin in bucket_ids:
                score += self.bucket_match_boost
            score += self.dense_similarity_weight * dense_scores.get(parent_asin, 0.0)
            fused.append((parent_asin, score, rank))

        # Original BM25 rank is the deterministic tie-breaker.
        fused.sort(key=lambda item: (-item[1], item[2]))
        return [parent_asin for parent_asin, _score, _rank in fused]

    def _confidence_from_routes(self, route_results: RouteResults) -> float:
        """Delegate read-only confidence to Member 4's integrated module."""
        try:
            from src.confidence import confidence_from_route_results

            return confidence_from_route_results(
                route_results,
                exact_match_boost=self.exact_match_boost,
                bucket_match_boost=self.bucket_match_boost,
                dense_similarity_weight=self.dense_similarity_weight,
                pool_limit=ROUTE_CANDIDATE_LIMIT,
            )
        except Exception:
            # Confidence is presentation-only. A diagnostic failure must never
            # affect ranking, recommendation selection, or response validity.
            return 0.0

    def _update_retrieval_context(
        self,
        session: dict,
        user_message: str,
        turn: int,
        *,
        override_detected: bool = False,
    ) -> str:
        """Update dialog slots and produce a compact cumulative search query."""
        slot_state = session.get("slot_state")
        if slot_state is not None:
            try:
                slot_state.update(user_message, turn)
            except Exception:
                slot_state = None

        if slot_state is not None and self.enable_llm_normalize:
            try:
                from src.dialog import OVERRIDE_RE
                from src.normalize import normalize_constraints

                values = normalize_constraints(
                    user_message,
                    self._exact_route,
                    scenario=slot_state.scenario,
                    turn=turn,
                )
                regex_override = bool(OVERRIDE_RE.search(user_message))
                slot_state.add_external_constraints(
                    values,
                    turn,
                    is_override=(
                        self.enable_llm_override
                        and override_detected
                        and not regex_override
                    ),
                )
            except Exception:
                pass
        elif (
            slot_state is not None
            and self.enable_llm_override
            and override_detected
        ):
            try:
                from src.dialog import OVERRIDE_RE

                if not OVERRIDE_RE.search(user_message):
                    slot_state.add_external_constraints([], turn, is_override=True)
            except Exception:
                pass

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

        stored_category = session.get("category")
        query_constraints = list(active_constraints)
        if self.enable_llm_override and override_detected and not active_constraints:
            query_constraints.append(user_message)
        try:
            from src.retrieval import compose_bm25_query

            return compose_bm25_query(
                stored_category if isinstance(stored_category, str) else None,
                query_constraints,
                fallback=user_message,
            )
        except Exception:
            parts = [
                stored_category.strip()
                if isinstance(stored_category, str) and stored_category.strip()
                else ""
            ]
            parts.extend(value.strip() for value in query_constraints if value.strip())
            return " ".join(part for part in parts if part) or user_message

    def _random_fill(self, candidates: list[str], session: dict, turn: int) -> list[str]:
        """Pad route results reproducibly without repeating session history."""
        result: list[str] = []
        seen: set[str] = set()
        shown = session.get("shown", set()) if isinstance(session, dict) else set()
        excluded = (
            shown if self.enable_freshness and isinstance(shown, set) else set()
        )
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
        seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big"
        )
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

        # If almost the whole catalog has already been shown, first try unseen
        # IDs (not in excluded), then fall back to old items as a last-resort
        # schema-preserving fallback. This cannot happen in the real
        # 50k/10-turn evaluation, but keeps tiny unit-test catalogs crash-safe.
        if len(result) < RECOMMENDATION_COUNT:
            excluded_set = set(excluded or ())
            for parent_asin in self._catalog_ids:
                if parent_asin not in seen and parent_asin not in excluded_set:
                    seen.add(parent_asin)
                    result.append(parent_asin)
                    if len(result) == RECOMMENDATION_COUNT:
                        return result
        if len(result) < RECOMMENDATION_COUNT:
            for parent_asin in self._catalog_ids:
                if parent_asin not in result:
                    result.append(parent_asin)
                    if len(result) == RECOMMENDATION_COUNT:
                        return result

        placeholder_index = 0
        while len(result) < RECOMMENDATION_COUNT:
            placeholder = f"__missing_catalog_{placeholder_index}"
            placeholder_index += 1
            if placeholder not in result:
                result.append(placeholder)
        return result

    def _choose_ask_attribute(self, session: dict) -> str:
        """Choose a safe non-null attribute using Sheng Yan's measured policy."""
        try:
            from src.dialog import QuestionPolicy, SAFE_ASK_ATTRIBUTES

            slot_state = session.get("slot_state")
            if slot_state is None:
                return "other"
            ask_attribute = QuestionPolicy(self.question_policy_mode).next_attribute(
                slot_state
            )
            if ask_attribute not in SAFE_ASK_ATTRIBUTES:
                return "other"
            return ask_attribute
        except Exception:
            return "other"

    @staticmethod
    def _question_message(ask_attribute: str) -> str:
        """Keep the customer-facing question aligned with ask_attribute."""
        prompts = {
            "other": "I am refining the shortlist. What else should I consider?",
            "feature": "Which product feature matters most to you?",
            "material": "Do you have a material preference?",
            "color": "Do you have a color preference?",
            "style": "Is there a style or fit you prefer?",
            "size": "Do you have a size or width requirement?",
            "budget": "What budget should I work within?",
            "use_case": "What will you mainly use it for?",
        }
        return prompts.get(
            ask_attribute, "I am refining the shortlist. What else should I consider?"
        )

    def _is_override_message(self, user_message: str) -> bool:
        """Recognize the evaluator override and close private paraphrases."""
        try:
            from src.dialog import is_override_message

            if is_override_message(user_message):
                return True
        except Exception:
            if "ignore my earlier preference" in user_message.lower():
                return True
        if not self.enable_llm_override:
            return False
        try:
            from src.normalize import detect_override

            return detect_override(user_message)
        except Exception:
            return False

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """Return a valid response even if session state or a route is broken."""
        customer_message = "I am refining the shortlist. What else should I consider?"
        ask_attribute = "other"
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        telemetry_start = 0
        llm_enabled = (
            self.enable_llm_normalize
            or self.enable_llm_override
            or self.enable_llm_message
        )
        if llm_enabled:
            try:
                from src import llm

                telemetry_start = len(llm.telemetry())
            except Exception:
                llm_enabled = False
        try:
            key = str(session_id)
            session = self._sessions.get(
                key, {
                    "seed_key": "missing-session",
                    "user_profile": {},
                    "shown": set(),
                },
            )
            safe_message = user_message if isinstance(user_message, str) else ""
            safe_turn = turn if isinstance(turn, int) else 0
            override_detected = self._is_override_message(safe_message)
            retrieval_query = self._update_retrieval_context(
                session,
                safe_message,
                safe_turn,
                override_detected=override_detected,
            )
            shown = session.get("shown")
            if not isinstance(shown, set):
                shown = set()
                session["shown"] = shown
            if self.enable_freshness and override_detected:
                shown.clear()
            route_results = self._route_candidates(
                session, retrieval_query, safe_turn
            )
            routed_ids = self._fuse_bm25_pool(route_results)
            parent_asins = self._random_fill(routed_ids, session, safe_turn)
            if self.enable_confidence or self.enable_llm_message:
                session["confidence"] = self._confidence_from_routes(route_results)
            if self.enable_freshness:
                shown.update(parent_asins)
            ask_attribute = self._choose_ask_attribute(session)
            slot_state = session.get("slot_state")
            if slot_state is not None:
                try:
                    slot_state.record_ask(ask_attribute)
                except Exception:
                    ask_attribute = "other"
            customer_message = self._question_message(ask_attribute)
            if self.enable_confidence or self.enable_llm_message:
                try:
                    from src.explain import explain

                    constraints = session.get("active_constraints", [])
                    constraints = constraints if isinstance(constraints, list) else []
                    products = [
                        self._catalog[parent_asin]
                        for parent_asin in parent_asins[:3]
                        if parent_asin in self._catalog
                    ]
                    customer_message = explain(
                        constraints,
                        products,
                        float(session.get("confidence", 0.0)),
                        use_llm=self.enable_llm_message,
                    )
                except Exception:
                    pass
        except Exception:
            parent_asins = self._random_fill([], {"seed_key": "error"}, 0)
            ask_attribute = "other"
            customer_message = self._question_message(ask_attribute)

        if llm_enabled:
            try:
                from src import llm

                records = llm.telemetry()[telemetry_start:]
                usage = {
                    "prompt_tokens": sum(
                        int(record.get("prompt_tokens", 0)) for record in records
                    ),
                    "completion_tokens": sum(
                        int(record.get("completion_tokens", 0)) for record in records
                    ),
                }
            except Exception:
                pass

        return {
            "message": customer_message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin} for parent_asin in parent_asins
            ],
            "usage": usage,
        }

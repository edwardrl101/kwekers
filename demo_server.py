"""Dependency-free local server for the Shopping Copilot technical demo.

The demo calls the production Agent unchanged. Request-scoped wrappers observe
intermediate values; they neither call a route twice nor alter returned values.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from starter.agent import Agent


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo"
CATALOG = ROOT / "data" / "catalog.jsonl"
LOCK = threading.RLock()
REPLAY_PRESETS = (
    {"sample_id": "public_0008", "label": "Buying scenario"},
    {"sample_id": "public_0012", "label": "Browsing scenario"},
    {"sample_id": "public_0003", "label": "Intent override scenario"},
    {"sample_id": "public_0041", "label": "Boundary scenario"},
)


def _constraint(item: object) -> dict:
    return {
        "text": str(getattr(item, "text", "")),
        "attribute": str(getattr(item, "attribute", "other")),
        "turn": int(getattr(item, "turn", 0)),
        "source": str(getattr(item, "source", "regex")),
        "active": bool(getattr(item, "active", False)),
        "soft_reason": getattr(item, "soft_reason", None),
    }


def _state(session: dict) -> dict:
    slots = session.get("slot_state")
    return {
        "scenario": getattr(slots, "scenario", "unknown"),
        "active": [_constraint(x) for x in getattr(slots, "constraints", []) if x.active],
        "soft": [_constraint(x) for x in getattr(slots, "soft_constraints", [])],
        "override_count": int(getattr(slots, "override_count", 0)),
        "boundary_seen": bool(getattr(slots, "boundary_seen", False)),
        "asked_attributes": list(getattr(slots, "asked_attributes", [])),
        "category": session.get("category", ""),
        "shown_count": len(session.get("shown", set())),
    }


class DemoApp:
    def __init__(self) -> None:
        self.agent = Agent(CATALOG)
        self.turns: dict[str, int] = {}
        from evaluator.local_evaluator import load_jsonl

        self.public_samples = {
            str(item["sample_id"]): item
            for item in load_jsonl(ROOT / "data" / "public_set.jsonl")
        }
        self.replays: dict[str, dict] = {}

    def reset(self, session_id: str | None = None) -> dict:
        sid = session_id or uuid.uuid4().hex[:12]
        with LOCK:
            self.agent.reset(sid, {})
            self.turns[sid] = 0
        return {"session_id": sid}

    def chat(self, payload: dict) -> dict:
        sid = str(payload.get("session_id") or uuid.uuid4().hex[:12])
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message must not be empty")
        with LOCK:
            if sid not in self.agent._sessions:
                self.agent.reset(sid, {})
                self.turns[sid] = 0
            turn = self.turns.get(sid, 0) + 1
            self.turns[sid] = turn
            session = self.agent._sessions[sid]
            before = _state(session)
            shown_before = set(session.get("shown", set()))
            observed: dict = {}

            original_update = self.agent._update_retrieval_context
            original_routes = self.agent._route_candidates
            original_fuse = self.agent._fuse_bm25_pool

            def watch_update(*args, **kwargs):
                value = original_update(*args, **kwargs)
                observed["query"] = value
                observed["override"] = bool(kwargs.get("override_detected"))
                return value

            def watch_routes(*args, **kwargs):
                value = original_routes(*args, **kwargs)
                observed["routes"] = value
                return value

            def watch_fuse(*args, **kwargs):
                value = original_fuse(*args, **kwargs)
                observed["fused"] = value
                return value

            self.agent._update_retrieval_context = watch_update
            self.agent._route_candidates = watch_routes
            self.agent._fuse_bm25_pool = watch_fuse
            try:
                response = self.agent.respond(sid, message, turn, 10)
            finally:
                self.agent._update_retrieval_context = original_update
                self.agent._route_candidates = original_routes
                self.agent._fuse_bm25_pool = original_fuse

            after = _state(session)
            routes = observed.get("routes", {})
            bm25 = routes.get("bm25", [])
            exact_ids = {x[0] for x in routes.get("exact", [])}
            bucket_ids = {x[0] for x in routes.get("bucket", [])}
            bm25_rank = {x[0]: i + 1 for i, x in enumerate(bm25)}
            fused_rank = {x: i + 1 for i, x in enumerate(observed.get("fused", []))}
            recommendations = []
            for final_rank, rec in enumerate(response["recommendations"], 1):
                asin = rec["parent_asin"]
                product = self.agent._catalog.get(asin, {})
                image = product.get("image")
                if not image:
                    images = product.get("images")
                    image = images[0] if isinstance(images, list) and images else None
                recommendations.append({
                    "parent_asin": asin,
                    "title": product.get("title"),
                    "categories": product.get("categories"),
                    "price": product.get("price"),
                    "rating": product.get("average_rating"),
                    "rating_count": product.get("rating_number"),
                    "store": product.get("store"),
                    "features": (product.get("features") or [])[:4],
                    "image": image if isinstance(image, str) and image.startswith("http") else None,
                    "final_rank": final_rank,
                    "bm25_rank": bm25_rank.get(asin),
                    "fused_rank": fused_rank.get(asin),
                    "exact_evidence": asin in exact_ids,
                    "bucket_evidence": asin in bucket_ids,
                    "previously_shown": asin in shown_before,
                    "promoted": bool(bm25_rank.get(asin) and fused_rank.get(asin) and fused_rank[asin] < bm25_rank[asin]),
                })

            active = [x["text"] for x in after["active"]]
            supported, unsupported = [], []
            exact_route = self.agent._exact_route
            for value in active:
                try:
                    (supported if exact_route._get_single_constraint_matches(value) is not None else unsupported).append(value)
                except Exception:
                    unsupported.append(value)
            response["session_id"] = sid
            response["recommendations"] = recommendations
            ask_attribute = response.get("ask_attribute")
            response_message = str(response.get("message") or "").strip()
            fallback_questions = {
                "material": "Do you have a preferred material?",
                "color": "Do you have a preferred color?",
                "budget": "What budget are you working with?",
                "size": "Do you have a preferred size?",
                "style": "Is there a particular style you prefer?",
                "feature": "Any specific feature you're looking for?",
                "use_case": "What will you mainly use it for?",
                "brand": "Do you have a preferred brand?",
                "category": "What type of product are you looking for?",
                "other": "Anything else that matters to you?",
            }
            # Presentation-only fallback. The Agent's own question always wins.
            response["next_question"] = (
                response_message
                if "?" in response_message
                else fallback_questions.get(ask_attribute, "")
            )
            response["debug"] = {
                "turn": turn,
                "raw_message": message,
                "parser": "regex-first",
                "llm_fallback": "used" if response.get("usage", {}).get("prompt_tokens") else "not used",
                "override_detected": observed.get("override", False),
                "state_before": before,
                "state_after": after,
                "retrieval_query": observed.get("query", message),
                "catalog_size": len(self.agent._catalog),
                "candidate_limit": 500,
                "route_counts": {name: len(items) for name, items in routes.items()},
                "route_samples": {name: [{"asin": a, "score": round(s, 4)} for a, s in items[:3]] for name, items in routes.items() if items},
                "exact_supported": supported,
                "exact_unsupported": unsupported,
                "bm25_exact_overlap": sum(1 for a, _ in bm25 if a in exact_ids),
                "fusion": {"base": "BM25 rank", "exact_boost": self.agent.exact_match_boost, "bucket_boost": self.agent.bucket_match_boost, "dense_weight": self.agent.dense_similarity_weight, "dense_enabled": bool(self.agent._dense_route)},
                "shown_before": len(shown_before),
                "shown_after": after["shown_count"],
                "freshness_filtered": sum(1 for a in observed.get("fused", []) if a in shown_before),
                "override_reset": bool(observed.get("override") and shown_before),
                "route_errors": self.agent._route_errors,
                "ask_attribute": ask_attribute,
                "next_question": response["next_question"],
            }
            return response

    def replay_start(self, sample_id: str) -> dict:
        """Start one evaluator-faithful labeled replay without leaking target to Agent."""
        from evaluator.local_evaluator import (
            coarse_category,
            initial_message,
            materialize_hidden_fields,
        )

        allowed = {item["sample_id"] for item in REPLAY_PRESETS}
        if sample_id not in allowed or sample_id not in self.public_samples:
            raise ValueError("unknown replay preset")
        with LOCK:
            sample = self.public_samples[sample_id]
            target = str(sample["ground_truth"]["parent_asin"])
            product = self.agent._catalog.get(target)
            if not product:
                raise ValueError("replay target is unavailable in the catalog")
            card, behavior = materialize_hidden_fields(sample, self.agent._catalog)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            message = initial_message(
                effective,
                coarse_category([str(x) for x in product.get("categories") or []]),
                disclosed,
            )
            sid = f"replay_{sample_id}_{uuid.uuid4().hex[:8]}"
            self.agent.reset(sid, sample.get("user_profile", {}))
            self.turns[sid] = 0
            self.replays[sid] = {
                "sample": effective,
                "target": target,
                "disclosed": disclosed,
                "boundary_used": False,
                "override_applied": sample["scenario_type"] != "intent_override",
                "message": message,
                "done": False,
            }
            return {
                "session_id": sid,
                "sample_id": sample_id,
                "scenario": sample["scenario_type"],
                "next_customer_message": message,
                "target": {
                    "parent_asin": target,
                    "title": product.get("title"),
                    "price": product.get("price"),
                    "rating": product.get("average_rating"),
                },
                "note": "Target is visible to the audience only and is never passed to Agent.",
            }

    def replay_step(self, session_id: str) -> dict:
        """Advance exactly one turn using the evaluator's message-generation rules."""
        from evaluator.local_evaluator import MAX_TURNS, customer_reply

        with LOCK:
            replay = self.replays.get(str(session_id))
            if replay is None:
                raise ValueError("unknown replay session")
            if replay["done"]:
                raise ValueError("replay session is already complete")
            message = replay["message"]
            response = self.chat({"session_id": session_id, "message": message})
            turn = int(response["debug"]["turn"])
            ranked = [str(item["parent_asin"]) for item in response["recommendations"]]
            target = replay["target"]
            rank = (
                ranked.index(target) + 1
                if replay["override_applied"] and target in ranked
                else None
            )
            hit = rank is not None
            done = hit or turn >= MAX_TURNS
            next_message = None
            if not done:
                sample = replay["sample"]
                override = sample.get("behavior", {}).get("override") or {}
                if (
                    not replay["override_applied"]
                    and turn + 1 == int(override.get("turn", 3))
                ):
                    replay["override_applied"] = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        replay["disclosed"].add(new_value)
                    next_message = str(
                        override.get(
                            "message",
                            "Actually, please ignore my earlier preference.",
                        )
                    )
                else:
                    next_message, replay["boundary_used"] = customer_reply(
                        sample,
                        response.get("ask_attribute"),
                        replay["disclosed"],
                        replay["boundary_used"],
                    )
                replay["message"] = next_message
            replay["done"] = done
            return {
                "session_id": session_id,
                "turn": turn,
                "customer_message": message,
                "agent_response": response,
                "hit": hit,
                "target_rank": rank,
                "done": done,
                "outcome": "target_found" if hit else ("turn_limit" if done else "continue"),
                "next_customer_message": next_message,
            }


APP: DemoApp | None = None


class Handler(SimpleHTTPRequestHandler):
    def _json(self, status: int, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 1_000_000:
                raise ValueError("request body too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/chat":
                self._json(200, APP.chat(payload))
            elif self.path == "/api/reset":
                self._json(200, APP.reset(payload.get("session_id")))
            elif self.path == "/api/replay/start":
                self._json(200, APP.replay_start(str(payload.get("sample_id", ""))))
            elif self.path == "/api/replay/step":
                self._json(200, APP.replay_step(str(payload.get("session_id", ""))))
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            print(f"[demo] unhandled error: {error!r}")
            self._json(500, {"error": "demo request failed"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            self._json(200, {"catalog_size": len(APP.agent._catalog), "routes": {"bm25": bool(APP.agent._bm25_route), "exact": bool(APP.agent._exact_route), "bucket": bool(APP.agent._bucket_route), "dense": bool(APP.agent._dense_route)}, "replay_presets": REPLAY_PRESETS})
            return
        target = DEMO / ("index.html" if path == "/" else path.lstrip("/"))
        try:
            target = target.resolve()
            if DEMO.resolve() not in target.parents and target != DEMO.resolve():
                raise OSError
            content = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        print(f"[demo] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    global APP
    print("Loading the 50,000-product catalog and offline indexes…")
    APP = DemoApp()
    print(f"Shopping Copilot demo: http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

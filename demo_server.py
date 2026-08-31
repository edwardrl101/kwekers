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
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/chat":
                self._json(200, APP.chat(payload))
            elif self.path == "/api/reset":
                self._json(200, APP.reset(payload.get("session_id")))
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            self._json(500, {"error": f"demo request failed: {error}"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            self._json(200, {"catalog_size": len(APP.agent._catalog), "routes": {"bm25": bool(APP.agent._bm25_route), "exact": bool(APP.agent._exact_route), "bucket": bool(APP.agent._bucket_route), "dense": bool(APP.agent._dense_route)}})
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

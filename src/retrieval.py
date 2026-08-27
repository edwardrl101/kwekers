"""Member 4 - retrieval routes.

Three independent search engines over the 50k catalog. Each exposes the same
interface so Member 1 can fuse them with RRF on Day 2:

    route.query(text, limit) -> list[(parent_asin, score)]  # best first

Routes:
    BM25Route   - lexical / keyword, SQLite FTS5, no dependencies
    NgramRoute  - char 3-5 gram TF-IDF, fuzzy + truncation robust
    DenseRoute  - bge-small bi-encoder embeddings, semantic

IMPORTANT for the team: norm() below is aggressive (NFKC + symbol stripping).
It is for THESE ROUTES ONLY. Member 3's exact-match index must mirror the
evaluator's _clean_constraint() byte-for-byte instead, which does NOT do NFKC.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- normalization

_KEEP = re.compile(r"[^\w\s$%.,'-]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """NFKC-fold, strip decorative symbols, collapse whitespace, lowercase.

    NFKC maps fullwidth/decorative variants onto plain ASCII (A->A, (1)->1),
    which matters because ~20% of catalog rows carry non-ASCII in
    features/description. Must be applied identically at index and query time.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def flatten(value: object) -> str:
    """Collapse a catalog field (str | list | dict | None) into one string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


FIELDS = ("title", "categories", "features", "details", "store", "description")


def product_fields(product: dict) -> dict[str, str]:
    return {field: norm(flatten(product.get(field))) for field in FIELDS}


def product_text(product: dict, cap: int = 900) -> str:
    """One blob per product. Title and categories repeated for extra weight."""
    parts = product_fields(product)
    blob = " ".join(
        [parts["title"], parts["title"], parts["categories"], parts["categories"],
         parts["features"], parts["details"], parts["store"], parts["description"]]
    )
    return _WS.sub(" ", blob).strip()[:cap]


def load_catalog(path: str | Path = "data/catalog.jsonl") -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return {
            str(row["parent_asin"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


# ------------------------------------------------------------ query preparation

# Boilerplate that appears in EVERY simulated customer message. Zero signal,
# pure noise for lexical routes. Ordered longest-first so the big ones go first.
BOILERPLATE = [
    "those options are not quite right yet ask me about one specific attribute",
    "actually ignore my earlier preference what i need is",
    "i don't have an additional preference for",
    "i dont have an additional preference for",
    "i don't have a preference for",
    "i dont have a preference for",
    "please use your judgment",
    "but i'm still exploring",
    "but im still exploring",
    "a key requirement is",
    "for that what matters is",
    "i'm looking for",
    "im looking for",
]

# Markers after which the interesting content lives.
_CONSTRAINT_MARKERS = [
    r"a key requirement is:\s*(.+)",
    r"for that,?\s*what matters is:\s*(.+)",
    r"what i need is:\s*(.+)",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "still", "exploring", "key", "requirement", "matters", "need",
    "have", "preference", "dont", "about", "ask", "one", "specific",
    "attribute", "options", "not", "quite", "right", "yet", "actually",
    "ignore", "earlier", "judgment", "use", "your",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def extract_constraints(message: str) -> list[str]:
    """Pull the content-bearing spans out of a simulated customer message.

    Returns normalized phrases. Falls back to the whole message if no marker
    is present (e.g. a browsing opener, which is category-only).
    """
    lowered = norm(message)
    found: list[str] = []
    for pattern in _CONSTRAINT_MARKERS:
        match = re.search(pattern, lowered)
        if match:
            # 'For that, what matters is: X; Y.' -> two constraints
            for chunk in match.group(1).split(";"):
                chunk = chunk.strip(" .,;")
                if chunk:
                    found.append(chunk)
    return found or [strip_boilerplate(message)]


def strip_boilerplate(message: str) -> str:
    text = norm(message)
    for phrase in BOILERPLATE:
        text = text.replace(phrase, " ")
    return _WS.sub(" ", text).strip(" .,;:")


def content_terms(text: str, limit: int = 24) -> list[str]:
    terms = [
        token for token in _TOKEN.findall(norm(text))
        if len(token) > 1 and token not in STOPWORDS
    ]
    return list(dict.fromkeys(terms))[:limit]


# --------------------------------------------------------------------- BM25

def _fts_phrase(text: str) -> str:
    """Quote a phrase for FTS5 (alnum tokens only, doubled quotes are moot)."""
    tokens = _TOKEN.findall(norm(text))
    return '"' + " ".join(tokens) + '"' if tokens else ""


class BM25Route:
    """Field-weighted BM25 over SQLite FTS5, standard library only.

    Fixes the starter's core defect: the starter ORs every query token, so
    'black cotton running shirt' matches anything containing 'black' and the
    signal drowns. This runs a strictness cascade instead - exact phrase, then
    AND, then OR - and concatenates the tiers so stricter matches outrank
    looser ones.

    Field weights favour title/categories because ~20% of rows carry unicode
    junk and seller spam in features/description.
    """

    # (parent_asin, title, categories, features, details, store, description)
    WEIGHTS = (0.0, 10.0, 12.0, 3.0, 3.0, 1.5, 1.0)

    def __init__(self, catalog: dict[str, dict]) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, "
            "store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        for asin, product in catalog.items():
            parts = product_fields(product)
            batch.append((asin, parts["title"], parts["categories"],
                          parts["features"], parts["details"],
                          parts["store"], parts["description"]))
            if len(batch) >= 2000:
                cursor.executemany(
                    "INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    def _run(self, expression: str, limit: int) -> list[tuple[str, float]]:
        if not expression:
            return []
        weights = ", ".join(str(w) for w in self.WEIGHTS)
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin, bm25(products, {weights}) AS s "
                "FROM products WHERE products MATCH ? ORDER BY s LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # SQLite bm25() is negative, more negative = better. Flip the sign.
        return [(str(asin), -float(score)) for asin, score in rows]

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        constraints = extract_constraints(text)
        terms = content_terms(" ".join(constraints))
        tiers: list[list[tuple[str, float]]] = []

        # Tier 1: exact phrase for each extracted constraint.
        phrases = [p for p in (_fts_phrase(c) for c in constraints) if p]
        if phrases:
            tiers.append(self._run(" OR ".join(phrases), limit))
        # Tier 2: every content term must appear somewhere.
        if terms:
            tiers.append(self._run(" AND ".join(f'"{t}"' for t in terms), limit))
        # Tier 3: any term (the starter's behaviour, kept only as a backstop).
        if terms:
            tiers.append(self._run(" OR ".join(f'"{t}"' for t in terms), limit))

        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for tier_index, tier in enumerate(tiers):
            offset = 1000.0 * (len(tiers) - tier_index)  # keep tiers ordered
            for asin, score in tier:
                if asin in seen:
                    continue
                seen.add(asin)
                results.append((asin, offset + score))
                if len(results) >= limit:
                    return results
        return results


# ------------------------------------------------------------------- n-grams

class NgramRoute:
    """Character 3-5 gram TF-IDF. Fuzzy matcher, not a semantic one.

    Why characters: the evaluator's _clean_constraint() truncates constraints
    at 180 chars, often mid-word, and the catalog is full of near-variants
    (grey/gray, Button-Down/button down, unicode junk). Character n-grams
    survive all of that; word tokens do not. No model, no download, no network,
    which makes this the offline-safe route.
    """

    def __init__(self, catalog: dict[str, dict], max_features: int = 300_000) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.asins = np.array(list(catalog.keys()))
        documents = [product_text(catalog[a]) for a in catalog]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=max_features, min_df=3, dtype=np.float32,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(documents)  # L2-normalized

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        cleaned = " ".join(extract_constraints(text))
        if not cleaned:
            return []
        vector = self.vectorizer.transform([cleaned])
        scores = (self.matrix @ vector.T).toarray().ravel()
        return _top_k(self.asins, scores, limit)


# --------------------------------------------------------------------- dense

class DenseRoute:
    """bge-small bi-encoder embeddings, cosine via one matmul. Fully in-memory.

    50k x 384 float16 is ~38MB, so this satisfies the 'no external vector DB'
    constraint. Vectors are cached to disk after the first build - encoding 50k
    rows takes 5-15 min on CPU and you never want to redo it.
    """

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        catalog: dict[str, dict],
        cache: str | Path = "data/dense_cache.npz",
        model_name: str | None = None,
        device: str | None = "cpu",
    ) -> None:
        """device defaults to 'cpu'.

        Apple MPS can emit NaN on transformer forward passes, which silently
        corrupts rows and poisons the top-k ordering. CPU is a couple of
        minutes slower for a one-off 50k encode and is deterministic. Pass
        device=None to let sentence-transformers choose.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or self.MODEL, device=device)
        cache_path = Path(cache)
        asins = list(catalog.keys())

        if cache_path.exists():
            blob = np.load(cache_path, allow_pickle=False)
            if [str(a) for a in blob["asins"]] == asins:
                self.asins = np.array(asins)
                self.embeddings = self._sanitize(blob["embeddings"])
                return

        documents = [product_text(catalog[a], cap=512) for a in asins]
        raw = self.model.encode(
            documents, batch_size=128, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        self.asins = np.array(asins)
        self.embeddings = self._sanitize(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # float16 on disk (~38MB), float32 in RAM for fast, safe matmuls.
        np.savez(cache_path, asins=self.asins,
                 embeddings=self.embeddings.astype(np.float16))

    @staticmethod
    def _sanitize(matrix: np.ndarray) -> np.ndarray:
        """Zero out non-finite rows and re-normalize. Loud about damage."""
        result = np.asarray(matrix, dtype=np.float32)
        bad = ~np.isfinite(result).all(axis=1)
        if bad.any():
            print(f"[DenseRoute] WARNING: {int(bad.sum())} of {len(result)} "
                  f"embeddings were NaN/Inf and have been zeroed. "
                  f"Delete the cache and re-encode with device='cpu'.")
            result[bad] = 0.0
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-8)

    def query(self, text: str, limit: int = 200) -> list[tuple[str, float]]:
        cleaned = strip_boilerplate(text) or text
        vector = np.asarray(
            self.model.encode([cleaned], normalize_embeddings=True,
                              convert_to_numpy=True)[0], dtype=np.float32)
        if not np.isfinite(vector).all():
            return []
        return _top_k(self.asins, self.embeddings @ vector, limit)


# -------------------------------------------------------------------- helpers

def _top_k(asins: np.ndarray, scores: np.ndarray, limit: int) -> list[tuple[str, float]]:
    limit = min(limit, len(scores))
    if limit <= 0:
        return []
    idx = np.argpartition(-scores, limit - 1)[:limit]
    idx = idx[np.argsort(-scores[idx])]
    return [(str(asins[i]), float(scores[i])) for i in idx if scores[i] > 0]


if __name__ == "__main__":
    import time

    catalog = load_catalog()
    print(f"catalog: {len(catalog)} products")
    for name, cls in (("bm25", BM25Route), ("ngram", NgramRoute)):
        start = time.perf_counter()
        route = cls(catalog)
        build = time.perf_counter() - start
        probe = "I'm looking for Sandals. A key requirement is: 95% Cotton, 5% Spandex."
        start = time.perf_counter()
        hits = route.query(probe, 10)
        latency = (time.perf_counter() - start) * 1000
        print(f"{name:6s} build {build:6.1f}s  query {latency:6.1f}ms  top={hits[:2]}")
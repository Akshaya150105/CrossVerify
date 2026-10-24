"""One entry point that wires CLIP, BM25, expansion, RRF, re-ranking,
Rocchio feedback, pHash reuse flags and consistency badges together.

Used by the evaluation scripts and the FastAPI service so they can never drift.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import norm_path
from consistency.reuse import matches_for
from retrieval.bm25 import BM25Retriever
from retrieval.feedback import rocchio
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.query_expansion import expand_query

DEFAULT_DB = Path("data/processed/crossverify.db")
DEFAULT_INDEX = Path("data/processed/clip_index")
MODES = ("clip", "bm25", "rrf")


class SearchPipeline:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        index_dir: Path = DEFAULT_INDEX,
        clip=None,
        bm25=None,
        reranker=None,
        checker=None,
    ):
        self.db_path = Path(db_path)
        self.index_dir = Path(index_dir)
        self._clip, self._bm25, self._reranker, self._checker = clip, bm25, reranker, checker
        self._lock = threading.Lock()

    # ---- lazily-loaded components (CLIP/cross-encoder are slow to load) ----
    # Guarded by a lock: the API serves concurrent requests from a threadpool,
    # and an unguarded check-then-set here lets one thread hand out a
    # half-built BM25Retriever/model to another before it finishes loading.
    @property
    def clip(self):
        if self._clip is None:
            with self._lock:
                if self._clip is None:
                    from retrieval.hybrid_search import ClipImageRetriever

                    self._clip = ClipImageRetriever(self.index_dir)
        return self._clip

    @property
    def bm25(self):
        if self._bm25 is None:
            with self._lock:
                if self._bm25 is None:
                    bm25 = BM25Retriever(self.db_path)
                    bm25.load_documents()
                    bm25.build_index()
                    self._bm25 = bm25
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    from reranking.cross_encoder import CrossEncoderReranker

                    self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def checker(self):
        if self._checker is None:
            with self._lock:
                if self._checker is None:
                    from consistency.checker import ConsistencyChecker, load_threshold

                    self._checker = ConsistencyChecker(self.clip, load_threshold())
        return self._checker

    # ---- data access ----
    def fetch_records(self, article_ids: list[int]) -> dict[int, dict]:
        if not article_ids:
            return {}
        marks = ",".join("?" * len(article_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT a.id, a.headline, a.body, a.date, a.source, a.url, a.fetched_at,
                       i.filepath, i.caption, i.caption_source, i.image_url, i.phash
                FROM articles a LEFT JOIN images i ON i.article_id = a.id
                WHERE a.id IN ({marks})
                """,
                [int(a) for a in article_ids],
            ).fetchall()
        keys = ["article_id", "headline", "body", "date", "source", "url", "fetched_at",
                "filepath", "caption", "caption_source", "image_url", "phash"]
        records: dict[int, dict] = {}
        for row in rows:
            rec = dict(zip(keys, row))
            rec["filepath"] = norm_path(rec["filepath"]) if rec["filepath"] else None
            records.setdefault(rec["article_id"], rec)  # first image per article
        return records

    # ---- feedback ----
    def feedback_vector(self, query: str, feedback: dict):
        rel = [v for a in feedback.get("relevant", []) if (v := self.clip.vector_for_article(a)) is not None]
        irr = [v for a in feedback.get("irrelevant", []) if (v := self.clip.vector_for_article(a)) is not None]
        return rocchio(self.clip.encode_text(query), rel, irr)

    # ---- enrichment: provenance, reuse, consistency ----
    def _enrich(self, docs: list[dict], explain: dict, with_consistency: bool, max_distance: int = 6):
        with sqlite3.connect(self.db_path) as conn:
            for doc in docs:
                aid = doc["article_id"]
                doc["reuse_matches"] = matches_for(conn, doc.get("phash"), aid, max_distance)
                doc["explanation"] = {
                    name: {"rank": ranks.get(aid, (None, None))[0], "score": ranks.get(aid, (None, None))[1]}
                    for name, ranks in explain.items()
                }
        if with_consistency:
            for doc in docs:
                vec = self.clip.vector_for_article(doc["article_id"])
                if vec is not None and doc.get("caption"):
                    doc["consistency"] = self.checker.check_vector(vec, doc["caption"])
        return docs

    @staticmethod
    def _rank_map(results: list[dict]) -> dict[int, tuple[int, float]]:
        return {int(r["article_id"]): (rank, float(r["score"])) for rank, r in enumerate(results, 1)}

    # ---- public API ----
    def search(
        self,
        query: str,
        mode: str = "rrf",
        top_k: int = 10,
        candidate_k: int = 50,
        expand: bool = False,
        rerank: bool = True,
        feedback: dict | None = None,
        use_keybert: bool = False,
        with_consistency: bool = True,
    ) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        variants = expand_query(query, use_keybert=use_keybert) if expand else [query]
        clip_lists, bm25_lists = [], []

        if mode in ("clip", "rrf"):
            for n, variant in enumerate(variants):
                if n == 0 and feedback:
                    clip_lists.append(self.clip.search_vector(self.feedback_vector(query, feedback), candidate_k))
                else:
                    clip_lists.append(self.clip.search(variant, candidate_k))
        if mode in ("bm25", "rrf"):
            bm25_lists = [self.bm25.search(v, candidate_k) for v in variants]

        all_lists = clip_lists + bm25_lists
        if len(all_lists) == 1:
            ranked = [(int(r["article_id"]), float(r["score"])) for r in all_lists[0]]
        else:
            ranked = reciprocal_rank_fusion(all_lists)

        ranked = ranked[: candidate_k if rerank else top_k]
        records = self.fetch_records([a for a, _ in ranked])
        docs = []
        for aid, score in ranked:
            if aid in records:
                docs.append({**records[aid], "score": score, "variants": variants})

        if rerank and docs:
            docs = self.reranker.rerank(query, docs)
        docs = docs[:top_k]

        explain = {}
        if clip_lists:
            explain["clip"] = self._rank_map(clip_lists[0])
        if bm25_lists:
            explain["bm25"] = self._rank_map(bm25_lists[0])
        return self._enrich(docs, explain, with_consistency and bool(clip_lists))

    def search_by_image(self, image, top_k: int = 10, with_consistency: bool = True) -> list[dict]:
        """Image query. The index stores image vectors, so this is image-to-image retrieval."""

        results = self.clip.search_vector(self.clip.encode_image(image), top_k)
        records = self.fetch_records([int(r["article_id"]) for r in results])
        docs = [
            {**records[int(r["article_id"])], "score": float(r["score"])}
            for r in results
            if int(r["article_id"]) in records
        ]
        return self._enrich(docs, {"clip": self._rank_map(results)}, with_consistency)
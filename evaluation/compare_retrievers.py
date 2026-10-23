"""Compare ranking variants with Recall@K, MAP and nDCG@10 (guide sections 7 and 10).

Variants: BM25, CLIP, RRF, RRF+expansion, RRF+rerank, RRF+expansion+rerank,
plus one simulated round of Rocchio feedback (nDCG before/after).

Queries come from evaluation/queries.json (hand-written, relevant article IDs)
and optionally N automatic caption->article queries (--caption-queries N).
Re-check IDs in queries.json whenever the corpus is rebuilt.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import average_precision, ndcg_at_k, recall_at_k
from retrieval.pipeline import SearchPipeline

VARIANTS = {
    "BM25": dict(mode="bm25"),
    "CLIP": dict(mode="clip"),
    "RRF": dict(mode="rrf"),
    "RRF+expansion": dict(mode="rrf", expand=True),
    "RRF+rerank": dict(mode="rrf", rerank=True),
    "RRF+expansion+rerank": dict(mode="rrf", expand=True, rerank=True),
}


def load_queries(path: Path, db: Path, n_caption: int, seed: int) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if n_caption:
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT article_id, caption FROM images "
                "WHERE caption_source = 'figcaption' AND length(caption) > 20 ORDER BY article_id"
            ).fetchall()
        random.Random(seed).shuffle(rows)
        queries += [
            {"query": caption, "relevant": [article_id], "auto": True}
            for article_id, caption in rows[:n_caption]
        ]
    return queries


def ranked_ids(pipe: SearchPipeline, query: str, depth: int, **kwargs) -> list[int]:
    results = pipe.search(query, top_k=depth, candidate_k=depth, with_consistency=False, **kwargs)
    return [int(r["article_id"]) for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K")
    parser.add_argument("--depth", type=int, default=50, help="Ranking depth used for MAP")
    parser.add_argument("--queries", type=Path, default=Path("evaluation/queries.json"))
    parser.add_argument("--caption-queries", type=int, default=0,
                        help="Add N automatic caption->article queries (seeded)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-rerank", action="store_true", help="Skip cross-encoder variants")
    parser.add_argument("--skip-feedback", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/clip_index"))
    parser.add_argument("--output", type=Path, help="Optional CSV for the results table")
    args = parser.parse_args()

    queries = load_queries(args.queries, args.db, args.caption_queries, args.seed)
    if not queries:
        raise SystemExit("No queries. Fill evaluation/queries.json or use --caption-queries N.")

    pipe = SearchPipeline(args.db, args.index_dir)
    rows = []

    for name, kwargs in VARIANTS.items():
        if args.skip_rerank and kwargs.get("rerank"):
            continue
        recalls, aps, ndcgs = [], [], []
        for q in queries:
            relevant = set(q["relevant"])
            ids = ranked_ids(pipe, q["query"], args.depth, **kwargs)
            recalls.append(recall_at_k(ids, relevant, args.k))
            aps.append(average_precision(ids, relevant))
            ndcgs.append(ndcg_at_k(ids, relevant, 10))
        n = len(queries)
        rows.append((name, sum(recalls) / n, sum(aps) / n, sum(ndcgs) / n))

    print(f"\n{len(queries)} queries | depth={args.depth}")
    print(f"{'Variant':<24}{'Recall@' + str(args.k):>10}{'MAP':>8}{'nDCG@10':>10}")
    for name, rec, ap, nd in rows:
        print(f"{name:<24}{rec:>10.3f}{ap:>8.3f}{nd:>10.3f}")

    if not args.skip_feedback:
        before, after, used = [], [], 0
        for q in queries:
            relevant = set(q["relevant"])
            ids = ranked_ids(pipe, q["query"], args.depth, mode="rrf")
            top = ids[: args.k]
            fb = {"relevant": [a for a in top if a in relevant],
                  "irrelevant": [a for a in top if a not in relevant]}
            if not fb["relevant"] or not fb["irrelevant"]:
                continue  # nothing to learn from (all right or all wrong)
            used += 1
            before.append(ndcg_at_k(ids, relevant, 10))
            new_ids = ranked_ids(pipe, q["query"], args.depth, mode="rrf", feedback=fb)
            after.append(ndcg_at_k(new_ids, relevant, 10))
        if used:
            print(f"\nRocchio feedback (1 round, {used} queries with both relevant and irrelevant in top-{args.k})")
            print(f"nDCG@10 before: {sum(before)/used:.3f} | after: {sum(after)/used:.3f}")
        else:
            print("\nRocchio feedback: no query had both relevant and irrelevant results in the top-K.")

    if args.output:
        import csv

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["variant", f"recall@{args.k}", "map", "ndcg@10"])
            writer.writerows(rows)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
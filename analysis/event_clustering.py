"""Group articles about the same evolving news story into timelines.

Clusters on CLIP text-embedding similarity between headlines, gated by a
maximum day gap so that two merely-similar-sounding stories from unrelated
weeks never merge into one "event". Within a cluster, articles are sorted by
date to form a timeline: a story's headline/image/caption as it is reported
over consecutive days.

Usage as a library (API serves these from cache when present):
    from analysis.event_clustering import build_event_clusters, load_or_build_events

Usage as a script, to precompute the cache for a large corpus ahead of time:
    poetry run python analysis/event_clustering.py --db data/processed/crossverify.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date as date_cls
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CACHE = Path("data/processed/event_clusters.json")
# Cosine distance threshold for "same story"; lower = stricter. CLIP text
# embeddings are L2-normalised, so cosine distance = 1 - cosine_similarity.
# Tuned on a 70-article BBC/Guardian pull: 0.15 kept every coherent cluster
# (the French-protests arc, Steele-John, Ebola, ...) while 0.18+ started
# merging unrelated articles that just share a topic word. Retune per corpus.
DEFAULT_DISTANCE_THRESHOLD = 0.15
DEFAULT_MIN_SIZE = 2
DEFAULT_MAX_DAYS_GAP = 10
# Dates further apart than max_days_gap are forced apart with this distance,
# which must be > distance_threshold so Agglomerative never merges them.
_GATED_DISTANCE = 10.0


def _fetch_articles(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.headline, a.date, a.source, a.url,
                   i.filepath, i.caption
            FROM articles a JOIN images i ON i.article_id = a.id
            WHERE a.headline IS NOT NULL AND a.date IS NOT NULL
            """
        ).fetchall()
    keys = ["article_id", "headline", "date", "source", "url", "filepath", "caption"]
    return [dict(zip(keys, row)) for row in rows]


def _day_gap(a: str, b: str) -> float:
    try:
        return abs((date_cls.fromisoformat(a[:10]) - date_cls.fromisoformat(b[:10])).days)
    except ValueError:
        return 0.0  # unparsable dates: don't gate on time, let embeddings decide


def build_event_clusters(
    db_path: Path,
    encode_text: Callable[[str], np.ndarray],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    min_size: int = DEFAULT_MIN_SIZE,
    max_days_gap: int = DEFAULT_MAX_DAYS_GAP,
) -> list[dict]:
    """Cluster articles into story timelines.

    `encode_text` returns an L2-normalised (1, dim) embedding for one string
    (e.g. ClipImageRetriever.encode_text), so this stays decoupled from CLIP.
    """

    articles = _fetch_articles(db_path)
    if len(articles) < min_size:
        return []

    vectors = np.vstack([encode_text(a["headline"]) for a in articles])
    similarity = vectors @ vectors.T
    distance = np.clip(1.0 - similarity, 0.0, None)

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if _day_gap(articles[i]["date"], articles[j]["date"]) > max_days_gap:
                distance[i, j] = distance[j, i] = _GATED_DISTANCE

    np.fill_diagonal(distance, 0.0)

    labels = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    ).fit_predict(distance)

    clusters: dict[int, list[dict]] = {}
    for article, label in zip(articles, labels):
        clusters.setdefault(int(label), []).append(article)

    events = []
    for members in clusters.values():
        if len(members) < min_size:
            continue
        members.sort(key=lambda a: a["date"])
        events.append(
            {
                "headline": members[0]["headline"],  # earliest article names the story
                "size": len(members),
                "start_date": members[0]["date"],
                "end_date": members[-1]["date"],
                "sources": sorted({m["source"] for m in members}),
                "articles": members,
            }
        )
    events.sort(key=lambda e: e["end_date"], reverse=True)
    return events


def load_or_build_events(
    db_path: Path,
    encode_text: Callable[[str], np.ndarray],
    cache_path: Path = DEFAULT_CACHE,
    **kwargs,
) -> list[dict]:
    """Serve a precomputed cache if present (built by this module's __main__),
    otherwise compute on the fly — fine for a demo-sized corpus, slow for a
    large one (one CLIP text encode per article)."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return build_event_clusters(db_path, encode_text, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/clip_index"))
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-days-gap", type=int, default=DEFAULT_MAX_DAYS_GAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    from retrieval.hybrid_search import ClipImageRetriever

    clip = ClipImageRetriever(args.index_dir)
    events = build_event_clusters(
        args.db,
        clip.encode_text,
        distance_threshold=args.distance_threshold,
        min_size=args.min_size,
        max_days_gap=args.max_days_gap,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"{len(events)} event cluster(s) (min size {args.min_size}) -> {args.out}")
    for e in events:
        print(f"  [{e['start_date']} -> {e['end_date']}] ({e['size']} articles) {e['headline'][:70]!r}")


if __name__ == "__main__":
    main()

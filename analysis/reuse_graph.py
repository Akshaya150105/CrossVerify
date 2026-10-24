"""Build a graph of articles connected by shared/near-duplicate images.

Nodes are articles; an edge means their images matched within a pHash
Hamming-distance threshold (same provenance signal as consistency/reuse.py's
find_reuse_pairs, reused here rather than duplicated). Isolated articles
(no reuse detected) are left out of the graph — it exists to show the
reuse network, not the whole corpus.

Usage as a library (API serves this from cache when present):
    from analysis.reuse_graph import build_reuse_graph, load_or_build_graph

Usage as a script, to precompute the cache for a large corpus ahead of time
(find_reuse_pairs is O(n^2) in the number of images, so do this offline for
a corpus of more than a few thousand images):
    poetry run python analysis/reuse_graph.py --db data/processed/crossverify.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency.reuse import DEFAULT_MAX_DISTANCE, find_reuse_pairs

DEFAULT_CACHE = Path("data/processed/reuse_graph.json")


def build_reuse_graph(db_path: Path, max_distance: int = DEFAULT_MAX_DISTANCE) -> dict:
    """Return {"nodes": [...], "edges": [...]} for every article that shares
    a near-duplicate image with at least one other article."""

    pairs = find_reuse_pairs(db_path, max_distance)

    nodes: dict[int, dict] = {}
    edges = []
    for p in pairs:
        for aid, headline, source, dt in (
            (p["article_a"], p["headline_a"], p["source_a"], p["date_a"]),
            (p["article_b"], p["headline_b"], p["source_b"], p["date_b"]),
        ):
            nodes.setdefault(aid, {"id": aid, "headline": headline, "source": source, "date": dt})
        edges.append(
            {
                "source": p["article_a"],
                "target": p["article_b"],
                "distance": p["distance"],
                "headline_overlap": p["headline_overlap"],
            }
        )

    return {"nodes": list(nodes.values()), "edges": edges}


def load_or_build_graph(
    db_path: Path,
    cache_path: Path = DEFAULT_CACHE,
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> dict:
    """Serve a precomputed cache if present, otherwise compute on the fly —
    fine for a demo-sized corpus, slow (O(n^2)) for a large one."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return build_reuse_graph(db_path, max_distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    graph = build_reuse_graph(args.db, args.max_distance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"{len(graph['nodes'])} node(s), {len(graph['edges'])} edge(s) -> {args.out}")


if __name__ == "__main__":
    main()

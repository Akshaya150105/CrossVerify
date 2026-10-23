"""Search the live CrossVerify article corpus with BM25."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from bm25 import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Keyword or natural-language article query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    args = parser.parse_args()

    retriever = BM25Retriever(args.db)
    retriever.load_documents()
    retriever.build_index()
    results = retriever.search(args.query, args.top_k)

    with sqlite3.connect(args.db) as conn:
        for rank, result in enumerate(results, start=1):
            row = conn.execute(
                """
                SELECT a.headline, a.source, a.url, i.caption
                FROM articles a
                LEFT JOIN images i ON i.article_id = a.id
                WHERE a.id = ?
                """,
                (result["article_id"],),
            ).fetchone()
            if row is None:
                continue
            headline, source, url, caption = row
            print(f"{rank}. score={result['score']:.3f} | {headline}")
            print(f"   source: {source} | caption: {caption}\n   url: {url}\n")


if __name__ == "__main__":
    main()

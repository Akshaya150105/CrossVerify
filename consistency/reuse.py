"""Visual-reuse detection with perceptual hashes (guide section 8).

Group images whose pHash Hamming distance is small, then compare the reporting
context (headline, date, source) so a reviewer can decide whether it is the
same story or a recycled image.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MAX_DISTANCE = 6  # of 64 bits; 0 = identical


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def headline_overlap(a: str, b: str) -> float:
    """Jaccard overlap of headline words: low = materially different context."""

    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def matches_for(
    conn: sqlite3.Connection,
    phash: str | None,
    exclude_article_id: int | None = None,
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> list[dict]:
    """Other articles whose image is the same/near-identical to `phash`."""

    if not phash:
        return []
    rows = conn.execute(
        """
        SELECT a.id, a.headline, a.source, a.date, a.url, i.phash
        FROM images i JOIN articles a ON a.id = i.article_id
        WHERE i.phash IS NOT NULL
        """
    ).fetchall()
    out = []
    for article_id, headline, source, date, url, other in rows:
        if exclude_article_id is not None and article_id == exclude_article_id:
            continue
        try:
            distance = hamming(phash, other)
        except ValueError:
            continue
        if distance <= max_distance:
            out.append(
                {
                    "article_id": article_id,
                    "headline": headline,
                    "source": source,
                    "date": date,
                    "url": url,
                    "distance": distance,
                }
            )
    return sorted(out, key=lambda m: m["distance"])


def find_reuse_pairs(db_path: Path, max_distance: int = DEFAULT_MAX_DISTANCE) -> list[dict]:
    """All near-duplicate image pairs across different articles."""

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.headline, a.source, a.date, a.url, i.phash, i.filepath
            FROM images i JOIN articles a ON a.id = i.article_id
            WHERE i.phash IS NOT NULL
            """
        ).fetchall()

    pairs = []
    for x in range(len(rows)):
        for y in range(x + 1, len(rows)):
            a, b = rows[x], rows[y]
            if a[0] == b[0]:
                continue
            try:
                distance = hamming(a[5], b[5])
            except ValueError:
                continue
            if distance <= max_distance:
                pairs.append(
                    {
                        "article_a": a[0], "article_b": b[0],
                        "headline_a": a[1], "headline_b": b[1],
                        "source_a": a[2], "source_b": b[2],
                        "date_a": a[3], "date_b": b[3],
                        "image_a": a[6], "image_b": b[6],
                        "distance": distance,
                        "headline_overlap": round(headline_overlap(a[1], b[1]), 3),
                    }
                )
    return sorted(pairs, key=lambda p: (p["distance"], p["headline_overlap"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        help="Write pairs here for MANUAL review before adding any to the benchmark.",
    )
    args = parser.parse_args()

    pairs = find_reuse_pairs(args.db, args.max_distance)
    print(f"{len(pairs)} near-duplicate image pair(s) (pHash distance <= {args.max_distance})")
    for p in pairs:
        print(
            f"  d={p['distance']} overlap={p['headline_overlap']} | "
            f"#{p['article_a']} {p['headline_a'][:60]!r}  <->  #{p['article_b']} {p['headline_b'][:60]!r}"
        )

    if args.candidates_csv and pairs:
        args.candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.candidates_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pairs[0]) + ["reviewer_decision", "notes"])
            writer.writeheader()
            writer.writerows(pairs)
        print(f"Wrote {args.candidates_csv} (fill reviewer_decision/notes by hand)")


if __name__ == "__main__":
    main()
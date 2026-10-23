"""Create a reproducible, auditable image-caption consistency benchmark."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from pathlib import Path


FIELDS = [
    "image_path", "caption", "label", "notes", "article_id", "caption_article_id"
]


def load_source_pairs(db_path: Path) -> list[dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT i.article_id, i.filepath, i.caption
            FROM images AS i
            JOIN articles AS a ON a.id = i.article_id
            WHERE i.caption_source = 'figcaption'
              AND i.caption IS NOT NULL
              AND length(trim(i.caption)) >= 20
            ORDER BY i.article_id
            """
        ).fetchall()
    return [
        {"article_id": str(article_id), "image_path": path, "caption": caption}
        for article_id, path, caption in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled_benchmark/benchmark.csv"))
    parser.add_argument("--pairs", type=int, default=40, help="Number of original and mismatch pairs each.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}. Pass --overwrite to regenerate it.")
    source_pairs = load_source_pairs(args.db)
    if len(source_pairs) < args.pairs:
        raise SystemExit(f"Only {len(source_pairs)} eligible figcaption pairs; requested {args.pairs}.")

    rng = random.Random(args.seed)
    selected = rng.sample(source_pairs, args.pairs)
    shuffled_captions = selected[:]
    rng.shuffle(shuffled_captions)
    # Rotate if a rare shuffle leaves an image with its original caption.
    while any(a["article_id"] == b["article_id"] for a, b in zip(selected, shuffled_captions)):
        shuffled_captions = shuffled_captions[1:] + shuffled_captions[:1]

    rows: list[dict[str, str]] = []
    for pair in selected:
        rows.append({
            "image_path": pair["image_path"],
            "caption": pair["caption"],
            "label": "consistent",
            "notes": "Publisher original figcaption; manually verify before final reporting.",
            "article_id": pair["article_id"],
            "caption_article_id": pair["article_id"],
        })
    for image_pair, caption_pair in zip(selected, shuffled_captions):
        rows.append({
            "image_path": image_pair["image_path"],
            "caption": caption_pair["caption"],
            "label": "mismatched",
            "notes": "Synthetic negative: caption deliberately taken from a different article.",
            "article_id": image_pair["article_id"],
            "caption_article_id": caption_pair["article_id"],
        })
    rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"consistent={args.pairs}, mismatched={args.pairs}, seed={args.seed}")


if __name__ == "__main__":
    main()

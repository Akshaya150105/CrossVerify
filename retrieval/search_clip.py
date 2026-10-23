"""Search the persisted CrossVerify CLIP image index with text."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def get_text_features(
    model: CLIPModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Generate CLIP text embeddings.

    Explicitly runs the CLIP text encoder and projection layer
    to avoid Transformers versions where get_text_features()
    returns BaseModelOutputWithPooling.
    """

    outputs = model.text_model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )

    # Pooled representation from CLIP's text transformer
    pooled_output = outputs.pooler_output

    # Project into the shared CLIP image/text embedding space
    features = model.text_projection(
        pooled_output
    )

    return features


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "query",
        help="Natural-language image query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path(
            "data/processed/crossverify.db"
        ),
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(
            "data/processed/clip_index"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Check CLIP index
    # ---------------------------------------------------------

    metadata_path = (
        args.index_dir / "metadata.json"
    )

    if not metadata_path.is_file():
        raise SystemExit(
            "CLIP index not found. "
            "Run indexing/build_clip_index.py first."
        )

    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    model_name = metadata["model"]

    # ---------------------------------------------------------
    # Load FAISS index
    # ---------------------------------------------------------

    index_path = (
        args.index_dir / "images.faiss"
    )

    ids_path = (
        args.index_dir / "article_ids.npy"
    )

    if not index_path.is_file():
        raise SystemExit(
            f"FAISS index not found: {index_path}"
        )

    if not ids_path.is_file():
        raise SystemExit(
            f"Article ID mapping not found: {ids_path}"
        )

    index = faiss.read_index(
        str(index_path)
    )

    article_ids = np.load(
        ids_path
    )

    if index.ntotal == 0:
        raise SystemExit(
            "The FAISS index contains no vectors."
        )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading {model_name} on {device}"
    )

    # ---------------------------------------------------------
    # Load CLIP
    # ---------------------------------------------------------

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model = CLIPModel.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Encode text query
    # ---------------------------------------------------------

    inputs = processor(
        text=[args.query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.inference_mode():

        query_features = get_text_features(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            device=device,
        )

    # ---------------------------------------------------------
    # Convert to NumPy
    # ---------------------------------------------------------

    query = (
        query_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # ---------------------------------------------------------
    # Normalise query vector
    # ---------------------------------------------------------

    faiss.normalize_L2(
        query
    )

    # ---------------------------------------------------------
    # Search FAISS
    # ---------------------------------------------------------

    top_k = min(
        args.top_k,
        index.ntotal,
    )

    scores, positions = index.search(
        query,
        top_k,
    )

    # ---------------------------------------------------------
    # Retrieve metadata from SQLite
    # ---------------------------------------------------------

    with sqlite3.connect(
        args.db
    ) as conn:

        print()
        print(
            f'Query: "{args.query}"'
        )
        print(
            f"Results: {top_k}"
        )
        print("-" * 80)

        for rank, (
            score,
            position,
        ) in enumerate(
            zip(
                scores[0],
                positions[0],
            ),
            start=1,
        ):

            # FAISS can return -1 when there
            # aren't enough results.
            if position < 0:
                continue

            article_id = int(
                article_ids[position]
            )

            # -------------------------------------------------
            # Fetch article/image information
            # -------------------------------------------------

            row = conn.execute(
                """
                SELECT
                    a.headline,
                    a.source,
                    a.url,
                    i.caption,
                    i.filepath,
                    i.caption_source
                FROM articles AS a
                JOIN images AS i
                    ON i.article_id = a.id
                WHERE a.id = ?
                """,
                (article_id,),
            ).fetchone()

            if row is None:

                print(
                    f"{rank}. score={score:.3f}"
                    f" | Article ID {article_id}"
                    f" | Metadata not found"
                )

                continue

            (
                headline,
                source,
                url,
                caption,
                filepath,
                caption_source,
            ) = row

            # -------------------------------------------------
            # Display result
            # -------------------------------------------------

            print(
                f"{rank}. "
                f"score={score:.3f} | "
                f"{headline}"
            )

            print(
                f"   article_id: {article_id}"
            )

            print(
                f"   source: {source}"
            )

            print(
                f"   caption "
                f"({caption_source}): "
                f"{caption}"
            )

            print(
                f"   image: {filepath}"
            )

            print(
                f"   url: {url}"
            )

            print()


if __name__ == "__main__":
    main()
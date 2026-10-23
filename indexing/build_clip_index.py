"""Build a persisted CLIP image index from the CrossVerify database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise embedding vectors."""
    vectors = vectors.astype(np.float32, copy=False)
    faiss.normalize_L2(vectors)
    return vectors


def load_records(db_path: Path) -> list[tuple[int, str]]:
    """Load article IDs and image paths from the SQLite database."""

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                i.article_id,
                i.filepath
            FROM images AS i
            JOIN articles AS a
                ON a.id = i.article_id
            ORDER BY i.article_id
            """
        ).fetchall()

    return [
        (int(article_id), filepath)
        for article_id, filepath in rows
    ]


def get_image_features(
    model: CLIPModel,
    pixel_values: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Extract CLIP image embeddings.

    This explicitly runs the vision encoder and projection layer
    instead of using model.get_image_features(), because some
    Transformers versions return BaseModelOutputWithPooling
    from get_image_features().
    """

    outputs = model.vision_model(
        pixel_values=pixel_values.to(device)
    )

    # Pooled output from CLIP's vision transformer
    pooled_output = outputs.pooler_output

    # Project into CLIP's shared image/text embedding space
    features = model.visual_projection(
        pooled_output
    )

    return features


def encode_images(
    records: list[tuple[int, str]],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all readable images into normalised CLIP vectors."""

    vectors: list[np.ndarray] = []
    kept_ids: list[int] = []

    total = len(records)

    for start in range(0, total, batch_size):

        batch = records[start : start + batch_size]

        images: list[Image.Image] = []
        ids: list[int] = []

        # ---------------------------------------------------------
        # Load images
        # ---------------------------------------------------------

        for article_id, filename in batch:

            try:
                image_path = Path(filename)

                if not image_path.is_file():
                    print(
                        f"Skipping missing image: "
                        f"{image_path}"
                    )
                    continue

                with Image.open(image_path) as image:
                    images.append(
                        image.convert("RGB")
                    )

                ids.append(article_id)

            except (OSError, ValueError) as error:

                print(
                    f"Skipping unreadable image "
                    f"{filename}: {error}"
                )

        if not images:
            continue

        # ---------------------------------------------------------
        # Preprocess images
        # ---------------------------------------------------------

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # CLIP image encoding
        # ---------------------------------------------------------

        with torch.inference_mode():

            features = get_image_features(
                model=model,
                pixel_values=inputs["pixel_values"],
                device=device,
            )

        # ---------------------------------------------------------
        # Convert to NumPy and normalise
        # ---------------------------------------------------------

        batch_vectors = features.cpu().numpy()

        batch_vectors = normalise(
            batch_vectors
        )

        vectors.append(batch_vectors)

        kept_ids.extend(ids)

        processed = min(
            start + len(batch),
            total,
        )

        print(
            f"Encoded {processed}/{total} records "
            f"({len(ids)} images in this batch)"
        )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    if not vectors:
        raise RuntimeError(
            "No readable images were found in the database."
        )

    embeddings = np.vstack(vectors)

    article_ids = np.asarray(
        kept_ids,
        dtype=np.int64,
    )

    return embeddings, article_ids


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path(
            "data/processed/crossverify.db"
        ),
        help="Path to CrossVerify SQLite database.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/clip_index"
        ),
        help="Directory where the FAISS index is saved.",
    )

    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images processed per batch.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Validate database
    # -------------------------------------------------------------

    if not args.db.is_file():

        raise SystemExit(
            f"Database not found: {args.db}"
        )

    # -------------------------------------------------------------
    # Load records
    # -------------------------------------------------------------

    records = load_records(args.db)

    if not records:

        raise SystemExit(
            "No image records found. "
            "Run the data collector first."
        )

    print(
        f"Found {len(records)} image records."
    )

    # -------------------------------------------------------------
    # Select device
    # -------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading {args.model} on {device}"
    )

    # -------------------------------------------------------------
    # Load CLIP
    # -------------------------------------------------------------

    processor = CLIPProcessor.from_pretrained(
        args.model
    )

    model = CLIPModel.from_pretrained(
        args.model
    )

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Encode images
    # -------------------------------------------------------------

    embeddings, article_ids = encode_images(
        records=records,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    print(
        f"Generated embeddings: "
        f"{embeddings.shape}"
    )

    # -------------------------------------------------------------
    # Build FAISS index
    # -------------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save FAISS index
    # -------------------------------------------------------------

    index_path = (
        args.output_dir / "images.faiss"
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    # -------------------------------------------------------------
    # Save article IDs
    # -------------------------------------------------------------

    ids_path = (
        args.output_dir / "article_ids.npy"
    )

    np.save(
        ids_path,
        article_ids,
    )

    # -------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------

    metadata_path = (
        args.output_dir / "metadata.json"
    )

    metadata = {
        "model": args.model,
        "vectors": int(index.ntotal),
        "dimension": int(index.d),
        "device": device,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print()
    print("CLIP image index successfully created.")
    print(
        f"Vectors  : {index.ntotal}"
    )
    print(
        f"Dimension: {index.d}"
    )
    print(
        f"FAISS    : {index_path}"
    )
    print(
        f"IDs      : {ids_path}"
    )
    print(
        f"Metadata : {metadata_path}"
    )


if __name__ == "__main__":
    main()
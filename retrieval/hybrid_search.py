"""Fuse CLIP image retrieval and BM25 article retrieval with RRF."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25 import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion  # noqa: F401  (re-exported)


class ClipImageRetriever:
    def __init__(self, index_dir: Path):
        index_dir = Path(index_dir)
        metadata_path = index_dir / "metadata.json"
        index_path = index_dir / "images.faiss"
        ids_path = index_dir / "article_ids.npy"

        for path, label in (
            (metadata_path, "CLIP metadata"),
            (index_path, "CLIP index"),
            (ids_path, "Article ID file"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.index = faiss.read_index(str(index_path))
        self.article_ids = np.load(ids_path)

        if self.index.ntotal != len(self.article_ids):
            raise ValueError(
                "FAISS index and article_ids.npy do not match: "
                f"{self.index.ntotal} vectors vs {len(self.article_ids)} article IDs"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = metadata["model"]

        print(f"Loading CLIP model: {model_name}")
        print(f"Device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"Loaded CLIP index: {self.index.ntotal} vectors")

    def _get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the CLIP text encoder + projection explicitly, because in some
        Transformers versions model.get_text_features() returns an output
        object rather than a Tensor."""

        outputs = self.model.text_model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        return self.model.text_projection(outputs.pooler_output)

    # ---- encoding ----

    def encode_text(self, text: str) -> np.ndarray:
        """Return an L2-normalised (1, dim) CLIP text embedding."""

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            features = self._get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        vector = features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        return vector

    def encode_image(self, image) -> np.ndarray:
        """Return an L2-normalised (1, dim) CLIP image embedding (PIL image)."""

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
            features = self.model.visual_projection(outputs.pooler_output)
        vector = features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        return vector

    # ---- search ----

    def vector_for_article(self, article_id: int) -> np.ndarray | None:
        """Return the stored image embedding for an article, if indexed."""

        positions = np.where(self.article_ids == int(article_id))[0]
        if len(positions) == 0:
            return None
        return self.index.reconstruct(int(positions[0])).reshape(1, -1)

    def search_vector(self, vector: np.ndarray, top_k: int) -> list[dict[str, float | int]]:
        """Search the image index with an already-encoded query vector."""

        if self.index.ntotal == 0:
            return []

        vector = np.ascontiguousarray(vector, dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, positions = self.index.search(vector, min(top_k, self.index.ntotal))

        return [
            {"article_id": int(self.article_ids[position]), "score": float(score)}
            for score, position in zip(scores[0], positions[0])
            if position != -1
        ]

    def search(self, text: str, top_k: int) -> list[dict[str, float | int]]:
        """Search the CLIP image index using a text query."""

        if not text.strip():
            return []
        return self.search_vector(self.encode_text(text), top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--db", type=Path, default=Path("data/processed/crossverify.db"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/clip_index"))
    args = parser.parse_args()

    if not (args.index_dir / "images.faiss").is_file():
        raise SystemExit("CLIP index not found. Run indexing/build_clip_index.py first.")

    clip = ClipImageRetriever(args.index_dir)
    clip_results = clip.search(args.query, args.candidate_k)

    bm25 = BM25Retriever(args.db)
    bm25.load_documents()
    bm25.build_index()
    bm25_results = bm25.search(args.query, args.candidate_k)

    fused = reciprocal_rank_fusion([clip_results, bm25_results], args.rrf_k)[: args.top_k]

    print()
    print("=" * 70)
    print(f"Query: {args.query}")
    print("=" * 70)

    with sqlite3.connect(args.db) as conn:
        for rank, (article_id, rrf_score) in enumerate(fused, start=1):
            row = conn.execute(
                """
                SELECT a.headline, a.source, a.url, i.caption, i.filepath
                FROM articles a JOIN images i ON i.article_id = a.id
                WHERE a.id = ?
                """,
                (article_id,),
            ).fetchone()
            if row is None:
                continue
            headline, source, url, caption, filepath = row
            print(f"{rank}. rrf={rrf_score:.5f} | {headline}")
            print(f"   source: {source}")
            print(f"   caption: {caption}")
            print(f"   image: {filepath}")
            print(f"   url: {url}")
            print()


if __name__ == "__main__":
    main()
"""Image-caption consistency using CLIP cosine similarity.

A low score means weak image-caption *alignment*. It is NOT evidence that a
claim is false or that anyone is spreading misinformation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

DEFAULT_SCORES_CSV = Path("data/labeled_benchmark/clip_scores.csv")


def load_threshold(scores_csv: Path = DEFAULT_SCORES_CSV) -> float | None:
    """Youden-J threshold fitted on the TRAIN split only (matches the eval script)."""

    scores_csv = Path(scores_csv)
    if not scores_csv.is_file():
        return None
    frame = pd.read_csv(scores_csv)
    train = frame[frame["split"] == "train"]
    if train.empty or train["label"].nunique() < 2:
        return None
    targets = (train["label"] == "consistent").astype(int)
    fpr, tpr, thresholds = roc_curve(targets, train["clip_cosine_similarity"])
    return float(thresholds[np.argmax(tpr - fpr)])


class ConsistencyChecker:
    def __init__(self, clip, threshold: float | None):
        self.clip = clip  # ClipImageRetriever (encode_text / encode_image)
        self.threshold = threshold

    @staticmethod
    def cosine(image_vec: np.ndarray, text_vec: np.ndarray) -> float:
        return float(np.asarray(image_vec).reshape(-1) @ np.asarray(text_vec).reshape(-1))

    def verdict(self, similarity: float) -> dict:
        if self.threshold is None:
            return {"similarity": similarity, "threshold": None, "status": "unknown",
                    "note": "No calibrated threshold (run consistency/evaluate_clip_consistency.py)."}
        aligned = similarity >= self.threshold
        return {
            "similarity": similarity,
            "threshold": self.threshold,
            "status": "aligned" if aligned else "low_alignment",
            "note": "Image and caption look semantically aligned." if aligned else
                    "Low image-caption alignment. This does not mean the caption is false.",
        }

    def check_vector(self, image_vec: np.ndarray, caption: str) -> dict:
        return self.verdict(self.cosine(image_vec, self.clip.encode_text(caption)))

    def check_image(self, image, caption: str) -> dict:
        return self.verdict(self.cosine(self.clip.encode_image(image), self.clip.encode_text(caption)))
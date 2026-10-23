"""Score the labelled benchmark with CLIP image-caption cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor


def encode_scores(frame: pd.DataFrame, processor: CLIPProcessor, model: CLIPModel, device: str, batch_size: int) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        images = []
        for path in batch["image_path"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, text=batch["caption"].tolist(), return_tensors="pt", padding=True, truncation=True)
        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            # Using the combined CLIP output avoids version-specific return
            # types from get_image_features()/get_text_features().
            outputs = model(**model_inputs)
        image_vectors = outputs.image_embeds.cpu().numpy().astype(np.float32)
        text_vectors = outputs.text_embeds.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(image_vectors)
        faiss.normalize_L2(text_vectors)
        scores.append(np.sum(image_vectors * text_vectors, axis=1))
        print(f"Scored {min(start + len(batch), len(frame))}/{len(frame)} pairs")
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/labeled_benchmark/benchmark.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled_benchmark/clip_scores.csv"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.benchmark)
    required = {"image_path", "caption", "label"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"Benchmark is missing columns: {sorted(missing)}")
    frame = frame[frame["label"].isin(["consistent", "mismatched"])].reset_index(drop=True)
    targets = (frame["label"] == "consistent").astype(int).to_numpy()
    if len(np.unique(targets)) != 2:
        raise SystemExit("Benchmark needs both consistent and mismatched labels.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {args.model}\nDevice: {device}")
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    frame["clip_cosine_similarity"] = encode_scores(frame, processor, model, device, args.batch_size)

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=args.seed, stratify=targets)
    fpr, tpr, thresholds = roc_curve(targets[train_idx], frame.loc[train_idx, "clip_cosine_similarity"])
    threshold = float(thresholds[np.argmax(tpr - fpr)])
    predictions = (frame["clip_cosine_similarity"].to_numpy() >= threshold).astype(int)
    frame["split"] = "train"
    frame.loc[test_idx, "split"] = "test"
    frame["predicted_label"] = np.where(predictions == 1, "consistent", "mismatched")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Threshold selected on training split: {threshold:.4f}")
    print("Held-out test metrics")
    print(f"Precision: {precision_score(targets[test_idx], predictions[test_idx], zero_division=0):.3f}")
    print(f"Recall:    {recall_score(targets[test_idx], predictions[test_idx], zero_division=0):.3f}")
    print(f"F1:        {f1_score(targets[test_idx], predictions[test_idx], zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(targets[test_idx], frame.loc[test_idx, 'clip_cosine_similarity']):.3f}")
    print(f"Saved scored rows to {args.output}")


if __name__ == "__main__":
    main()

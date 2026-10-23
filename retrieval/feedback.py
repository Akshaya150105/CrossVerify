"""Rocchio relevance feedback in CLIP embedding space (guide section 9).

    q_new = alpha * q_original + beta * mean(relevant) - gamma * mean(irrelevant)
"""

from __future__ import annotations

import numpy as np


def rocchio(
    query_vec: np.ndarray,
    relevant: list[np.ndarray],
    irrelevant: list[np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Return the updated, L2-normalised (1, dim) query vector."""

    q = alpha * np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    if relevant:
        q = q + beta * np.mean(np.vstack(relevant), axis=0, keepdims=True)
    if irrelevant:
        q = q - gamma * np.mean(np.vstack(irrelevant), axis=0, keepdims=True)
    norm = np.linalg.norm(q)
    return (q / norm if norm else q).astype(np.float32)
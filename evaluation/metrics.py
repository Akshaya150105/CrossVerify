"""Binary-relevance IR metrics: Recall@K, Average Precision, nDCG@K."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in the top k (= hit rate when |relevant| = 1)."""

    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def average_precision(ranked: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
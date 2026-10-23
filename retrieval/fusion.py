"""Reciprocal Rank Fusion (torch-free so it is cheap to import and test)."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Return (article_id, score) sorted by sum over lists of 1 / (k + rank)."""

    scores: defaultdict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            scores[int(result["article_id"])] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
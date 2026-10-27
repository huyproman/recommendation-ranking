"""Per-user Top-K ranking metrics. Ranks are 1-based."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

K_VALUES = (5, 10, 20)


def top_k(ranking: Sequence[int], k: int) -> list[int]:
    """L_K: the first min(K, |candidates|) items of a full candidate ranking."""
    return list(ranking[: min(k, len(ranking))])


def hits(ranking: Sequence[int], gt: set[int], k: int) -> int:
    return sum(1 for m in top_k(ranking, k) if m in gt)


def recall_at_k(ranking: Sequence[int], gt: set[int], k: int) -> float:
    return hits(ranking, gt, k) / len(gt)


def hitrate_at_k(ranking: Sequence[int], gt: set[int], k: int) -> float:
    return 1.0 if hits(ranking, gt, k) > 0 else 0.0


def dcg_at_k(ranking: Sequence[int], gt: set[int], k: int) -> float:
    return sum(1.0 / math.log2(r + 1) for r, m in enumerate(top_k(ranking, k), start=1) if m in gt)


def idcg_at_k(n_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(r + 1) for r in range(1, min(k, n_relevant) + 1))


def ndcg_at_k(ranking: Sequence[int], gt: set[int], k: int) -> float:
    return dcg_at_k(ranking, gt, k) / idcg_at_k(len(gt), k)


def user_metrics(ranking: Sequence[int], gt: set[int], k: int) -> dict[str, float]:
    if len(gt) == 0:
        raise ValueError("ground truth must be non-empty for an eligible user")
    return {
        "recall": recall_at_k(ranking, gt, k),
        "hitrate": hitrate_at_k(ranking, gt, k),
        "ndcg": ndcg_at_k(ranking, gt, k),
    }


def macro_average(values: Iterable[float]) -> float | None:
    """Unweighted mean over users; None for an empty cohort (never a fabricated 0.0)."""
    vals = list(values)
    return (math.fsum(vals) / len(vals)) if vals else None


def catalog_coverage(top_lists: Iterable[Sequence[int]], catalog_size: int) -> float:
    """|union of recommended top-K items| / |C_s| (stage level, not per-user averaged)."""
    union: set[int] = set()
    for lst in top_lists:
        union.update(int(x) for x in lst)
    return len(union) / catalog_size

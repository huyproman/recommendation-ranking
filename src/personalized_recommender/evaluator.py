"""Model-agnostic full-catalog Top-K evaluator.

Any model implementing :class:`Ranker` (``fit`` on a :class:`Prefix`, ``rank`` returning a full
ordering of the provided candidate ids) can be evaluated without changing split, candidate or
metric semantics. The evaluator verifies that every returned ranking is an exact permutation of
``Candidates_s(u)``: the full catalog is ranked, nothing sampled, nothing added.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .errors import EvaluationError
from .metrics import K_VALUES, catalog_coverage, macro_average, user_metrics
from .stage import COLD_START_MAX_HISTORY, Prefix, Stage

METRIC_NAMES = ("recall", "hitrate", "ndcg")


class Ranker(Protocol):
    name: str

    def fit(self, prefix: Prefix) -> "Ranker": ...

    def rank(self, user_id: int, candidates: np.ndarray) -> np.ndarray: ...


def rank_by_score(candidates: np.ndarray, score: np.ndarray) -> np.ndarray:
    """General rule: score DESC, raw numeric movie_id ASC. Scores must be finite."""
    candidates = np.asarray(candidates, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if score.shape != candidates.shape:
        raise EvaluationError("INVALID_SCORES", "one score per candidate is required")
    if not np.all(np.isfinite(score)):
        raise EvaluationError("NONFINITE_SCORE", "all ranking scores must be finite")
    return candidates[np.lexsort((candidates, -score))]


def cohort_of(history: int) -> str:
    if history == 0:
        return "cold_start_0"
    if history <= COLD_START_MAX_HISTORY:
        return "cold_start_1_to_5"
    return "warm_6_plus"


def _aggregate(per_user: list[dict[int, dict[str, float]]], ks) -> dict:
    return {str(k): {m: macro_average(u[k][m] for u in per_user) for m in METRIC_NAMES} for k in ks}


def evaluate(stage: Stage, ranker: Ranker, ks=K_VALUES) -> dict:
    """Evaluate a fitted ranker on a stage. Returns a deterministic JSON-serialisable dict."""
    users = stage.eligible_users
    if len(users) == 0:
        raise EvaluationError("NO_EVALUABLE_USERS", f"stage {stage.name!r} has no eligible users")
    ks = tuple(sorted(ks))
    kmax = ks[-1]
    catalog_size = len(stage.prefix.catalog)
    per_user: list[dict[int, dict[str, float]]] = []
    cohorts: dict[str, list[int]] = {"cold_start_0": [], "cold_start_1_to_5": [], "warm_6_plus": []}
    tops: dict[int, list[list[int]]] = {k: [] for k in ks}
    short_lists = 0
    ranked_total = 0
    for idx, u in enumerate(users.tolist()):
        cands = stage.candidates(u)
        ranking = np.asarray(ranker.rank(u, cands), dtype=np.int64)
        if len(ranking) != len(cands) or not np.array_equal(np.sort(ranking), cands):
            raise EvaluationError(
                "INVALID_RANKING",
                f"ranker {ranker.name!r} did not return a permutation of the full candidate set for user {u}",
            )
        ranked_total += len(ranking)
        short_lists += int(len(cands) < kmax)
        head = ranking[:kmax].tolist()
        gt = set(stage.ground_truth[u].tolist())
        per_user.append({k: user_metrics(head, gt, k) for k in ks})
        for k in ks:
            tops[k].append(head[:k])
        cohorts[cohort_of(stage.prefix.history_size(u))].append(idx)

    def slice_(idxs: list[int]) -> dict:
        return {"n_users": len(idxs), "metrics_by_k": _aggregate([per_user[i] for i in idxs], ks)}

    metrics = _aggregate(per_user, ks)
    for k in ks:
        metrics[str(k)]["catalog_coverage"] = catalog_coverage(tops[k], catalog_size)
    return {
        "metrics_by_k": metrics,
        "eligible_users_count": int(len(users)),
        "ranked_candidate_pairs": int(ranked_total),
        "users_with_fewer_candidates_than_max_k": short_lists,
        "cold_start": {
            "definition": "count of ALL observed prefix ratings (positive and non-positive); 0..5 = cold start",
            "cold_start_0_to_5": slice_(sorted(cohorts["cold_start_0"] + cohorts["cold_start_1_to_5"])),
            "cold_start_0": slice_(cohorts["cold_start_0"]),
            "cold_start_1_to_5": slice_(cohorts["cold_start_1_to_5"]),
            "warm_6_plus": slice_(cohorts["warm_6_plus"]),
        },
    }

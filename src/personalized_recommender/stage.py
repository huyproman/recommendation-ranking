"""Snapshot prefix, feedback semantics, stage catalog, candidates, ground truth and eligibility.

A :class:`Prefix` is constructed exclusively from the allowed
historical prefix events of a stage; every model statistic and the candidate catalog derive
from it. Held-out events are only used as relevance targets by :class:`Stage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import MOVIE, RATING, USER, canonical_sort
from .split import Split

POSITIVE = "POSITIVE"
NONPOSITIVE = "OBSERVED_NONPOSITIVE"
UNKNOWN = "UNKNOWN"
POSITIVE_THRESHOLD = 4
COLD_START_MAX_HISTORY = 5


def feedback_class(rating: int | None) -> str:
    """rating>=4 -> POSITIVE; 1..3 -> OBSERVED_NONPOSITIVE; no observation (None) -> UNKNOWN."""
    if rating is None:
        return UNKNOWN
    if not 1 <= int(rating) <= 5:
        raise ValueError(f"rating {rating} outside 1..5")
    return POSITIVE if int(rating) >= POSITIVE_THRESHOLD else NONPOSITIVE


def _group(events: np.ndarray) -> dict[int, np.ndarray]:
    """user_id -> sorted array of movie_ids for the given events."""
    if len(events) == 0:
        return {}
    order = np.lexsort((events[:, MOVIE], events[:, USER]))
    ev = events[order]
    users, starts = np.unique(ev[:, USER], return_index=True)
    bounds = list(starts[1:]) + [len(ev)]
    return {int(u): ev[s:e, MOVIE].copy() for u, s, e in zip(users, starts, bounds)}


_EMPTY = np.empty(0, dtype=np.int64)


class Prefix:
    """Allowed historical prefix P_s of a stage (TRAIN for validation, TRAIN+VALIDATION for test)."""

    def __init__(self, events: np.ndarray):
        ev = canonical_sort(np.asarray(events, dtype=np.int64).reshape(-1, 4))
        self.events = ev
        # C_s: every item with >=1 observed rating (any value) in the prefix, ascending numeric id.
        self.catalog = np.unique(ev[:, MOVIE]).astype(np.int64)
        self.positive_events = ev[ev[:, RATING] >= POSITIVE_THRESHOLD]
        self._seen = _group(ev)  # all observed ratings (positive and non-positive)
        self._pos = _group(self.positive_events)  # positive-only
        self._ratings = {(int(u), int(m)): int(r) for u, m, r, _ in ev} if len(ev) <= 200_000 else None
        self.users = np.array(sorted(self._seen), dtype=np.int64)

    def seen(self, user_id: int) -> np.ndarray:
        return self._seen.get(int(user_id), _EMPTY)

    def positives(self, user_id: int) -> np.ndarray:
        return self._pos.get(int(user_id), _EMPTY)

    def history_size(self, user_id: int) -> int:
        """All observed prefix ratings (positive and non-positive) for cold-start classification."""
        return int(len(self.seen(user_id)))

    def feedback(self, user_id: int, movie_id: int) -> str:
        if self._ratings is not None:
            return feedback_class(self._ratings.get((int(user_id), int(movie_id))))
        m = (self.events[:, USER] == user_id) & (self.events[:, MOVIE] == movie_id)
        idx = np.nonzero(m)[0]
        return feedback_class(int(self.events[idx[0], RATING]) if len(idx) else None)

    def candidates(self, user_id: int) -> np.ndarray:
        """Candidates_s(u) = C_s \\ Seen_s(u) (ascending movie id)."""
        return np.setdiff1d(self.catalog, self.seen(user_id), assume_unique=True)


EXCLUSION_REASONS = (
    "no_heldout_positive",
    "positives_only_out_of_catalog",
    "positives_not_in_candidates",
    "no_candidates",
)


@dataclass
class Stage:
    name: str
    prefix: Prefix
    heldout: np.ndarray
    eligible_users: np.ndarray
    ground_truth: dict[int, np.ndarray]
    diagnostics: dict = field(default_factory=dict)

    def candidates(self, user_id: int) -> np.ndarray:
        return self.prefix.candidates(user_id)


def build_stage(name: str, prefix_events: np.ndarray, heldout_events: np.ndarray) -> Stage:
    prefix = Prefix(prefix_events)
    heldout = canonical_sort(np.asarray(heldout_events, dtype=np.int64).reshape(-1, 4))
    pos_heldout = heldout[heldout[:, RATING] >= POSITIVE_THRESHOLD]
    pos_by_user = _group(pos_heldout)
    heldout_users = np.unique(heldout[:, USER]) if len(heldout) else _EMPTY
    reasons = {r: 0 for r in EXCLUSION_REASONS}
    gt: dict[int, np.ndarray] = {}
    eligible: list[int] = []
    oop_events = 0
    oop_users = 0
    zero_history_eligible = 0
    for u in heldout_users.tolist():
        pos = pos_by_user.get(u, _EMPTY)
        if len(pos) == 0:
            reasons["no_heldout_positive"] += 1
            continue
        in_cat = np.isin(pos, prefix.catalog)
        n_oop = int((~in_cat).sum())
        oop_events += n_oop
        oop_users += int(n_oop > 0)
        cands = prefix.candidates(u)
        g = np.intersect1d(pos, cands, assume_unique=True)
        if len(g) == 0:
            if n_oop == len(pos):
                reasons["positives_only_out_of_catalog"] += 1
            else:
                reasons["positives_not_in_candidates"] += 1
            continue
        if len(cands) == 0:  # unreachable while GT ⊆ Candidates, kept as a defensive check
            reasons["no_candidates"] += 1
            continue
        gt[u] = g
        eligible.append(u)
        zero_history_eligible += int(prefix.history_size(u) == 0)
    pos_counts = np.isin(prefix.catalog, np.unique(prefix.positive_events[:, MOVIE]))
    diag = {
        "stage": name,
        "prefix_events": int(len(prefix.events)),
        "prefix_positive_events": int(len(prefix.positive_events)),
        "prefix_users": int(len(prefix.users)),
        "catalog_size": int(len(prefix.catalog)),
        "catalog_items_without_positive": int((~pos_counts).sum()),
        "heldout_events": int(len(heldout)),
        "heldout_positive_events": int(len(pos_heldout)),
        "heldout_nonpositive_events": int(len(heldout) - len(pos_heldout)),
        "heldout_users": int(len(heldout_users)),
        "eligible_users_count": len(eligible),
        "excluded_users_count": int(sum(reasons.values())),
        "exclusion_reasons": reasons,
        "out_of_catalog_positive_count": oop_events,
        "users_with_out_of_catalog_positives": oop_users,
        "ground_truth_positive_count": int(sum(len(v) for v in gt.values())),
        "eligible_users_zero_prefix_history": zero_history_eligible,
    }
    return Stage(name, prefix, heldout, np.array(eligible, dtype=np.int64), gt, diag)


def validation_stage(split: Split) -> Stage:
    """Validation snapshot: prefix = TRAIN only; held-out = VALIDATION."""
    return build_stage("validation", split.train, split.validation)


def build_test_stage(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> Stage:
    """Test snapshot: prefix = TRAIN+VALIDATION; held-out = TEST.

    Used by the benchmark workflow and by synthetic tests; `evaluate` never calls it.
    """
    return build_stage("test", np.concatenate([train, validation]), test)

"""Global temporal split by timestamp quantile cutoffs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .data import TS, canonical_sort
from .errors import SplitError


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    t_val: int
    t_test: int
    n: int
    idx_val: int
    idx_test: int


def compute_cutoffs(sorted_events: np.ndarray) -> tuple[int, int, int, int]:
    """Return (idx_val, idx_test, T_val, T_test) using 0-based floor(0.8N)/floor(0.9N) indices.

    ``sorted_events`` must be in canonical (timestamp, user_id, movie_id) order.
    Integer arithmetic gives exact floor(0.80*N) and floor(0.90*N).
    """
    n = len(sorted_events)
    if n == 0:
        raise SplitError("UNSPLITTABLE_CUTOFFS", "no events to split")
    idx_val = (8 * n) // 10
    idx_test = (9 * n) // 10
    t_val = int(sorted_events[idx_val, TS])
    t_test = int(sorted_events[idx_test, TS])
    if not t_val < t_test:
        raise SplitError(
            "UNSPLITTABLE_CUTOFFS",
            f"T_val={t_val} (index {idx_val}) is not < T_test={t_test} (index {idx_test}); "
            "no fallback split is permitted",
        )
    return idx_val, idx_test, t_val, t_test


def partition_by_cutoffs(events: np.ndarray, t_val: int, t_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition by timestamp alone: TRAIN t<T_val, VALIDATION T_val<=t<T_test, TEST t>=T_test."""
    ev = canonical_sort(events)
    t = ev[:, TS]
    return ev[t < t_val], ev[(t >= t_val) & (t < t_test)], ev[t >= t_test]


def temporal_split(events: np.ndarray) -> Split:
    ev = canonical_sort(events)
    idx_val, idx_test, t_val, t_test = compute_cutoffs(ev)
    tr, va, te = partition_by_cutoffs(ev, t_val, t_test)
    assert len(tr) + len(va) + len(te) == len(ev)
    return Split(tr, va, te, t_val, t_test, len(ev), idx_val, idx_test)


def events_fingerprint(events: np.ndarray) -> str:
    """SHA256 of canonical little-endian int64 (user, movie, rating, ts) rows in canonical order."""
    arr = np.ascontiguousarray(canonical_sort(events), dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def ids_fingerprint(ids: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.sort(np.asarray(ids)), dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()

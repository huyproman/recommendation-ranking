"""Latent-factor models: implicit ALS (Hu-Koren-Volinsky) and BPR-MF, plus a ``Ranker`` adapter.

Both models are pure-NumPy CPU implementations fitted on the positive (rating >= 4) events of a
single :class:`~personalized_recommender.stage.Prefix`. Mappings are derived from that prefix
only:

* items: every item of the prefix catalog ``C_s`` in ascending numeric movie id (including items
  with zero positives, which therefore always get a factor row);
* users: every user with >= 1 positive prefix event, in ascending numeric user id. Users without a
  positive prefix event have no factor and receive the exact stage Popularity ranking.

Scores are ``dot(user_factor[u], item_factor[i])``; ranking is ``score DESC, movie_id ASC`` over
the full candidate set supplied by the evaluator (seen-item masking stays in ``Stage``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from .baselines import Popularity
from .data import MOVIE, USER
from .errors import ModelIntegrityError
from .stage import Prefix

BASE_SEED = 42


@dataclass(frozen=True)
class PositiveMatrix:
    """Binary positive user x item matrix in CSR-like form, built from one prefix only."""

    user_ids: np.ndarray  # raw ids of users with >=1 positive (ascending)
    item_ids: np.ndarray  # raw ids of ALL prefix catalog items (ascending)
    indptr: np.ndarray  # per-user offsets into indices
    indices: np.ndarray  # item indices of positives, sorted per user

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def items_of(self, uidx: int) -> np.ndarray:
        return self.indices[self.indptr[uidx] : self.indptr[uidx + 1]]

    def transpose_lists(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-item user lists (indptr, indices) for the item-side ALS step."""
        rows = np.repeat(np.arange(self.n_users), np.diff(self.indptr))
        order = np.lexsort((rows, self.indices))
        counts = np.bincount(self.indices, minlength=self.n_items)
        return np.concatenate([[0], np.cumsum(counts)]), rows[order]


def build_positive_matrix(prefix: Prefix) -> PositiveMatrix:
    pe = prefix.positive_events  # ratings >= 4 only; rating magnitude is discarded (binary)
    item_ids = prefix.catalog.astype(np.int64)
    user_ids = np.unique(pe[:, USER]).astype(np.int64)
    u = np.searchsorted(user_ids, pe[:, USER])
    i = np.searchsorted(item_ids, pe[:, MOVIE])
    order = np.lexsort((i, u))
    u, i = u[order], i[order]
    counts = np.bincount(u, minlength=len(user_ids))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return PositiveMatrix(user_ids, item_ids, indptr, i.astype(np.int64))


def check_factors(name: str, factors: np.ndarray, rows: int, k: int) -> None:
    if factors.shape != (rows, k):
        raise ModelIntegrityError("INVALID_FACTOR_SHAPE", f"{name} shape {factors.shape} != {(rows, k)}")
    if not np.all(np.isfinite(factors)):
        raise ModelIntegrityError("NONFINITE_FACTORS", f"{name} contains NaN/Inf")


# ----------------------------------------------------------------------------- ALS


def _als_half_step(fixed: np.ndarray, indptr: np.ndarray, indices: np.ndarray, alpha: float, reg: float) -> np.ndarray:
    """Exact HKV solve for every row given the fixed side.

    For row x with positive set S (p=1, c=1+alpha) and all other columns (p=0, c=1):
        (F^T F + alpha * F_S^T F_S + reg * I) x = (1 + alpha) * sum_{s in S} f_s
    which is the closed form of argmin sum_all c (p - x.f)^2 + reg ||x||^2.
    """
    k = fixed.shape[1]
    gram = fixed.T @ fixed + reg * np.eye(k)
    n = len(indptr) - 1
    out = np.zeros((n, k), dtype=np.float64)
    for r in range(n):
        s = indices[indptr[r] : indptr[r + 1]]
        if len(s) == 0:
            continue  # b = 0 -> x = 0 exactly (A is positive definite)
        fs = fixed[s]
        a = gram + alpha * (fs.T @ fs)
        out[r] = np.linalg.solve(a, (1.0 + alpha) * fs.sum(axis=0))
    return out


def als_loss(x: np.ndarray, y: np.ndarray, m: PositiveMatrix, alpha: float, reg: float) -> float:
    """Full HKV objective sum_{u,i} c_ui (p_ui - x_u.y_i)^2 + reg(||X||^2 + ||Y||^2) (dense; small data/tests)."""
    s = x @ y.T
    p = np.zeros_like(s)
    for r in range(m.n_users):
        p[r, m.items_of(r)] = 1.0
    c = 1.0 + alpha * p
    return float((c * (p - s) ** 2).sum() + reg * ((x**2).sum() + (y**2).sum()))


@dataclass
class ALSConfig:
    config_id: str
    factors: int
    alpha: float
    regularization: float = 0.05
    iterations: int = 15
    seed: int = BASE_SEED

    def as_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "family": "als",
            "factors": self.factors,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "seed": self.seed,
        }


def fit_als(m: PositiveMatrix, cfg: ALSConfig, track_loss: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(cfg.seed)
    k = cfg.factors
    x = rng.standard_normal((m.n_users, k)) * 0.01
    y = rng.standard_normal((m.n_items, k)) * 0.01
    it_ptr, it_idx = m.transpose_lists()
    losses = []
    with threadpool_limits(limits=1):
        for _ in range(cfg.iterations):
            x = _als_half_step(y, m.indptr, m.indices, cfg.alpha, cfg.regularization)
            y = _als_half_step(x, it_ptr, it_idx, cfg.alpha, cfg.regularization)
            if track_loss:
                losses.append(als_loss(x, y, m, cfg.alpha, cfg.regularization))
    return x, y, {"loss_trace": losses}


# ----------------------------------------------------------------------------- BPR


@dataclass
class BPRConfig:
    config_id: str
    factors: int
    learning_rate: float
    regularization: float = 0.01
    iterations: int = 15
    seed: int = BASE_SEED
    batch_size: int = 1024

    def as_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "family": "bpr",
            "factors": self.factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }


def sample_contrasts(rng: np.random.Generator, users: np.ndarray, m: PositiveMatrix, pos_keys: np.ndarray) -> np.ndarray:
    """Uniform contrast items from the SAME prefix catalog, excluding each user's positive items.

    Rejection sampling: draw j ~ U{0..n_items-1}; redraw while (u, j) is a prefix positive.
    Observed non-positive (1..3) items and unobserved items are both admissible contrasts.
    """
    n = m.n_items
    j = rng.integers(0, n, size=len(users))
    while True:
        keys = users * n + j
        pos = np.searchsorted(pos_keys, keys)
        bad = (pos < len(pos_keys)) & (pos_keys[np.minimum(pos, len(pos_keys) - 1)] == keys)
        if not bad.any():
            return j
        j[bad] = rng.integers(0, n, size=int(bad.sum()))


def bpr_epoch_triples(rng: np.random.Generator, m: PositiveMatrix, pos_keys: np.ndarray, eligible: np.ndarray):
    """One epoch: every (u, i_positive) pair exactly once in random order, each with a fresh contrast j."""
    rows = np.repeat(np.arange(m.n_users), np.diff(m.indptr))
    keep = eligible[rows]
    u = rows[keep]
    i = m.indices[keep]
    perm = rng.permutation(len(u))
    u, i = u[perm], i[perm]
    j = sample_contrasts(rng, u, m, pos_keys)
    return u, i, j


def fit_bpr(m: PositiveMatrix, cfg: BPRConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minibatch SGD on the BPR-Opt objective  sum ln sigma(x_u.(y_i - y_j)) - reg (||x_u||^2+||y_i||^2+||y_j||^2)."""
    rng = np.random.default_rng(cfg.seed)
    k = cfg.factors
    x = (rng.random((m.n_users, k)) - 0.5) / k
    y = (rng.random((m.n_items, k)) - 0.5) / k
    pos_keys = np.sort(np.repeat(np.arange(m.n_users), np.diff(m.indptr)) * m.n_items + m.indices)
    # a user who liked every catalog item has no admissible contrast and is skipped by the sampler
    eligible = np.diff(m.indptr) < m.n_items
    lr, reg, bs = cfg.learning_rate, cfg.regularization, cfg.batch_size
    losses, n_triples = [], 0
    with threadpool_limits(limits=1):
        for _ in range(cfg.iterations):
            u, i, j = bpr_epoch_triples(rng, m, pos_keys, eligible)
            n_triples += len(u)
            total = 0.0
            for s in range(0, len(u), bs):
                bu, bi, bj = u[s : s + bs], i[s : s + bs], j[s : s + bs]
                xu, yi, yj = x[bu], y[bi], y[bj]
                d = np.einsum("bk,bk->b", xu, yi - yj)
                total += float(np.logaddexp(0.0, -d).sum())  # -ln sigma(d)
                g = 1.0 / (1.0 + np.exp(d))  # sigma(-d) = d/dd ln sigma(d)
                np.add.at(x, bu, lr * (g[:, None] * (yi - yj) - reg * xu))
                np.add.at(y, bi, lr * (g[:, None] * xu - reg * yi))
                np.add.at(y, bj, lr * (-g[:, None] * xu - reg * yj))
            losses.append(total / max(1, len(u)))
    return x, y, {"loss_trace": losses, "triples_per_epoch": int(n_triples // max(1, cfg.iterations)),
                  "users_without_admissible_contrast": int((~eligible).sum())}


# ----------------------------------------------------------------------------- Ranker adapter


class LatentFactorRanker:
    """``Ranker`` for ALS / BPR. ``fit`` always builds a fresh mapping and model from ``prefix``."""

    def __init__(self, config: ALSConfig | BPRConfig):
        self.config = config
        self.name = "als" if isinstance(config, ALSConfig) else "bpr"
        self.fallback_users = 0
        self.personalized_users = 0

    def fit(self, prefix: Prefix) -> "LatentFactorRanker":
        t0 = time.perf_counter()
        self.prefix = prefix
        self.pop = Popularity().fit(prefix)
        self.matrix = m = build_positive_matrix(prefix)
        if m.n_users == 0 or m.n_items == 0:
            raise ModelIntegrityError("UNVERIFIED_TRAINING_INPUT", "prefix has no positive interactions")
        if not np.array_equal(m.item_ids, prefix.catalog):
            raise ModelIntegrityError("INCONSISTENT_MAPPING", "item mapping differs from prefix catalog")
        if isinstance(self.config, ALSConfig):
            x, y, info = fit_als(m, self.config)
        else:
            x, y, info = fit_bpr(m, self.config)
        check_factors("user_factors", x, m.n_users, self.config.factors)
        check_factors("item_factors", y, m.n_items, self.config.factors)
        self.user_factors, self.item_factors, self.fit_info = x, y, info
        self.train_seconds = time.perf_counter() - t0
        return self

    def user_index(self, user_id: int) -> int | None:
        idx = int(np.searchsorted(self.matrix.user_ids, user_id))
        if idx < self.matrix.n_users and self.matrix.user_ids[idx] == user_id:
            return idx
        return None

    def score(self, user_id: int, candidates: np.ndarray) -> np.ndarray | None:
        uidx = self.user_index(user_id)
        if uidx is None:
            return None
        cidx = self.pop._index(candidates)  # raises if a candidate is outside the prefix catalog
        s = self.item_factors[cidx] @ self.user_factors[uidx]
        if not np.all(np.isfinite(s)):
            raise ModelIntegrityError("NONFINITE_SCORE", f"{self.name} produced NaN/Inf scores for user {user_id}")
        return s

    def rank(self, user_id: int, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        s = self.score(user_id, candidates)
        if s is None:  # no positive prefix history -> no trained factor -> exact Popularity
            self.fallback_users += 1
            return self.pop.rank(user_id, candidates)
        self.personalized_users += 1
        return candidates[np.lexsort((candidates, -s))]  # score DESC, numeric movie_id ASC


ALS_GRID = (
    ALSConfig("ALS-01", factors=32, alpha=10.0),
    ALSConfig("ALS-02", factors=32, alpha=40.0),
    ALSConfig("ALS-03", factors=64, alpha=10.0),
    ALSConfig("ALS-04", factors=64, alpha=40.0),
)
BPR_GRID = (
    BPRConfig("BPR-01", factors=32, learning_rate=0.01),
    BPRConfig("BPR-02", factors=32, learning_rate=0.05),
    BPRConfig("BPR-03", factors=64, learning_rate=0.01),
    BPRConfig("BPR-04", factors=64, learning_rate=0.05),
)


def config_from_dict(d: dict) -> ALSConfig | BPRConfig:
    """Rebuild a config and require it to be exactly one of the fixed grid entries."""
    for c in ALS_GRID + BPR_GRID:
        if c.as_dict() == d:
            return c
    raise ValueError(f"configuration not in fixed validation grid: {d}")

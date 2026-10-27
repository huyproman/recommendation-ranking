"""Popularity and binary-cosine Item-KNN baselines.

Both are fitted exclusively on a :class:`Prefix` (the stage's allowed historical prefix) and use
only its positive observed events. Internal item indices are positions in the prefix catalog
sorted by ascending numeric movie id (deterministic, snapshot-derived).
"""

from __future__ import annotations

import numpy as np

from .data import MOVIE, USER
from .stage import Prefix

KNN_NEIGHBORS = 50


class Popularity:
    name = "popularity"

    def fit(self, prefix: Prefix) -> "Popularity":
        self.catalog = prefix.catalog
        idx = np.searchsorted(self.catalog, prefix.positive_events[:, MOVIE])
        # pop(i) = number of positive observed prefix events; 0 for items with no positive
        self.counts = np.bincount(idx, minlength=len(self.catalog)).astype(np.int64)
        return self

    def _index(self, movie_ids: np.ndarray) -> np.ndarray:
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        idx = np.searchsorted(self.catalog, movie_ids)
        if len(movie_ids) and (idx.max() >= len(self.catalog) or np.any(self.catalog[idx] != movie_ids)):
            raise ValueError("candidate outside the fitted prefix catalog")
        return idx

    def pop_of(self, movie_ids: np.ndarray) -> np.ndarray:
        return self.counts[self._index(movie_ids)]

    def score(self, user_id: int, candidates: np.ndarray) -> np.ndarray:
        return self.pop_of(candidates).astype(np.float64)

    def rank(self, user_id: int, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        return candidates[np.lexsort((candidates, -self.pop_of(candidates)))]  # pop DESC, id ASC


class ItemKNN:
    name = "item_knn"

    def __init__(self, n_neighbors: int = KNN_NEIGHBORS):
        self.n_neighbors = n_neighbors
        self.fallback_users = 0
        self.personalized_users = 0

    def fit(self, prefix: Prefix) -> "ItemKNN":
        self.pop = Popularity().fit(prefix)
        self.prefix = prefix
        self.catalog = ids = prefix.catalog
        n_items = len(ids)
        pe = prefix.positive_events
        users = np.unique(pe[:, USER])
        x = np.zeros((len(users), n_items), dtype=np.float64)  # binary positive matrix
        x[np.searchsorted(users, pe[:, USER]), np.searchsorted(ids, pe[:, MOVIE])] = 1.0
        deg = x.sum(axis=0)  # |UsersPositive(i)|, exact integers
        co = x.T @ x  # |U_i ∩ U_j|, exact integers (< 2^53) in float64
        del x
        np.fill_diagonal(co, 0.0)  # sim(i,i) = 0
        self.degree = deg.astype(np.int64)
        w = np.zeros((n_items, n_items), dtype=np.float64)  # row j: sim(i,j) for i in N50(j), else 0
        self.neighbor_lists: dict[int, list[tuple[int, float]]] = {}
        for j in range(n_items):
            c = co[j]
            nz = np.nonzero(c > 0)[0]  # strictly positive similarity; zero-degree items never appear
            if len(nz) == 0:
                self.neighbor_lists[int(ids[j])] = []
                continue
            # sim^2 = c^2 / (d_i d_j): one correctly-rounded division of exact integers, so
            # mathematically equal similarities are bit-identical and distinct ones keep their order.
            sim = np.sqrt((c[nz] * c[nz]) / (deg[nz] * deg[j]))
            o = np.lexsort((ids[nz], -sim))[: self.n_neighbors]  # sim DESC, movie_id ASC
            w[j, nz[o]] = sim[o]
            self.neighbor_lists[int(ids[j])] = [(int(ids[nz[t]]), float(sim[t])) for t in o]
        del co
        self._w = w
        return self

    def similarity(self, a: int, b: int) -> float:
        """Untruncated binary cosine sim(a,b) recomputed from prefix positive user sets."""
        if a == b:
            return 0.0
        pe = self.prefix.positive_events
        ua = set(pe[pe[:, MOVIE] == a][:, USER].tolist())
        ub = set(pe[pe[:, MOVIE] == b][:, USER].tolist())
        if not ua or not ub:
            return 0.0
        return len(ua & ub) / float(np.sqrt(len(ua) * len(ub)))

    def score(self, user_id: int, candidates: np.ndarray) -> np.ndarray:
        """s(u,i) = sum over j in Pos(u) with i in N50(j) of sim(i,j)."""
        cidx = self.pop._index(candidates)
        pos = self.prefix.positives(user_id)
        if len(pos) == 0:
            return np.zeros(len(cidx), dtype=np.float64)
        seeds = np.searchsorted(self.catalog, pos)  # ascending movie id -> fixed summation order
        return self._w[seeds].sum(axis=0)[cidx]

    def rank(self, user_id: int, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        s = self.score(user_id, candidates)
        if len(self.prefix.positives(user_id)) == 0 or not np.any(s > 0):
            self.fallback_users += 1
            return self.pop.rank(user_id, candidates)  # exact Popularity fallback
        self.personalized_users += 1
        # personalized score DESC, popularity DESC, movie_id ASC
        return candidates[np.lexsort((candidates, -self.pop.pop_of(candidates), -s))]


MODELS = {"popularity": Popularity, "item_knn": ItemKNN}

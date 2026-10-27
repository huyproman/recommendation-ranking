"""Positive matrix/mappings, REAL ALS and BPR fitting, fallback/ranking integration."""

import math

import numpy as np
import pytest
from conftest import ev

from personalized_recommender.baselines import Popularity
from personalized_recommender.errors import ModelIntegrityError
from personalized_recommender.evaluator import evaluate
from personalized_recommender.models import (
    ALS_GRID,
    BPR_GRID,
    ALSConfig,
    BPRConfig,
    LatentFactorRanker,
    PositiveMatrix,
    _als_half_step,
    als_loss,
    build_positive_matrix,
    bpr_epoch_triples,
    fit_als,
    fit_bpr,
)
from personalized_recommender.split import partition_by_cutoffs
from personalized_recommender.stage import Prefix, build_stage, build_test_stage

# Two taste clusters: users 1-4 like movies 10,20,30 ; users 5-8 like 40,50,60. Item 70 only low ratings.
CLUSTER = ev(
    *[(u, m, 5, u * 10 + k) for u in (1, 2, 3, 4) for k, m in enumerate((10, 20, 30)) if not (u == 4 and m == 30)],
    *[(u, m, 4, u * 10 + k) for u in (5, 6, 7, 8) for k, m in enumerate((40, 50, 60)) if not (u == 8 and m == 60)],
    (1, 70, 1, 200), (5, 70, 2, 201), (9, 70, 3, 202),  # user 9: only a low rating
)

SMALL_ALS = ALSConfig("T-ALS", factors=4, alpha=10.0, iterations=15)
SMALL_BPR = BPRConfig("T-BPR", factors=4, learning_rate=0.05, iterations=200, batch_size=4)

# ---------------- positive matrix and mappings ----------------


def test_positive_matrix_mapping():
    pr = Prefix(CLUSTER)
    m = build_positive_matrix(pr)
    assert m.item_ids.tolist() == [10, 20, 30, 40, 50, 60, 70]  # every catalog item incl. zero-positive 70
    assert m.user_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]  # user 9 (low-rated only) has no factor row
    assert [m.item_ids[m.items_of(r)].tolist() for r in range(m.n_users)][3] == [10, 20]
    assert all(70 not in m.item_ids[m.items_of(r)].tolist() for r in range(m.n_users))  # low ratings not targets
    assert pr.seen(1).tolist() == [10, 20, 30, 70]  # low rating still in seen history
    # stable: input order irrelevant
    m2 = build_positive_matrix(Prefix(CLUSTER[::-1]))
    assert np.array_equal(m.indices, m2.indices) and np.array_equal(m.indptr, m2.indptr)


def test_train_only_mapping_excludes_future():
    full = np.concatenate([CLUSTER, ev((1, 99, 5, 500), (10, 10, 5, 501))])
    tr, va, te = partition_by_cutoffs(full, 400, 600)
    m = build_positive_matrix(build_stage("validation", tr, va).prefix)
    assert 99 not in m.item_ids.tolist() and 10 not in m.user_ids.tolist()
    r = LatentFactorRanker(SMALL_ALS).fit(build_stage("validation", tr, va).prefix)
    assert r.item_factors.shape == (7, 4) and r.user_factors.shape == (8, 4)


# ---------------- implicit ALS ----------------


def test_als_half_step_matches_dense_hkv_solution():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((6, 3))
    indptr = np.array([0, 2, 2, 5])
    indices = np.array([0, 3, 1, 2, 5])
    alpha, reg = 40.0, 0.05
    got = _als_half_step(y, indptr, indices, alpha, reg)
    for r in range(3):
        p = np.zeros(6)
        p[indices[indptr[r] : indptr[r + 1]]] = 1.0
        c = 1.0 + alpha * p  # c=1+alpha at positives, 1 elsewhere
        expected = np.linalg.solve(y.T @ np.diag(c) @ y + reg * np.eye(3), y.T @ (c * p))
        assert np.allclose(got[r], expected, atol=1e-10)
    assert np.all(got[1] == 0)  # no positives -> exact zero vector


def test_als_alpha_zero_is_unweighted_ridge():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((5, 2))
    got = _als_half_step(y, np.array([0, 2]), np.array([1, 4]), 0.0, 0.1)
    p = np.array([0, 1, 0, 0, 1.0])
    assert np.allclose(got[0], np.linalg.solve(y.T @ y + 0.1 * np.eye(2), y.T @ p))


def test_als_real_fit_monotone_and_learns_clusters():
    pr = Prefix(CLUSTER)
    m = build_positive_matrix(pr)
    x, y, info = fit_als(m, SMALL_ALS, track_loss=True)
    lt = info["loss_trace"]
    assert all(b <= a + 1e-9 for a, b in zip(lt, lt[1:]))  # exact alternating minimisation
    assert lt[-1] < als_loss(np.zeros_like(x), np.zeros_like(y), m, 10.0, 0.05)
    r = LatentFactorRanker(SMALL_ALS).fit(pr)
    # user 4 (cluster A, missing 30) and user 8 (cluster B, missing 60)
    assert r.rank(4, pr.candidates(4)).tolist()[0] == 30
    assert r.rank(8, pr.candidates(8)).tolist()[0] == 60
    s = r.score(4, np.array([30, 60]))
    assert np.all(np.isfinite(s)) and s[0] > s[1]


# ---------------- BPR-MF ----------------


def test_contrast_sampling_oracle():
    pr = Prefix(CLUSTER)
    m = build_positive_matrix(pr)
    pos_keys = np.sort(np.repeat(np.arange(m.n_users), np.diff(m.indptr)) * m.n_items + m.indices)
    rng = np.random.default_rng(42)
    seen_70 = 0
    for _ in range(200):
        u, i, j = bpr_epoch_triples(rng, m, pos_keys, np.ones(m.n_users, bool))
        assert len(u) == len(m.indices)  # every positive pair once per epoch
        for uu, ii, jj in zip(u, i, j):
            pos = set(m.items_of(uu).tolist())
            assert ii in pos and jj not in pos  # positive i; contrast j never a user positive
            assert 0 <= jj < m.n_items  # only prefix catalog (no future-only items exist in mapping)
        seen_70 += int(((u == 0) & (j == 6)).sum())  # user 1 rated item 70 low (observed non-positive)
    assert seen_70 > 0  # observed 1..3 items are admissible contrast samples (not "verified negatives")
    # seeded determinism
    a = bpr_epoch_triples(np.random.default_rng(7), m, pos_keys, np.ones(m.n_users, bool))
    b = bpr_epoch_triples(np.random.default_rng(7), m, pos_keys, np.ones(m.n_users, bool))
    assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_single_sgd_step_matches_hand_gradient():
    # 1 user, 2 items, positive item 0 -> contrast must be item 1
    m = PositiveMatrix(np.array([1]), np.array([5, 6]), np.array([0, 1]), np.array([0]))
    cfg = BPRConfig("T", factors=2, learning_rate=0.1, regularization=0.01, iterations=1, batch_size=1)
    rng = np.random.default_rng(cfg.seed)
    x0 = (rng.random((1, 2)) - 0.5) / 2
    y0 = (rng.random((2, 2)) - 0.5) / 2
    x, y, info = fit_bpr(m, cfg)
    d = x0[0] @ (y0[0] - y0[1])
    g = 1 / (1 + math.exp(d))
    assert np.allclose(x[0], x0[0] + 0.1 * (g * (y0[0] - y0[1]) - 0.01 * x0[0]))
    assert np.allclose(y[0], y0[0] + 0.1 * (g * x0[0] - 0.01 * y0[0]))
    assert np.allclose(y[1], y0[1] + 0.1 * (-g * x0[0] - 0.01 * y0[1]))
    assert info["loss_trace"][0] == pytest.approx(math.log1p(math.exp(-d)))


def test_bpr_real_fit_optimises_and_learns():
    pr = Prefix(CLUSTER)
    r = LatentFactorRanker(SMALL_BPR).fit(pr)
    lt = r.fit_info["loss_trace"]
    assert lt[-1] < lt[0] * 0.5  # pairwise logistic loss actually optimised
    assert r.rank(4, pr.candidates(4)).tolist()[0] == 30
    assert r.rank(8, pr.candidates(8)).tolist()[0] == 60


# ---------------- fallback and ranking integration ----------------


@pytest.mark.parametrize("cfg", [SMALL_ALS, SMALL_BPR])
def test_fallback_mask_full_catalog(cfg):
    tr = CLUSTER
    va = ev((9, 10, 5, 300), (4, 30, 5, 301), (11, 20, 4, 302))
    st = build_stage("validation", tr, va)
    r = LatentFactorRanker(cfg).fit(st.prefix)
    pop = Popularity().fit(st.prefix)
    for u in (9, 11):  # low-rated-only user, unknown user -> exact Popularity
        assert r.rank(u, st.candidates(u)).tolist() == pop.rank(u, st.candidates(u)).tolist()
    assert 70 not in r.rank(9, st.candidates(9)).tolist()  # low-rated seen item stays masked
    r.fallback_users = r.personalized_users = 0
    res = evaluate(st, r)
    assert r.fallback_users == 2 and r.personalized_users == 1
    assert res["ranked_candidate_pairs"] == sum(len(st.candidates(u)) for u in st.eligible_users.tolist())


def test_negative_scores_not_fallback_and_numeric_ties():
    pr = Prefix(CLUSTER)
    r = LatentFactorRanker(SMALL_ALS).fit(pr)
    r.user_factors = r.user_factors.copy()
    r.item_factors = np.zeros_like(r.item_factors)
    r.item_factors[:, 0] = -1.0  # every score equal and negative
    r.user_factors[:, 0] = 1.0
    c = pr.candidates(4)
    out = r.rank(4, c).tolist()
    assert out == sorted(c.tolist())  # ties -> numeric movie_id ASC
    assert r.fallback_users == 0  # negative scores are legitimate, no fallback


def test_nonfinite_rejected():
    pr = Prefix(CLUSTER)
    r = LatentFactorRanker(SMALL_ALS).fit(pr)
    r.item_factors = r.item_factors.copy()
    r.item_factors[0, 0] = np.nan  # item 10 is a candidate for user 5
    assert 10 in pr.candidates(5).tolist()
    with pytest.raises(ModelIntegrityError) as e:
        r.rank(5, pr.candidates(5))
    assert e.value.code == "NONFINITE_SCORE" and e.value.exit_code == 1


def test_nonfinite_factors_rejected(monkeypatch):
    import personalized_recommender.models as M

    monkeypatch.setattr(M, "fit_als", lambda m, c: (np.full((m.n_users, c.factors), np.inf), np.zeros((m.n_items, c.factors)), {"loss_trace": []}))
    with pytest.raises(ModelIntegrityError) as e:
        LatentFactorRanker(SMALL_ALS).fit(Prefix(CLUSTER))
    assert e.value.code == "NONFINITE_FACTORS"
    monkeypatch.setattr(M, "fit_als", lambda m, c: (np.zeros((m.n_users, c.factors + 1)), np.zeros((m.n_items, c.factors)), {"loss_trace": []}))
    with pytest.raises(ModelIntegrityError) as e:
        LatentFactorRanker(SMALL_ALS).fit(Prefix(CLUSTER))
    assert e.value.code == "INVALID_FACTOR_SHAPE"


def test_frozen_grid_exact():
    assert [(c.config_id, c.factors, c.alpha, c.regularization, c.iterations, c.seed) for c in ALS_GRID] == [
        ("ALS-01", 32, 10.0, 0.05, 15, 42), ("ALS-02", 32, 40.0, 0.05, 15, 42),
        ("ALS-03", 64, 10.0, 0.05, 15, 42), ("ALS-04", 64, 40.0, 0.05, 15, 42)]
    assert [(c.config_id, c.factors, c.learning_rate, c.regularization, c.iterations, c.seed) for c in BPR_GRID] == [
        ("BPR-01", 32, 0.01, 0.01, 15, 42), ("BPR-02", 32, 0.05, 0.01, 15, 42),
        ("BPR-03", 64, 0.01, 0.01, 15, 42), ("BPR-04", 64, 0.05, 0.01, 15, 42)]


# ---------------- real ML-1M TRAIN fits (official data) ----------------


def test_real_ml1m_als_bpr_train_fit(ml1m_dir):
    from personalized_recommender.data import load_dataset
    from personalized_recommender.split import temporal_split

    s = temporal_split(load_dataset(ml1m_dir)[0])
    st = build_stage("validation", s.train, s.validation)
    for cfg in (ALS_GRID[0], BPR_GRID[0]):
        r = LatentFactorRanker(cfg).fit(st.prefix)
        assert r.item_factors.shape == (3662, 32) and r.user_factors.shape == (len(np.unique(st.prefix.positive_events[:, 0])), 32)
        assert np.all(np.isfinite(r.user_factors)) and np.all(np.isfinite(r.item_factors))
        assert np.abs(r.item_factors).max() > 1e-3  # genuinely learned, not zero/static
        res = evaluate(st, r)
        assert r.fallback_users == 611 and r.personalized_users == 492
        assert 0 < res["metrics_by_k"]["10"]["ndcg"] < 1
    # TEST-stage fit uses TRAIN+VALIDATION mapping only
    ts = build_test_stage(s.train, s.validation, s.test)
    r = LatentFactorRanker(ALS_GRID[0]).fit(ts.prefix)
    assert np.array_equal(r.matrix.item_ids, np.unique(np.concatenate([s.train, s.validation])[:, 1]))

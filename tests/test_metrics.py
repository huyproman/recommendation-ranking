"""Analytical metric oracles and full-catalog / ties / short lists / denominators.

Oracle values are hand-derived constants written out explicitly, not recomputed through the
implementation's helper functions.
"""

import math

import numpy as np
import pytest
from conftest import ev

from personalized_recommender import metrics as M
from personalized_recommender.errors import EvaluationError
from personalized_recommender.evaluator import evaluate, rank_by_score
from personalized_recommender.stage import build_stage

L3 = 1.5849625007211562  # log2(3)
R = [3, 1, 4, 2]
GT = {1, 4}

# ---------------- metric oracles ----------------


def test_k2_oracle():
    assert M.recall_at_k(R, GT, 2) == 0.5
    assert M.hitrate_at_k(R, GT, 2) == 1.0
    assert M.dcg_at_k(R, GT, 2) == pytest.approx(0.6309297535714575, abs=1e-15)  # 1/log2(3)
    assert M.idcg_at_k(2, 2) == pytest.approx(1.6309297535714575, abs=1e-15)  # 1 + 1/log2(3)
    assert M.ndcg_at_k(R, GT, 2) == pytest.approx(0.38685280723454163, abs=1e-15)
    assert M.ndcg_at_k(R, GT, 2) == pytest.approx((1 / L3) / (1 + 1 / L3), abs=1e-15)


def test_k3_oracle():
    assert M.recall_at_k(R, GT, 3) == 1.0
    assert M.hitrate_at_k(R, GT, 3) == 1.0
    assert M.dcg_at_k(R, GT, 3) == pytest.approx(0.6309297535714575 + 0.5, abs=1e-15)  # 1/log2(3)+1/log2(4)
    assert M.idcg_at_k(2, 3) == pytest.approx(1.6309297535714575, abs=1e-15)  # unchanged
    assert M.ndcg_at_k(R, GT, 3) == pytest.approx(1.1309297535714575 / 1.6309297535714575, abs=1e-15)


def test_k1_and_miss():
    assert M.recall_at_k(R, GT, 1) == 0.0 and M.hitrate_at_k(R, GT, 1) == 0.0 and M.ndcg_at_k(R, GT, 1) == 0.0
    assert M.ndcg_at_k([1, 4, 3, 2], GT, 2) == pytest.approx(1.0)


def _two_user_stage():
    # Catalog {1,2,3,4,5}; u1 saw item 5 -> candidates [1,2,3,4]; u2 saw item 1 -> [2,3,4,5]
    tr = ev((1, 5, 2, 1), (2, 1, 1, 2), (3, 2, 1, 3), (3, 3, 1, 4), (3, 4, 1, 5))
    va = ev((1, 1, 5, 10), (1, 4, 4, 11), (2, 5, 4, 12))
    return build_stage("validation", tr, va)


class FixedRanker:
    name = "fixed"

    def __init__(self, order):
        self.order = order

    def fit(self, prefix):
        return self

    def rank(self, u, cands):
        return np.array([m for m in self.order[u] if m in set(cands.tolist())], dtype=np.int64)


def test_macro_average_two_users_and_coverage_union():
    st = _two_user_stage()
    assert st.eligible_users.tolist() == [1, 2]
    r = evaluate(st, FixedRanker({1: [3, 1, 4, 2], 2: [2, 3, 4, 5]}), ks=(2, 3, 5))
    m2 = r["metrics_by_k"]["2"]
    # u1: recall 0.5, hit 1, ndcg 0.386852807...; u2: GT {5} at rank 4 -> 0,0,0
    assert m2["recall"] == pytest.approx((0.5 + 0.0) / 2)
    assert m2["hitrate"] == pytest.approx(0.5)
    assert m2["ndcg"] == pytest.approx(0.38685280723454163 / 2)
    # micro-average would be 1/3 for recall@2; macro must be 0.25
    assert m2["recall"] != pytest.approx(1 / 3)
    m5 = r["metrics_by_k"]["5"]
    # K=5 > 4 candidates: u2 ndcg = (1/log2(5)) / 1
    assert m5["ndcg"] == pytest.approx(((0.6309297535714575 + 0.5) / 1.6309297535714575 + 1 / math.log2(5)) / 2)
    # coverage@2 = |{3,1} ∪ {2,3}| / |C| = 3/5 (independent set union)
    assert m2["catalog_coverage"] == pytest.approx(3 / 5)
    assert r["metrics_by_k"]["3"]["catalog_coverage"] == pytest.approx(len({3, 1, 4} | {2, 3, 4}) / 5)
    assert r["metrics_by_k"]["5"]["catalog_coverage"] == pytest.approx(1.0)


def test_coverage_denominator_is_entire_catalog():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)


# ---------------- full catalog, ties, short lists, denominators ----------------


def test_deterministic_numeric_tie_order():
    c = np.array([100, 9, 20, 3], dtype=np.int64)
    assert rank_by_score(c, np.array([1.0, 1.0, 2.0, 1.0])).tolist() == [20, 3, 9, 100]  # numeric, not "100"<"20"
    with pytest.raises(EvaluationError):
        rank_by_score(c, np.array([1.0, np.nan, 0.0, 0.0]))
    with pytest.raises(EvaluationError):
        rank_by_score(c, np.array([1.0, np.inf, 0.0, 0.0]))


class SamplingRanker:
    name = "sampler"

    def fit(self, prefix):
        return self

    def rank(self, u, cands):
        return cands[:2]  # sampled subset -> must be rejected


class ExtraRanker:
    name = "extra"

    def fit(self, prefix):
        return self

    def rank(self, u, cands):
        return np.concatenate([cands, [999]])


def test_full_catalog_required():
    st = _two_user_stage()
    for rk in (SamplingRanker(), ExtraRanker()):
        with pytest.raises(EvaluationError) as e:
            evaluate(st, rk)
        assert e.value.code == "INVALID_RANKING"


class RecordingRanker:
    name = "rec"

    def __init__(self):
        self.seen = {}

    def fit(self, prefix):
        return self

    def rank(self, u, cands):
        self.seen[u] = cands.tolist()
        return cands


def test_every_unobserved_candidate_ranked_and_short_lists():
    st = _two_user_stage()
    rr = RecordingRanker()
    r = evaluate(st, rr)
    assert rr.seen == {1: [1, 2, 3, 4], 2: [2, 3, 4, 5]}  # every catalog item not seen by user
    assert r["ranked_candidate_pairs"] == 8
    assert r["users_with_fewer_candidates_than_max_k"] == 2
    # K=20 with 4 candidates: L_K has 4 items; recall denominator |GT|, not K or sample size
    m20 = r["metrics_by_k"]["20"]
    assert m20["recall"] == pytest.approx(1.0) and m20["hitrate"] == pytest.approx(1.0)
    assert m20["catalog_coverage"] == pytest.approx(1.0)


def test_excluded_users_not_in_denominator():
    st = _two_user_stage()
    # add user 4 with only non-positive heldout -> excluded; macro must still average 2 users
    st2 = build_stage("validation", st.prefix.events, np.concatenate([st.heldout, ev((4, 2, 1, 20))]))
    a = evaluate(st, FixedRanker({1: [3, 1, 4, 2], 2: [2, 3, 4, 5]}))
    b = evaluate(st2, FixedRanker({1: [3, 1, 4, 2], 2: [2, 3, 4, 5]}))
    assert a["metrics_by_k"] == b["metrics_by_k"]
    assert st2.diagnostics["excluded_users_count"] == 1 and b["eligible_users_count"] == 2


def test_idcg_uses_min_k_gt():
    # |GT|=3, K=2 -> IDCG = 1 + 1/log2(3)
    assert M.ndcg_at_k([7, 8, 9], {7, 8, 9}, 2) == pytest.approx(1.0)
    assert M.idcg_at_k(3, 2) == pytest.approx(1.6309297535714575)

"""Popularity oracle and Item-KNN oracle (hand-verified values)."""

import math

import numpy as np
import pytest
from conftest import ev

from personalized_recommender.baselines import ItemKNN, Popularity
from personalized_recommender.split import partition_by_cutoffs
from personalized_recommender.stage import Prefix, build_stage

# ---------------- Popularity ----------------

POP_TRAIN = ev(
    (1, 30, 5, 1), (2, 30, 4, 2), (3, 30, 3, 3),  # item 30: 2 positives
    (1, 20, 4, 4), (2, 20, 5, 5), (4, 20, 1, 6),  # item 20: 2 positives (tie with 30)
    (3, 100, 5, 7), (4, 100, 5, 8), (5, 100, 4, 9),  # item 100: 3 positives
    (1, 7, 1, 10), (2, 7, 2, 11), (3, 7, 3, 12),  # item 7: nonpositive only -> pop 0
    (5, 9, 4, 13),  # item 9: 1 positive
)


def test_counts_positive_only():
    p = Popularity().fit(Prefix(POP_TRAIN))
    assert p.catalog.tolist() == [7, 9, 20, 30, 100]
    assert p.counts.tolist() == [0, 1, 2, 2, 3]


def test_ranking_ties_numeric_and_nonpositive_item_candidate():
    pr = Prefix(POP_TRAIN)
    p = Popularity().fit(pr)
    # new user 99: full catalog. 100(3), 20(2), 30(2) tie -> 20 before 30, 9(1), 7(0)
    assert p.rank(99, pr.candidates(99)).tolist() == [100, 20, 30, 9, 7]
    # user 4 saw 20 (rating 1) and 100 -> excluded although 20's rating was low
    assert p.rank(4, pr.candidates(4)).tolist() == [30, 9, 7]


def test_heldout_mutation_isolation():
    base = np.concatenate([POP_TRAIN, ev((6, 7, 5, 100), (7, 7, 5, 101), (8, 9, 5, 102))])
    tr, va, _ = partition_by_cutoffs(base, 50, 1000)
    a = Popularity().fit(build_stage("validation", tr, va).prefix)
    mut = base.copy()
    mut[mut[:, 3] >= 50, 2] = 1
    mut = np.concatenate([mut, ev((9, 7, 5, 200), (10, 7, 5, 201), (11, 7, 5, 202))])
    tr2, va2, _ = partition_by_cutoffs(mut, 50, 1000)
    b = Popularity().fit(build_stage("validation", tr2, va2).prefix)
    assert a.counts.tolist() == b.counts.tolist() == [0, 1, 2, 2, 3]
    assert a.rank(99, a.catalog).tolist() == b.rank(99, b.catalog).tolist()


# ---------------- Item-KNN ----------------

# Positive user sets (rating>=4):  item 1: {1,2,3}; item 2: {1,2}; item 3: {3,4}; item 4: {4}
# item 5: no positives (zero-degree, rated 2 by user 5); item 6: {6} (isolated)
KNN_TRAIN = ev(
    (1, 1, 5, 1), (2, 1, 4, 2), (3, 1, 5, 3),
    (1, 2, 4, 4), (2, 2, 5, 5),
    (3, 3, 4, 6), (4, 3, 5, 7),
    (4, 4, 4, 8),
    (5, 5, 2, 9), (4, 5, 3, 10),
    (6, 6, 5, 11),
    (7, 1, 2, 12),  # user 7: only a nonpositive rating
)


def test_cosine_hand_values():
    k = ItemKNN().fit(Prefix(KNN_TRAIN))
    assert k.similarity(1, 2) == pytest.approx(2 / math.sqrt(6))
    assert k.similarity(1, 3) == pytest.approx(1 / math.sqrt(6))
    assert k.similarity(3, 4) == pytest.approx(1 / math.sqrt(2))
    assert k.similarity(1, 4) == 0.0 and k.similarity(2, 3) == 0.0
    assert k.similarity(5, 1) == 0.0  # zero-degree
    assert k.similarity(1, 1) == 0.0  # self
    nl = k.neighbor_lists
    assert [m for m, _ in nl[1]] == [2, 3]
    assert nl[1][0][1] == pytest.approx(2 / math.sqrt(6)) and nl[1][1][1] == pytest.approx(1 / math.sqrt(6))
    assert [m for m, _ in nl[3]] == [4, 1]  # 0.7071 > 0.4082
    assert [m for m, _ in nl[4]] == [3]
    assert nl[5] == [] and nl[6] == []  # zero-degree / isolated: no zero-sim filling
    for j, lst in nl.items():
        assert j not in [m for m, _ in lst]  # no self
        assert all(s > 0 for _, s in lst)


def test_personalized_sum_and_all_observed_mask():
    pr = Prefix(KNN_TRAIN)
    k = ItemKNN().fit(pr)
    # user 3: positives {1,3}; seen {1,3}; candidates {2,4,5,6}
    c = pr.candidates(3)
    assert c.tolist() == [2, 4, 5, 6]
    s = dict(zip(c.tolist(), k.score(3, c).tolist()))
    assert s[2] == pytest.approx(2 / math.sqrt(6))  # from seed 1
    assert s[4] == pytest.approx(1 / math.sqrt(2))  # from seed 3
    assert s[5] == 0.0 and s[6] == 0.0
    # rank: 2 (0.8165) > 4 (0.7071) > then zero-score by pop: 6 (pop1) > 5 (pop0)
    assert k.rank(3, c).tolist() == [2, 4, 6, 5]
    # user 4: positives {3,4}, seen {3,4,5} (5 low-rated); candidates {1,2,6}
    c4 = pr.candidates(4)
    assert c4.tolist() == [1, 2, 6]
    assert k.score(4, c4)[0] == pytest.approx(1 / math.sqrt(6))  # item 1 from seed 3
    assert 5 not in k.rank(4, c4).tolist()


def test_sum_over_multiple_seeds():
    # user 9 positives {2,3}: item 1 in N50(2) (sim 2/sqrt6) and N50(3) (sim 1/sqrt6)
    pr = Prefix(np.concatenate([KNN_TRAIN, ev((9, 2, 4, 20), (9, 3, 4, 21))]))
    k = ItemKNN().fit(pr)
    # recompute with user 9 included: U1={1,2,3}, U2={1,2,9}, U3={3,4,9}
    s = k.score(9, np.array([1]))[0]
    assert s == pytest.approx(2 / 3 + 1 / 3)  # |U1∩U2|=2/sqrt(9) + |U1∩U3|=1/sqrt(9)


def test_no_signal_fallback_exact_popularity():
    pr = Prefix(KNN_TRAIN)
    k = ItemKNN().fit(pr)
    p = Popularity().fit(pr)
    # user 7: no positives -> popularity
    assert k.rank(7, pr.candidates(7)).tolist() == p.rank(7, pr.candidates(7)).tolist()
    # user 6: positive only on isolated item 6 -> all scores zero -> popularity
    assert np.all(k.score(6, pr.candidates(6)) == 0)
    assert k.rank(6, pr.candidates(6)).tolist() == p.rank(6, pr.candidates(6)).tolist()
    # brand-new user
    assert k.rank(42, pr.candidates(42)).tolist() == p.rank(42, pr.candidates(42)).tolist() == [1, 2, 3, 4, 6, 5]


def test_top50_truncation_and_tie_order():
    # seed item 1 liked by users 1..60; items 2..61 each liked by exactly one of them plus distinct
    # extra users so similarities differ: item m has degree (m-1) -> sim = 1/sqrt(60*(m-1))... use
    # simpler: all neighbors identical sim -> tie order by movie id ASC, truncated at 50.
    rows = []
    t = 1
    for u in range(1, 61):
        rows.append((u, 1, 5, t)); t += 1
        rows.append((u, 1000 + u, 5, t)); t += 1  # item 1000+u liked only by u -> sim = 1/sqrt(60)
    rows.append((100, 5000, 5, t))  # unrelated item
    k = ItemKNN().fit(Prefix(ev(*rows)))
    nl = k.neighbor_lists[1]
    assert len(nl) == 50
    assert [m for m, _ in nl] == list(range(1001, 1051))
    assert all(s == pytest.approx(1 / math.sqrt(60)) for _, s in nl)
    assert 5000 not in [m for m, _ in nl]
    # a user liking item 1 only: items 1051..1060 not in N50(1) -> score 0 -> popularity order after
    pr = Prefix(ev(*rows, (200, 1, 4, t + 1)))
    k2 = ItemKNN().fit(pr)
    sc = dict(zip(pr.candidates(200).tolist(), k2.score(200, pr.candidates(200)).tolist()))
    assert sc[1001] > 0 and sc[1060] == 0.0
    # neighbor list for item 1050 (liked by user 50 only) has only item 1
    assert [m for m, _ in k2.neighbor_lists[1050]] == [1]


def test_similarity_ordering_desc():
    # seed 1 with neighbors of distinct sims: 2 shares 3 users (deg3), 3 shares 2 (deg2), 4 shares 2 (deg 4)
    rows = [(u, 1, 5, u) for u in range(1, 6)]
    rows += [(u, 2, 5, 10 + u) for u in (1, 2, 3)]
    rows += [(u, 3, 5, 20 + u) for u in (1, 2)]
    rows += [(u, 4, 5, 30 + u) for u in (1, 2, 8, 9)]
    k = ItemKNN().fit(Prefix(ev(*rows)))
    sims = {2: 3 / math.sqrt(15), 3: 2 / math.sqrt(10), 4: 2 / math.sqrt(20)}
    assert [m for m, _ in k.neighbor_lists[1]] == sorted(sims, key=lambda m: (-sims[m], m))
    for m, s in k.neighbor_lists[1]:
        assert s == pytest.approx(sims[m])


def test_train_only_fit():
    base = np.concatenate([KNN_TRAIN, ev((1, 4, 5, 500), (2, 4, 5, 501), (3, 6, 5, 502))])
    tr, va, _ = partition_by_cutoffs(base, 100, 1000)
    k = ItemKNN().fit(build_stage("validation", tr, va).prefix)
    k0 = ItemKNN().fit(Prefix(KNN_TRAIN))
    assert np.array_equal(k._w, k0._w)
    assert k.similarity(1, 4) == 0.0  # validation co-likes not used


# ---------------- real ML-1M independent brute-force cross-check ----------------


def test_real_ml1m_baselines_bruteforce(ml1m_dir):
    """Popularity counts and Item-KNN N50 lists on the real TRAIN prefix vs. exact-rational brute force."""
    from collections import Counter, defaultdict
    from fractions import Fraction

    from personalized_recommender.data import load_dataset
    from personalized_recommender.split import temporal_split

    events, _ = load_dataset(ml1m_dir)
    s = temporal_split(events)
    pr = build_stage("validation", s.train, s.validation).prefix
    pop = Popularity().fit(pr)
    cnt = Counter(int(m) for u, m, r, t in s.train.tolist() if r >= 4)
    assert {int(m): int(c) for m, c in zip(pop.catalog, pop.counts) if c} == dict(cnt)
    assert set(pr.catalog.tolist()) == {int(m) for m in s.train[:, 1]}
    assert pr.catalog.max() < 2**31 and len(pr.catalog) == len(np.unique(s.train[:, 1]))
    knn = ItemKNN().fit(pr)
    users = defaultdict(set)
    for u, m, r, t in s.train.tolist():
        if r >= 4:
            users[m].add(u)
    seeds = [int(m) for m in pr.catalog[:: max(1, len(pr.catalog) // 25)]] + [2858, 260]
    for j in seeds:
        uj = users.get(j, set())
        cand = []
        for i, ui in users.items():
            if i == j or not uj:
                continue
            c = len(ui & uj)
            if c > 0:
                cand.append((-Fraction(c * c, len(ui) * len(uj)), i))
        cand.sort()
        expected = [i for _, i in cand[:50]]
        got = [m for m, _ in knn.neighbor_lists[j]]
        assert got == expected, j
        for (m, sim), (f, _) in zip(knn.neighbor_lists[j], cand[:50]):
            assert sim == pytest.approx(math.sqrt(-f), rel=1e-12)

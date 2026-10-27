"""Adversarial leakage: future-event mutations cannot affect a fixed historical snapshot.

Cutoffs are held fixed (partition_by_cutoffs with explicit T_val/T_test) so the tests measure
feature/catalog leakage rather than a different quantile split.
"""

import numpy as np
from conftest import ev

from personalized_recommender.baselines import ItemKNN, Popularity
from personalized_recommender.evaluator import evaluate
from personalized_recommender.split import partition_by_cutoffs
from personalized_recommender.stage import build_stage, build_test_stage

T_VAL, T_TEST = 100, 200

BASE = ev(
    (1, 1, 5, 10), (1, 2, 4, 11), (1, 3, 2, 12),
    (2, 1, 4, 13), (2, 2, 5, 14), (2, 4, 4, 15),
    (3, 2, 5, 16), (3, 4, 4, 17), (3, 5, 1, 18),
    (4, 1, 4, 19), (4, 5, 5, 20),
    # validation
    (1, 4, 5, 120), (2, 5, 4, 130), (3, 1, 5, 140), (5, 2, 4, 150), (4, 3, 2, 160),
    # test
    (1, 5, 4, 210), (5, 1, 5, 220), (6, 9, 5, 230),
)

FUTURE_MUTATIONS = ev(
    (1, 6, 5, 150),  # validation event on a future-only item
    (4, 2, 5, 170),  # extra validation positive
    (9, 1, 5, 199),
    (7, 3, 5, 250), (7, 7, 5, 260), (8, 7, 5, 270), (2, 3, 5, 300),  # test events
)


def _snapshot(events):
    tr, va, te = partition_by_cutoffs(events, T_VAL, T_TEST)
    st = build_stage("validation", tr, va)
    pop = Popularity().fit(st.prefix)
    knn = ItemKNN().fit(st.prefix)
    users = range(1, 10)
    return {
        "catalog": st.prefix.catalog.tolist(),
        "pop": pop.counts.tolist(),
        "knn_w": knn._w.copy(),
        "neighbors": knn.neighbor_lists,
        "seen": {u: st.prefix.seen(u).tolist() for u in users},
        "rank_pop": {u: pop.rank(u, st.candidates(u)).tolist() for u in users},
        "rank_knn": {u: knn.rank(u, st.candidates(u)).tolist() for u in users},
        "score_knn": {u: knn.score(u, st.candidates(u)).tolist() for u in users},
    }


def test_future_mutations_do_not_change_train_snapshot():
    a = _snapshot(BASE)
    mutated = np.concatenate([BASE, FUTURE_MUTATIONS])
    # also mutate the rating of existing held-out events
    mutated = mutated.copy()
    mutated[mutated[:, 3] >= T_VAL, 2] = 6 - mutated[mutated[:, 3] >= T_VAL, 2]
    b = _snapshot(mutated)
    for k in ("catalog", "pop", "neighbors", "seen", "rank_pop", "rank_knn", "score_knn"):
        assert a[k] == b[k], k
    assert np.array_equal(a["knn_w"], b["knn_w"])
    assert 6 not in b["catalog"] and 7 not in b["catalog"]


def test_test_events_do_not_change_validation_metrics():
    tr, va, te = partition_by_cutoffs(BASE, T_VAL, T_TEST)
    tr2, va2, te2 = partition_by_cutoffs(np.concatenate([BASE, FUTURE_MUTATIONS[3:]]), T_VAL, T_TEST)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and len(te2) > len(te)
    for M in (Popularity, ItemKNN):
        s1 = build_stage("validation", tr, va)
        s2 = build_stage("validation", tr2, va2)
        r1 = evaluate(s1, M().fit(s1.prefix))
        r2 = evaluate(s2, M().fit(s2.prefix))
        assert r1 == r2


def test_test_snapshot_built_from_train_plus_validation_only():
    tr, va, te = partition_by_cutoffs(BASE, T_VAL, T_TEST)
    st = build_test_stage(tr, va, te)
    assert st.prefix.catalog.tolist() == sorted(set(np.concatenate([tr, va])[:, 1].tolist()))
    assert 9 not in st.prefix.catalog.tolist()  # movie 9 appears only in TEST
    # mutating TEST events leaves pre-test features unchanged
    te_mut = te.copy()
    te_mut[:, 2] = 1
    st2 = build_test_stage(tr, va, np.concatenate([te_mut, ev((7, 2, 5, 290))]))
    assert Popularity().fit(st.prefix).counts.tolist() == Popularity().fit(st2.prefix).counts.tolist()
    assert np.array_equal(ItemKNN().fit(st.prefix)._w, ItemKNN().fit(st2.prefix)._w)
    # validation positives contribute to the test-stage prefix (e.g. movie 4 for user 1)
    assert 4 in st.prefix.positives(1).tolist()
    assert 4 not in build_stage("validation", tr, va).prefix.positives(1).tolist()


def test_validation_labels_do_not_drive_masking_or_catalog():
    tr, va, _ = partition_by_cutoffs(BASE, T_VAL, T_TEST)
    st = build_stage("validation", tr, va)
    # user 4's held-out non-positive (movie 3, rating 2) is still a candidate
    assert 3 in st.candidates(4).tolist()
    assert st.prefix.seen(4).tolist() == [1, 5]

"""Feedback types, catalog/ground truth, eligibility & cold-start (hand-verified fixtures)."""

import pytest
from conftest import ev

from personalized_recommender.baselines import ItemKNN, Popularity
from personalized_recommender.errors import EvaluationError
from personalized_recommender.evaluator import evaluate
from personalized_recommender.stage import (
    NONPOSITIVE,
    POSITIVE,
    UNKNOWN,
    Prefix,
    build_stage,
    feedback_class,
)

# ---------------- feedback types ----------------


def test_feedback_classes():
    assert [feedback_class(r) for r in (1, 2, 3, 4, 5)] == [NONPOSITIVE] * 3 + [POSITIVE] * 2
    assert feedback_class(None) == UNKNOWN
    with pytest.raises(ValueError):
        feedback_class(0)
    p = Prefix(ev((1, 10, 4, 1), (1, 11, 3, 2), (1, 12, 1, 3), (2, 10, 5, 4)))
    assert p.feedback(1, 10) == POSITIVE
    assert p.feedback(1, 11) == NONPOSITIVE
    assert p.feedback(1, 13) == UNKNOWN  # unobserved: unknown, not negative
    assert p.feedback(2, 11) == UNKNOWN


def test_positive_matrix_and_all_observed_mask():
    tr = ev((1, 10, 5, 1), (1, 11, 2, 2), (1, 12, 3, 3), (2, 11, 1, 4), (2, 13, 4, 5), (3, 10, 4, 6))
    p = Prefix(tr)
    assert p.positive_events[:, :3].tolist() == [[1, 10, 5], [2, 13, 4], [3, 10, 4]]
    assert p.positives(1).tolist() == [10]
    assert p.seen(1).tolist() == [10, 11, 12]  # low-rated items retained in seen
    assert p.candidates(1).tolist() == [13]
    assert p.candidates(2).tolist() == [10, 12]
    for M in (Popularity, ItemKNN):
        m = M().fit(p)
        assert m.rank(1, p.candidates(1)).tolist() == [13]
        assert set(m.rank(2, p.candidates(2)).tolist()) == {10, 12}


# ---------------- catalog and ground truth ----------------


def _t05_stage():
    tr = ev((1, 1, 5, 1), (1, 2, 1, 2), (2, 1, 4, 3), (2, 3, 2, 4), (3, 2, 5, 5))
    va = ev(
        (1, 3, 5, 10),  # known item, unseen by u1 -> GT
        (1, 9, 5, 11),  # future-only item -> out of catalog
        (2, 8, 4, 12),  # u2 only liked a future-only item -> ineligible
        (3, 1, 2, 13),  # u3 held-out non-positive -> no GT
        (3, 3, 4, 14),
    )
    return build_stage("validation", tr, va)


def test_stage_catalog_and_future_items():
    st = _t05_stage()
    assert st.prefix.catalog.tolist() == [1, 2, 3]  # includes nonpositive-only item 3
    assert 8 not in st.prefix.catalog and 9 not in st.prefix.catalog
    assert st.diagnostics["catalog_items_without_positive"] == 1


def test_ground_truth_and_out_of_catalog():
    st = _t05_stage()
    assert {u: g.tolist() for u, g in st.ground_truth.items()} == {1: [3], 3: [3]}
    d = st.diagnostics
    assert d["out_of_catalog_positive_count"] == 2
    assert d["users_with_out_of_catalog_positives"] == 2
    assert d["exclusion_reasons"]["positives_only_out_of_catalog"] == 1
    # u3's held-out non-positive item 1 remains a candidate (no future-label filtering)
    assert st.candidates(3).tolist() == [1, 3]


def test_heldout_positive_on_seen_item_not_gt():
    # Duplicate pairs are rejected at ingestion; an already-seen item cannot be GT by construction.
    st = build_stage("validation", ev((1, 1, 2, 1), (2, 2, 5, 2)), ev((1, 2, 5, 10)))
    assert st.ground_truth[1].tolist() == [2]
    assert 1 not in st.candidates(1).tolist()


# ---------------- eligibility and cold-start ----------------


def _t06_stage():
    tr = [(10, m, 3 + (m % 3 == 0), m) for m in range(1, 8)]  # u10: 7 prefix ratings (warm)
    tr += [(11, 1, 2, 20), (11, 2, 1, 21)]  # u11: 2 nonpositive ratings (cold 1..5)
    tr += [(12, 3, 5, 22)]  # u12: 1 rating, no held-out positive
    va = [(10, 8, 5, 100), (11, 3, 4, 101), (12, 4, 2, 102), (13, 5, 5, 103), (13, 2, 4, 104)]
    # u13: zero prefix history, liked known items -> eligible (unknown/new user)
    tr += [(14, 8, 4, 23)]
    return build_stage("validation", ev(*tr), ev(*va))


def test_eligibility():
    st = _t06_stage()
    assert st.eligible_users.tolist() == [10, 11, 13]
    assert st.diagnostics["exclusion_reasons"]["no_heldout_positive"] == 1  # u12
    assert st.diagnostics["excluded_users_count"] == 1
    assert st.diagnostics["eligible_users_zero_prefix_history"] == 1
    assert st.prefix.history_size(13) == 0
    assert st.prefix.history_size(11) == 2  # nonpositive ratings counted for history


def test_cold_start_cohorts_and_zero_history_fallback():
    st = _t06_stage()
    knn = ItemKNN().fit(st.prefix)
    r = evaluate(st, knn)
    cs = r["cold_start"]
    assert cs["cold_start_0"]["n_users"] == 1
    assert cs["cold_start_1_to_5"]["n_users"] == 1
    assert cs["cold_start_0_to_5"]["n_users"] == 2
    assert cs["warm_6_plus"]["n_users"] == 1
    # u13 (zero history) receives exact popularity ranking
    pop = Popularity().fit(st.prefix)
    assert knn.rank(13, st.candidates(13)).tolist() == pop.rank(13, st.candidates(13)).tolist()


def test_empty_cohort_is_null():
    tr = ev(*[(1, m, 4, m) for m in range(1, 9)], (2, 1, 4, 20))
    st = build_stage("validation", tr, ev((1, 2, 2, 50), (2, 9, 5, 51), (1, 9, 5, 52)))
    # u1 has seen 1..8, positive on future-only 9 -> ineligible; u2 also. => no evaluable users
    with pytest.raises(EvaluationError) as e:
        evaluate(st, Popularity().fit(st.prefix))
    assert e.value.code == "NO_EVALUABLE_USERS"
    # only u2 (1 prior rating) eligible -> cold_start_0 and warm cohorts empty
    st = build_stage("validation", tr, ev((2, 3, 5, 51)))
    r = evaluate(st, Popularity().fit(st.prefix))
    cs = r["cold_start"]
    assert cs["cold_start_0"]["n_users"] == 0
    assert cs["cold_start_0"]["metrics_by_k"]["10"] == {"recall": None, "hitrate": None, "ndcg": None}
    assert cs["warm_6_plus"]["n_users"] == 0 and cs["warm_6_plus"]["metrics_by_k"]["5"]["ndcg"] is None
    assert cs["cold_start_1_to_5"]["n_users"] == 1
    assert cs["cold_start_1_to_5"]["metrics_by_k"]["5"]["recall"] is not None

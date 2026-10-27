"""Global quantile-index cutoffs, tie handling, completeness/disjointness, no per-user split."""

import numpy as np
import pytest
from conftest import ev

from personalized_recommender.errors import SplitError
from personalized_recommender.split import compute_cutoffs, temporal_split
from personalized_recommender.data import canonical_sort


def test_exact_quantile_indices_n10():
    # 10 events with distinct timestamps 1..10; floor(0.8*10)=8 -> ts 9; floor(0.9*10)=9 -> ts 10
    e = ev(*[(u, u, 4, u) for u in range(1, 11)])
    s = temporal_split(e)
    assert (s.idx_val, s.idx_test, s.t_val, s.t_test) == (8, 9, 9, 10)
    assert len(s.train) == 8 and len(s.validation) == 1 and len(s.test) == 1


def test_floor_for_non_multiple_n():
    # N=13: floor(10.4)=10, floor(11.7)=11 (0-based)
    e = ev(*[(1, m, 3, 100 + m) for m in range(13)])
    s = temporal_split(e)
    assert (s.idx_val, s.idx_test) == (10, 11)
    assert (s.t_val, s.t_test) == (110, 111)


def test_ties_go_to_later_partition_whole_group():
    ts = [1, 2, 3, 4, 5, 6, 7, 7, 7, 8]  # idx8 -> 7 (tied group 7,7,7 starting at idx 6), idx9 -> 8
    e = ev(*[(i + 1, 1, 4, t) for i, t in enumerate(ts)])
    s = temporal_split(e)
    assert s.t_val == 7 and s.t_test == 8
    assert sorted(s.train[:, 3].tolist()) == [1, 2, 3, 4, 5, 6]  # 6 rows, not forced to 8
    assert s.validation[:, 3].tolist() == [7, 7, 7]  # entire tie group in later partition
    assert s.test[:, 3].tolist() == [8]


def test_tie_at_test_cutoff():
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]  # N=12: idx9 -> 9 (val), idx10 -> 9 => T_val==T_test
    e = ev(*[(i + 1, 1, 4, t) for i, t in enumerate(ts)])
    with pytest.raises(SplitError) as x:
        temporal_split(e)
    assert x.value.code == "UNSPLITTABLE_CUTOFFS"
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11]  # idx9 ->10, idx10 ->10 => unsplittable too
    with pytest.raises(SplitError):
        temporal_split(ev(*[(i + 1, 1, 4, t) for i, t in enumerate(ts)]))
    # N=20: idx16 -> ts 4 (T_val), idx18 -> ts 5 (T_test)
    ts = [1] * 5 + [2] * 5 + [3] * 5 + [4] * 2 + [5] * 2 + [6]  # idx16 -> 4, idx18 -> 5
    s = temporal_split(ev(*[(i + 1, 1, 4, t) for i, t in enumerate(ts)]))
    assert (s.t_val, s.t_test) == (4, 5)
    assert s.validation[:, 3].tolist() == [4, 4]
    assert s.test[:, 3].tolist() == [5, 5, 6]  # both tied ts=5 rows in TEST


def test_strict_boundaries_and_union_disjoint():
    rng = np.random.default_rng(0)
    rows = {(int(u), int(m)): (int(rng.integers(1, 6)), int(rng.integers(1, 60))) for u, m in rng.integers(1, 40, (600, 2))}
    e = ev(*[(u, m, r, t) for (u, m), (r, t) in rows.items()])
    s = temporal_split(e)
    assert (s.train[:, 3] < s.t_val).all()
    assert ((s.validation[:, 3] >= s.t_val) & (s.validation[:, 3] < s.t_test)).all()
    assert (s.test[:, 3] >= s.t_test).all()
    allp = np.concatenate([s.train, s.validation, s.test])
    assert len(allp) == len(e)
    assert np.array_equal(canonical_sort(allp), canonical_sort(e))  # union == source, no loss/dup
    keys = [set(map(tuple, p[:, :2].tolist())) for p in (s.train, s.validation, s.test)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    sorted_ts = np.sort(e[:, 3])
    assert s.t_val == sorted_ts[(8 * len(e)) // 10] and s.t_test == sorted_ts[(9 * len(e)) // 10]


def test_global_not_per_user_and_not_random():
    # user 1 has all events early; per-user leave-last-out would put one of them in held-out.
    rows = [(1, m, 5, m) for m in range(1, 9)] + [(2, 1, 5, 100), (2, 2, 5, 200)]
    s = temporal_split(ev(*rows))
    assert 1 not in s.validation[:, 0].tolist() + s.test[:, 0].tolist()
    # deterministic: input row order does not change the outcome
    s2 = temporal_split(ev(*reversed(rows)))
    for a, b in ((s.train, s2.train), (s.validation, s2.validation), (s.test, s2.test)):
        assert np.array_equal(a, b)


def test_cutoffs_require_canonical_sort():
    e = canonical_sort(ev((2, 5, 4, 10), (1, 5, 4, 10), (1, 6, 4, 9)))
    assert e.tolist() == [[1, 6, 4, 9], [1, 5, 4, 10], [2, 5, 4, 10]]
    with pytest.raises(SplitError):
        compute_cutoffs(e)  # N=3: idx2 & idx2 -> equal


def test_real_split(ml1m_dir):
    from personalized_recommender.data import load_dataset

    events, _ = load_dataset(ml1m_dir)
    s = temporal_split(events)
    srt = canonical_sort(events)
    n = len(events)
    assert s.t_val == int(srt[(8 * n) // 10, 3]) and s.t_test == int(srt[(9 * n) // 10, 3])
    assert s.t_val < s.t_test
    assert len(s.train) + len(s.validation) + len(s.test) == n == 1_000_209
    assert s.train[:, 3].max() < s.t_val <= s.validation[:, 3].min()
    assert s.validation[:, 3].max() < s.t_test <= s.test[:, 3].min()
    # no timestamp is shared by two partitions
    tr, va, te = (set(np.unique(p[:, 3]).tolist()) for p in (s.train, s.validation, s.test))
    assert not (tr & va) and not (va & te) and not (tr & te)

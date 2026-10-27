"""Eight-trial search, selection lock, benchmark refit/fairness, diagnostics, CLI and lineage."""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import PKG_FIXTURE, ev

import personalized_recommender.r2 as R2
from personalized_recommender.artifacts import canonical_json, prepare
from personalized_recommender.errors import ArtifactError
from personalized_recommender.models import ALS_GRID, BPR_GRID, LatentFactorRanker
from personalized_recommender.stage import build_stage, build_test_stage



def _cli(*args):
    return subprocess.run([sys.executable, "-m", "personalized_recommender", *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Synthetic fixture: prepare -> train (8 real trials) -> benchmark, all through the CLI."""
    d = tmp_path_factory.mktemp("r2")
    art, exp = d / "art", d / "exp"
    assert _cli("prepare", "--data-dir", PKG_FIXTURE, "--output-dir", art).returncode == 0
    t = _cli("train", "--artifacts", art, "--output-dir", exp)
    assert t.returncode == 0, t.stderr
    b = _cli("benchmark", "--artifacts", art, "--experiment-dir", exp, "--output", d / "bench.json")
    assert b.returncode == 0, b.stderr
    return {"dir": d, "art": art, "exp": exp, "train": json.loads(t.stdout),
            "sel": json.loads((exp / "selection_manifest.json").read_text()),
            "bench": json.loads((d / "bench.json").read_text())}


# ---------------- eight-trial search ----------------


def test_eight_trials_recorded(campaign):
    sel = campaign["sel"]
    ids = [t["config"]["config_id"] for t in sel["trials"]]
    assert ids == ["ALS-01", "ALS-02", "ALS-03", "ALS-04", "BPR-01", "BPR-02", "BPR-03", "BPR-04"]
    assert [t["config"] for t in sel["trials"]] == [c.as_dict() for c in ALS_GRID + BPR_GRID]
    for t in sel["trials"]:
        assert t["status"] == "completed" and t["config"]["seed"] == 42
        assert t["train_seconds"] > 0
        assert set(t["metrics_by_k"]) == {"5", "10", "20"}
        assert all(set(v) == {"recall", "hitrate", "ndcg", "catalog_coverage"} for v in t["metrics_by_k"].values())
        assert t["user_factor_shape"][1] == t["config"]["factors"]
    assert sel["base_seed"] == 42 and sel["thread_settings"]["model_training_threads"] == 1
    assert sel["test_labels_accessed"] is False


def test_train_never_decodes_test(tmp_path, monkeypatch):
    art = tmp_path / "art"
    prepare(PKG_FIXTURE, art)
    decoded = []
    orig = R2._decode_partition
    monkeypatch.setattr(R2, "_decode_partition", lambda n, raw: decoded.append(n) or orig(n, raw))
    import personalized_recommender.artifacts as A

    monkeypatch.setattr(A, "load_partitions", lambda *a, **k: pytest.fail("train must not load TEST rows"))
    R2.run_train(art, tmp_path / "exp", grid=(ALS_GRID[:1], BPR_GRID[:1]))
    assert sorted(decoded) == ["train", "validation"]


# ---------------- selection rules and manifest lock ----------------


def _trial(cid, ndcg, rec, hit):
    return {"config": {"config_id": cid}, "metrics_by_k": {"10": {"ndcg": ndcg, "recall": rec, "hitrate": hit}}}


def test_per_family_selection_and_ties():
    assert R2.select_config([_trial("A1", .2, .1, .5), _trial("A2", .3, .0, .0)])["config"]["config_id"] == "A2"
    assert R2.select_config([_trial("A1", .3, .1, .5), _trial("A2", .3, .2, .0)])["config"]["config_id"] == "A2"
    assert R2.select_config([_trial("A1", .3, .2, .4), _trial("A2", .3, .2, .5)])["config"]["config_id"] == "A2"
    assert R2.select_config([_trial("A1", .3, .2, .5), _trial("A2", .3, .2, .5)])["config"]["config_id"] == "A1"
    # no tolerance: 0.3000000001 beats 0.3
    assert R2.select_config([_trial("A1", .3, .9, .9), _trial("A2", .3000000001, 0, 0)])["config"]["config_id"] == "A2"


def test_default_method_rule():
    same = {m: _trial(m, .3, .2, .5) for m in ("popularity", "item_knn", "als", "bpr")}
    assert R2.select_default(same) == "popularity"
    same["bpr"] = _trial("bpr", .3, .2, .51)
    assert R2.select_default(same) == "bpr"
    same["item_knn"] = _trial("item_knn", .31, 0, 0)
    assert R2.select_default(same) == "item_knn"


def test_manifest_canonical_hash_and_lock(campaign):
    sel = campaign["sel"]
    assert R2.manifest_hash(sel) == sel["manifest_sha256"] == campaign["train"]["manifest_sha256"]
    p = campaign["exp"] / "selection_manifest.json"
    assert not os.access(p, os.W_OK) or os.stat(p).st_mode & 0o222 == 0
    r = _cli("train", "--artifacts", campaign["art"], "--output-dir", campaign["exp"])
    assert r.returncode == 2 and "SELECTION_ALREADY_LOCKED" in r.stderr
    assert campaign["bench"]["selection_manifest_sha256"] == sel["manifest_sha256"]
    assert campaign["bench"]["selection_verified_at_unix"] >= os.stat(p).st_mtime  # lock predates TEST
    assert sel["selected"]["default_method"] == campaign["bench"]["default_method"]


def test_future_label_mutation_cannot_affect_selection(tmp_path):
    """Change ratings of TEST-period events only (timestamps untouched -> identical cutoffs)."""
    raw = (PKG_FIXTURE / "ratings.dat").read_text().splitlines()
    rows = [tuple(map(int, line.split("::"))) for line in raw]
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    shutil.copy(PKG_FIXTURE / "ratings.dat", a / "ratings.dat")
    prepare(a, tmp_path / "art_a")
    t_test = json.loads((tmp_path / "art_a" / "manifest.json").read_text())["split"]["t_test"]
    mutated = [(u, m, (6 - r if t >= t_test else r), t) for u, m, r, t in rows]
    (b / "ratings.dat").write_text("".join(f"{u}::{m}::{r}::{t}\n" for u, m, r, t in mutated))
    prepare(b, tmp_path / "art_b")
    sb = json.loads((tmp_path / "art_b" / "manifest.json").read_text())["split"]
    assert (sb["t_val"], sb["t_test"]) == (json.loads((tmp_path / "art_a" / "manifest.json").read_text())["split"]["t_val"], t_test)
    sa = R2.run_train(tmp_path / "art_a", tmp_path / "ea")
    sb_ = R2.run_train(tmp_path / "art_b", tmp_path / "eb")
    assert sa["selected"] == sb_["selected"]
    strip = lambda s: [{k: v for k, v in t.items() if not k.endswith("_seconds")} for t in s["trials"]]  # noqa: E731
    assert strip(sa) == strip(sb_)
    assert sa["data_lineage"]["file_sha256"]["test"] != sb_["data_lineage"]["file_sha256"]["test"]


# ---------------- benchmark refit and fairness ----------------


def test_test_stage_refit_ignores_test_rows():
    from personalized_recommender.split import partition_by_cutoffs

    tr = ev(*[(u, m, 5 if (u + m) % 3 else 2, u * 7 + m) for u in range(1, 9) for m in range(1, 7)])
    va = ev((1, 7, 5, 500), (2, 8, 4, 501), (9, 1, 5, 502))
    te = ev((1, 9, 5, 900), (4, 7, 5, 901), (2, 7, 5, 902))
    te2 = ev((1, 9, 1, 900), (5, 7, 5, 903), (6, 10, 5, 904))
    st1 = build_test_stage(tr, va, te)
    st2 = build_test_stage(tr, va, te2)
    assert st1.prefix.catalog.tolist() == list(range(1, 9))  # TRAIN+VALIDATION items (7,8 from VALIDATION)
    for cfg in (ALS_GRID[0], BPR_GRID[0]):
        r1 = LatentFactorRanker(cfg).fit(st1.prefix)
        r2 = LatentFactorRanker(cfg).fit(st2.prefix)
        assert np.array_equal(r1.user_factors, r2.user_factors) and np.array_equal(r1.item_factors, r2.item_factors)
        assert 9 not in r1.matrix.item_ids.tolist() and 9 in r1.matrix.user_ids.tolist()  # val positive user
        rv = LatentFactorRanker(cfg).fit(build_stage("validation", tr, va).prefix)
        assert rv.item_factors.shape[0] != r1.item_factors.shape[0]  # fresh mapping, not reused validation factors


def test_benchmark_four_methods_same_population(campaign):
    b = campaign["bench"]
    assert b["methods"] == ["popularity", "item_knn", "als", "bpr"]
    res = b["results"]
    assert len({r["eligible_users_count"] for r in res.values()}) == 1
    assert len({r["ranked_candidate_pairs"] for r in res.values()}) == 1
    assert b["stage_diagnostics"]["eligible_users_count"] == res["als"]["eligible_users_count"]
    for name, r in res.items():
        assert set(r["metrics_by_k"]) == {"5", "10", "20"}
        assert r["train_seconds"] >= 0 and r["test_eval_seconds"] > 0
    for mdl in ("als", "bpr"):
        for base in ("popularity", "item_knn"):
            c = b["comparisons"][f"{mdl}_vs_{base}"]
            for metric in ("ndcg", "recall"):
                assert c[f"{metric}@10_delta"] == pytest.approx(res[mdl]["metrics_by_k"]["10"][metric] - res[base]["metrics_by_k"]["10"][metric])
    assert b["results"]["als"]["configuration"] == campaign["sel"]["selected"]["als"]
    assert b["results"]["bpr"]["configuration"] == campaign["sel"]["selected"]["bpr"]
    assert b["refit_prefix"].startswith("TRAIN+VALIDATION")


def test_benchmark_replay_identical_metrics(campaign):
    out = campaign["dir"] / "bench2.json"
    assert _cli("benchmark", "--artifacts", campaign["art"], "--experiment-dir", campaign["exp"], "--output", out).returncode == 0
    b2 = json.loads(out.read_text())
    for name in b2["methods"]:
        assert b2["results"][name]["metrics_by_k"] == campaign["bench"]["results"][name]["metrics_by_k"]
    assert b2["selected"] == campaign["bench"]["selected"]


# ---------------- diagnostics ----------------


def test_cold_start_and_fallback_oracle():
    from personalized_recommender.evaluator import evaluate

    tr = ev(*[(u, m, 5, u * 10 + m) for u in (1, 2, 3) for m in range(1, 8)])  # warm users, 7 ratings each
    tr = np.concatenate([tr, ev((4, 1, 2, 100), (4, 2, 3, 101))])  # u4: 2 low ratings (cold 1..5, no factor)
    va = ev((5, 3, 4, 200))  # u5 history comes from VALIDATION: 1 positive -> cold 1..5 with factor
    te = ev((1, 9, 5, 900), (6, 2, 5, 901), (4, 3, 5, 902), (5, 4, 5, 903), (2, 8, 5, 904))
    st = build_test_stage(tr, va, te)
    assert st.eligible_users.tolist() == [4, 5, 6]  # u1/u2 liked future-only items
    r = LatentFactorRanker(ALS_GRID[0]).fit(st.prefix)
    res = evaluate(st, r)
    cs = res["cold_start"]
    assert (cs["cold_start_0"]["n_users"], cs["cold_start_1_to_5"]["n_users"], cs["cold_start_0_to_5"]["n_users"], cs["warm_6_plus"]["n_users"]) == (1, 2, 3, 0)
    assert cs["warm_6_plus"]["metrics_by_k"]["10"] == {"recall": None, "hitrate": None, "ndcg": None}
    assert r.fallback_users == 2 and r.personalized_users == 1  # u6 unknown, u4 low-only
    # cold_start_0: u6 single relevant item 2; popularity order over candidates
    assert cs["cold_start_0"]["metrics_by_k"]["20"]["recall"] == 1.0


# ---------------- CLI ----------------


def test_head_tail_rule():
    tr = ev((1, 10, 5, 1), (2, 10, 5, 2), (1, 20, 5, 3), (3, 30, 1, 4), (2, 40, 4, 5), (3, 50, 4, 6), (4, 60, 2, 7))
    st = build_stage("validation", tr, ev((4, 10, 5, 50)))
    ht = R2.head_tail(st.prefix)
    # pops: 10:2, 20:1, 40:1, 50:1, 30:0, 60:0 ; |C|=6 -> head ceil(1.2)=2 -> [10, 20] (ties by id)
    assert ht["n_head"] == 2 and ht["head"].tolist() == [10, 20]
    assert ht["tail"].tolist() == [30, 40, 50, 60] and ht["tail_zero_positive_items"] == 2


class _Stage:
    def __init__(self, gt):
        self.eligible_users = np.array(sorted(gt))
        self.ground_truth = {u: np.array(v) for u, v in gt.items()}


def test_tail_recall_and_exposure_denominators():
    tail = np.array([3, 4, 5, 6])
    st = _Stage({1: [1, 3, 4], 2: [2], 3: [5]})
    top = {1: np.array([3, 1, 2]), 2: np.array([2, 6, 7, 8]), 3: np.array([9])}  # u1 emitted 3 (short list)
    m = R2.tail_metrics(st, top, tail)
    assert m["tail_eligible_users"] == 2  # u2 has no tail GT
    assert m["tail_recall_at_10"] == pytest.approx((1 / 2 + 0 / 1) / 2)
    assert m["emitted_positions_at_10"] == 8 and m["tail_positions_at_10"] == 2
    assert m["tail_exposure_at_10"] == pytest.approx(2 / 8)
    z = R2.tail_metrics(_Stage({1: [1], 2: [2]}), {1: np.array([1]), 2: np.array([2])}, tail)
    assert z["tail_eligible_users"] == 0 and z["tail_recall_at_10"] is None and z["tail_exposure_at_10"] == 0.0


def test_benchmark_long_tail_fields(campaign):
    b = campaign["bench"]
    h = b["head_tail_definition"]
    assert h["head_item_count"] == math.ceil(0.2 * h["catalog_size"])
    assert h["head_item_count"] + h["tail_item_count"] == h["catalog_size"]
    for r in b["results"].values():
        lt = r["long_tail"]
        assert lt["emitted_positions_at_10"] <= 10 * r["eligible_users_count"]


# ---------------- lineage and failure handling ----------------


def _tamper(exp: Path, fn, rehash=True):
    p = exp / "selection_manifest.json"
    os.chmod(p, 0o644)
    m = json.loads(p.read_text())
    fn(m)
    if rehash:
        m["manifest_sha256"] = R2.manifest_hash(m)
    p.write_text(canonical_json(m))


@pytest.mark.parametrize(
    "fn,rehash,code",
    [
        (lambda m: m["trials"][0]["metrics_by_k"]["10"].update(ndcg=0.99), False, "SELECTION_CHECKSUM_MISMATCH"),
        (lambda m: m.update(protocol_version="P02-R2-ACCEPTANCE-v0.9"), True, "PROTOCOL_MISMATCH"),
        (lambda m: m["code_identity"].update(source_sha256="0" * 64), True, "SOURCE_IDENTITY_MISMATCH"),
        (lambda m: m["data_lineage"].update(ratings_sha256="f" * 64), True, "DATA_LINEAGE_MISMATCH"),
        (lambda m: m["trials"][0]["config"].update(factors=128), True, "SELECTION_INCONSISTENT"),
        (lambda m: m["trials"].pop(), True, "SELECTION_INCONSISTENT"),
        (lambda m: m["selected"].update(default_method="bpr" if m["selected"]["default_method"] != "bpr" else "als"), True, "SELECTION_INCONSISTENT"),
        (lambda m: m.update(test_labels_accessed=True), True, "SELECTION_NOT_FINALIZED"),
        (lambda m: m.pop("selected"), True, "SELECTION_CORRUPT"),
    ],
)
def test_benchmark_rejects_invalid_selection(campaign, tmp_path, fn, rehash, code):
    exp = tmp_path / "exp"
    shutil.copytree(campaign["exp"], exp)
    _tamper(exp, fn, rehash)
    r = _cli("benchmark", "--artifacts", campaign["art"], "--experiment-dir", exp, "--output", tmp_path / "o.json")
    assert r.returncode == 2 and code in r.stderr, r.stderr
    assert r.stdout == "" and not (tmp_path / "o.json").exists()


def test_benchmark_without_manifest_and_corrupt(campaign, tmp_path):
    r = _cli("benchmark", "--artifacts", campaign["art"], "--experiment-dir", tmp_path / "none", "--output", tmp_path / "o.json")
    assert r.returncode == 2 and "SELECTION_MISSING" in r.stderr
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "selection_manifest.json").write_text("{oops")
    r = _cli("benchmark", "--artifacts", campaign["art"], "--experiment-dir", tmp_path / "bad", "--output", tmp_path / "o.json")
    assert r.returncode == 2 and "SELECTION_CORRUPT" in r.stderr


def test_benchmark_rejects_other_dataset(campaign, tmp_path):
    rows = (PKG_FIXTURE / "ratings.dat").read_text().splitlines()
    d = tmp_path / "d"
    d.mkdir()
    (d / "ratings.dat").write_text("\n".join(rows[:-1]) + "\n")
    prepare(d, tmp_path / "art2")
    r = _cli("benchmark", "--artifacts", tmp_path / "art2", "--experiment-dir", campaign["exp"], "--output", tmp_path / "o.json")
    assert r.returncode == 2 and "DATA_LINEAGE_MISMATCH" in r.stderr


def test_train_rejects_corrupt_artifacts(tmp_path):
    art = tmp_path / "art"
    prepare(PKG_FIXTURE, art)
    b = bytearray((art / "test.npy").read_bytes())
    b[-2] ^= 1
    (art / "test.npy").write_bytes(bytes(b))
    r = _cli("train", "--artifacts", art, "--output-dir", tmp_path / "exp")
    assert r.returncode == 2 and "CHECKSUM_MISMATCH" in r.stderr
    assert not (tmp_path / "exp" / "selection_manifest.json").exists()
    assert _cli("train", "--artifacts", tmp_path / "nope", "--output-dir", tmp_path / "e").returncode == 2
    assert _cli("benchmark", "--artifacts", art).returncode == 2  # malformed options


def test_legacy_test_seal_intact(campaign):
    for model in ("popularity", "item_knn"):
        r = _cli("evaluate", "--artifacts", campaign["art"], "--model", model, "--split", "test")
        assert r.returncode == 2 and "TEST_HOLDOUT_SEALED" in r.stderr and r.stdout == ""
    for bad in ("als", "bpr"):
        assert _cli("evaluate", "--artifacts", campaign["art"], "--model", bad, "--split", "validation").returncode == 2


def test_nonfinite_model_exit_1(tmp_path, monkeypatch):
    import personalized_recommender.cli as cli
    import personalized_recommender.models as M

    art = tmp_path / "art"
    prepare(PKG_FIXTURE, art)
    monkeypatch.setattr(M, "fit_als", lambda m, c: (np.full((m.n_users, c.factors), np.nan), np.zeros((m.n_items, c.factors)), {"loss_trace": []}))
    assert cli.main(["train", "--artifacts", str(art), "--output-dir", str(tmp_path / "exp")]) == 1
    assert not (tmp_path / "exp" / "selection_manifest.json").exists()


# ---------------- real-data run ----------------

def test_real_ml1m_train_validation_only(ml1m_dir, tmp_path):
    """Real eight-trial search on official ML-1M (VALIDATION only; the TEST campaign runs via the CLI)."""
    art = tmp_path / "art"
    if not os.environ.get("P02_ML1M_ARCHIVE"):
        pytest.skip("requires official ml-1m.zip (not shipped): set P02_ML1M_ARCHIVE")
    assert _cli("prepare", "--data-dir", ml1m_dir, "--output-dir", art, "--archive", os.environ["P02_ML1M_ARCHIVE"]).returncode == 0
    # R1 non-regression: prepared artifacts are byte-identical to the recorded official identity
    assert json.loads((art / "manifest.json").read_text())["manifest_fingerprint"] == \
        "97b40e93b843d598caf90408aa916afcdcb374d9be454d7229beb58e1312b532"
    r = _cli("train", "--artifacts", art, "--output-dir", tmp_path / "exp")
    assert r.returncode == 0, r.stderr
    sel = json.loads((tmp_path / "exp" / "selection_manifest.json").read_text())
    assert [t["status"] for t in sel["trials"]] == ["completed"] * 8
    assert sel["r1_baseline_reconciliation"]["status"] == "MATCH"
    assert sel["data_lineage"]["split_fingerprint"] == "046dd86418ca325b27f4a3dd41fea24f3a305d7c0fa976ccbd92d2cb05f1fc69"
    assert sel["validation_stage"]["diagnostics"]["eligible_users_count"] == 1103
    for t in sel["trials"]:
        assert t["fallback_users"] == 611 and t["personalized_users"] == 492
        assert t["item_factor_shape"] == [3662, t["config"]["factors"]]

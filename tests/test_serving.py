"""Serving bundle, offline service, parity, CLI, latency.

Synthetic-fixture oracles run always; tests marked REAL need P02_ML1M_DIR / P02_ML1M_ARCHIVE (official data).
"""

import copy
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

import personalized_recommender.serving as S
from personalized_recommender.artifacts import canonical_json, prepare
from personalized_recommender.baselines import ItemKNN, Popularity
from personalized_recommender.errors import ArtifactError, BundleError, InvalidRequestError, ModelIntegrityError
from personalized_recommender.models import LatentFactorRanker
from personalized_recommender.stage import Prefix


def _cli(*args, cwd=None, env=None):
    return subprocess.run([sys.executable, "-m", "personalized_recommender", *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, env=env)

@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("r3")
    art = d / "art"
    prepare(PKG_FIXTURE, art)
    manifest, ref = S.export_bundle(art, d / "bundle")
    return {"dir": d, "art": art, "bundle": d / "bundle", "manifest": manifest, "ref": ref,
            "svc": S.RecommenderService.load(d / "bundle")}

def _copy_bundle(fx, tmp_path):
    dst = tmp_path / "b"
    shutil.copytree(fx["bundle"], dst)
    return dst

def _rewrite_manifest(bundle: Path, mutate):
    m = json.loads((bundle / "manifest.json").read_text())
    mutate(m)
    m["manifest_sha256"] = S.manifest_self_hash(m)
    (bundle / "manifest.json").write_text(canonical_json(m))

# ----------------- bundle provenance and export

def test_provenance_and_prefix(fx):
    m = fx["manifest"]
    assert m["schema"] == "p02-serving-bundle/1" and m["dataset_kind"] == "synthetic_fixture"
    assert m["lineage"]["official_reference_match"] is False
    assert m["contracts"]["r1"]["sha256"].startswith("e5b6f10b") and m["contracts"]["r2"]["sha256"].startswith("b011ac87")
    assert m["contracts"]["r3"]["sha256"] .startswith("7eaa87a6")
    r2 = m["r2_provenance"]
    assert r2["default_method"] == "item_knn" and r2["selected_als"]["config_id"] == "ALS-01"
    assert r2["selected_bpr"]["config_id"] == "BPR-02" and r2["accepted_selection_manifest_sha256"].startswith("d0e33164")
    assert (r2["selected_als"]["factors"], r2["selected_als"]["alpha"], r2["selected_als"]["regularization"]) == (32, 10.0, 0.05)
    assert (r2["selected_bpr"]["learning_rate"], r2["selected_bpr"]["batch_size"], r2["selected_bpr"]["iterations"]) == (0.05, 1024, 15)
    # prefix is exactly TRAIN+VALIDATION: nothing at or after T_test
    assert m["prefix"]["max_timestamp"] < m["prefix"]["t_test"]
    parts = {p: np.load(fx["art"] / f"{p}.npy") for p in ("train", "validation", "test")}
    assert m["prefix"]["events"] == len(parts["train"]) + len(parts["validation"])
    test_pairs = {(int(u), int(i)) for u, i in parts["test"][:, :2]}
    h = np.load(fx["bundle"] / "history.npz")
    seen = {(int(u), int(i)) for u, a, b in zip(h["user_ids"], h["seen_indptr"][:-1], h["seen_indptr"][1:]) for i in h["seen_items"][a:b]}
    assert not (seen & test_pairs)  # no TEST row in histories
    assert fx["svc"].default_method == "item_knn"

def test_export_never_reads_test_rows(tmp_path, monkeypatch):
    art = tmp_path / "art"
    prepare(PKG_FIXTURE, art)
    (art / "test.npy").write_bytes((art / "test.npy").read_bytes())  # untouched bytes
    import personalized_recommender.r2 as R2

    decoded = []
    real = R2._decode_partition
    monkeypatch.setattr(R2, "_decode_partition", lambda n, raw: (decoded.append(n), real(n, raw))[1])
    S.export_bundle(art, tmp_path / "b")
    assert decoded == ["train", "validation"]

def test_state_complete_and_numeric(fx):
    b = fx["bundle"]
    assert sorted(p.name for p in b.iterdir()) == sorted(["manifest.json", *S.DATA_FILES])
    ref = fx["ref"]
    prefix = ref["prefix"]
    h = np.load(b / "history.npz", allow_pickle=False)
    assert np.array_equal(h["catalog"], prefix.catalog) and np.array_equal(h["user_ids"], prefix.users)
    n_seen = sum(len(prefix.seen(u)) for u in prefix.users)
    assert len(h["seen_items"]) == n_seen > len(h["pos_items"])  # includes ratings 1-3
    assert np.array_equal(np.load(b / "popularity.npz")["counts"], ref["popularity"].counts)
    k = np.load(b / "item_knn.npz")
    assert np.diff(k["indptr"]).max() <= 50
    for name in ("als", "bpr"):
        a = np.load(b / f"{name}.npz", allow_pickle=False)
        assert a["user_factors"].shape == (len(a["user_ids"]), 32) and a["item_factors"].shape == (len(prefix.catalog), 32)
        assert all(np.isfinite(a[x]).all() for x in ("user_factors", "item_factors"))
    for f in S.DATA_FILES:
        with np.load(b / f, allow_pickle=False) as z:
            assert all(z[n].dtype.kind in "if" for n in z.files)
    m = fx["manifest"]
    assert m["files"]["als.npz"]["arrays"]["user_factors"]["dtype"] == "<f8"
    assert S.manifest_self_hash(json.loads((b / "manifest.json").read_text())) == m["manifest_sha256"]

# ----------------- corruption and portability

def test_missing_and_tampered_files(fx, tmp_path):
    for fname in S.DATA_FILES:
        dst = tmp_path / ("m_" + fname)
        shutil.copytree(fx["bundle"], dst)
        (dst / fname).unlink()
        with pytest.raises(BundleError):
            S.RecommenderService.load(dst)
        dst2 = tmp_path / ("t_" + fname)
        shutil.copytree(fx["bundle"], dst2)
        raw = bytearray((dst2 / fname).read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (dst2 / fname).write_bytes(bytes(raw))
        with pytest.raises(BundleError) as e:
            S.RecommenderService.load(dst2)
        assert e.value.code == "BUNDLE_CHECKSUM_MISMATCH"
        dst3 = tmp_path / ("x_" + fname)
        shutil.copytree(fx["bundle"], dst3)
        (dst3 / fname).write_bytes((dst3 / fname).read_bytes()[:50])
        with pytest.raises(BundleError):
            S.RecommenderService.load(dst3)

def test_manifest_and_schema_tamper(fx, tmp_path):
    d = _copy_bundle(fx, tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["serving"]["catalog_count"] += 1
    (d / "manifest.json").write_text(canonical_json(m))  # self-hash NOT refreshed
    with pytest.raises(BundleError) as e:
        S.RecommenderService.load(d)
    assert e.value.code == "BUNDLE_MANIFEST_TAMPERED"
    for name, mut, code in [
        ("schema", lambda m: m.__setitem__("schema", "p02-serving-bundle/2"), "BUNDLE_UNSUPPORTED_SCHEMA"),
        ("default", lambda m: m["serving"].__setitem__("default_method", "als"), "BUNDLE_INCOMPATIBLE"),
        ("als_cfg", lambda m: m["serving"]["model_configs"]["als"].__setitem__("alpha", 40.0), "BUNDLE_INCOMPATIBLE"),
        ("selection", lambda m: m["r2_provenance"].__setitem__("accepted_selection_manifest_sha256", "0" * 64), "BUNDLE_INCOMPATIBLE"),
        ("kind", lambda m: m.__setitem__("dataset_kind", "official_movielens_1m"), "BUNDLE_LINEAGE"),
        ("synth_official", lambda m: m["lineage"].__setitem__("official_reference_match", True), "BUNDLE_LINEAGE"),
        ("methods", lambda m: m["serving"]["methods"].pop(), "BUNDLE_INCOMPATIBLE"),
    ]:
        dd = tmp_path / ("c_" + name)
        shutil.copytree(fx["bundle"], dd)
        _rewrite_manifest(dd, mut)
        with pytest.raises(BundleError) as e:
            S.RecommenderService.load(dd)
        assert e.value.code == code, (name, e.value)
    for p in (tmp_path / "nonexistent", tmp_path):
        with pytest.raises(BundleError):
            S.RecommenderService.load(p)

def _forge(fx, tmp_path, tag, fname, mutate):
    """Rewrite an NPZ AND refresh its hash + manifest self-hash: exercises the semantic validators, not the checksum."""
    d = tmp_path / tag
    shutil.copytree(fx["bundle"], d)
    with np.load(d / fname) as z:
        arrs = {k: z[k].copy() for k in z.files}
    mutate(arrs)
    np.savez(d / fname, **arrs)

    def upd(m):
        m["files"][fname]["sha256"] = S.sha256_bytes((d / fname).read_bytes())
        m["files"][fname]["arrays"] = {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrs.items()}

    _rewrite_manifest(d, upd)
    return d

@pytest.mark.parametrize("tag,fname,mutate", [
    ("unsorted_catalog", "history.npz", lambda a: a["catalog"].__setitem__(slice(0, 2), a["catalog"][1::-1].copy())),
    ("seen_oob", "history.npz", lambda a: a["seen_items"].__setitem__(0, 10**9)),
    ("pop_counts", "popularity.npz", lambda a: a["counts"].__setitem__(0, a["counts"][0] + 5)),
    ("knn_oob", "item_knn.npz", lambda a: a["neighbor_index"].__setitem__(0, 10**6)),
    ("knn_selfloop", "item_knn.npz", lambda a: a["neighbor_index"].__setitem__(0, 0)),
    ("als_nan", "als.npz", lambda a: a["item_factors"].__setitem__((0, 0), np.nan)),
    ("bpr_inf", "bpr.npz", lambda a: a["user_factors"].__setitem__((0, 0), np.inf)),
    ("als_dim", "als.npz", lambda a: a.__setitem__("user_factors", a["user_factors"][:, :16].copy())),
    ("bpr_users", "bpr.npz", lambda a: a.__setitem__("user_ids", a["user_ids"][::-1].copy())),
    ("als_items", "als.npz", lambda a: a["item_ids"].__setitem__(0, a["item_ids"][0] + 777)),
    ("object_dtype", "history.npz", lambda a: a.__setitem__("catalog", a["catalog"].astype("<i4"))),
])
def test_forged_state_rejected(fx, tmp_path, tag, fname, mutate):
    d = _forge(fx, tmp_path, tag, fname, mutate)
    with pytest.raises(BundleError):
        S.RecommenderService.load(d)

def test_pickle_never_executed(fx, tmp_path):
    d = _copy_bundle(fx, tmp_path)
    np.savez(d / "popularity.npz", counts=np.array([object()], dtype=object))  # pickled object array inside npz
    _rewrite_manifest(d, lambda m: m["files"]["popularity.npz"].__setitem__("sha256", S.sha256_bytes((d / "popularity.npz").read_bytes())))
    with pytest.raises(BundleError):
        S.RecommenderService.load(d)

def test_extra_file_and_portability(fx, tmp_path):
    d = _copy_bundle(fx, tmp_path)
    (d / "extra.bin").write_bytes(b"x")
    with pytest.raises(BundleError):
        S.RecommenderService.load(d)
    moved = tmp_path / "deep" / "er" / "moved_bundle"
    moved.parent.mkdir(parents=True)
    shutil.copytree(fx["bundle"], moved)
    a, b = fx["svc"].recommend(3, 10, "als"), S.RecommenderService.load(moved).recommend(3, 10, "als")
    assert a == b

def test_export_collision_refused(fx, tmp_path):
    with pytest.raises(ArtifactError) as e:
        S.export_bundle(fx["art"], fx["bundle"])
    assert e.value.code == "BUNDLE_EXISTS"

# ----------------- ranking semantics

def test_known_unknown_zero_positive(fx):
    svc, prefix = fx["svc"], fx["ref"]["prefix"]
    zero_pos = [int(u) for u in prefix.users if len(prefix.positives(u)) == 0]
    assert zero_pos, "fixture must contain low-rated-only users"
    known = [int(u) for u in prefix.users if len(prefix.positives(u))][0]
    r = svc.recommend(known, 10)
    assert r["requested_model"] == "default" and r["resolved_model"] == "item_knn" and r["schema_version"] == "1.0"
    assert r["returned_count"] == len(r["recommendations"]) == 10
    assert [x["rank"] for x in r["recommendations"]] == list(range(1, 11))
    seen = set(prefix.seen(known).tolist())
    ids = [x["movie_id"] for x in r["recommendations"]]
    assert not (set(ids) & seen) and len(set(ids)) == len(ids)
    assert all(isinstance(x["movie_id"], int) and math.isfinite(x["score"]) for x in r["recommendations"])
    for u in zero_pos[:3] + [10**7]:
        for m in S.METHODS:
            res = svc.recommend(u, 20, m)
            if m != "popularity":
                assert res["fallback_used"] is True
            cands = prefix.candidates(u)
            pop = fx["ref"]["popularity"]
            assert [x["movie_id"] for x in res["recommendations"]] == pop.rank(u, cands)[:20].tolist()
            assert [x["score"] for x in res["recommendations"]] == pop.pop_of(np.array([x["movie_id"] for x in res["recommendations"]])).astype(float).tolist()
            assert res["score_semantics"] == "popularity_positive_count"
    # low-rated-seen movies are excluded even though never positive
    u = zero_pos[0]
    assert len(prefix.seen(u)) > 0
    ids = {x["movie_id"] for x in svc.recommend(u, 1000, "popularity")["recommendations"]}
    assert not (ids & set(prefix.seen(u).tolist()))
    assert svc.recommend(10**7, 10**6)["returned_count"] == svc.catalog_size  # short list: all catalog, no fabrication

def test_input_validation(fx):
    svc = fx["svc"]
    for bad in (True, "1", 1.5, None, [1]):
        with pytest.raises(InvalidRequestError):
            svc.recommend(bad, 10)
    for bad in (0, -3, True, 2.0, "5", None):
        with pytest.raises(InvalidRequestError):
            svc.recommend(1, bad)
    for bad in ("ALS", "", None, "knn", 3):
        with pytest.raises(InvalidRequestError):
            svc.recommend(1, 10, bad)
    assert svc.recommend(np.int64(1), np.int64(3))["top_k"] == 3
    assert svc.recommend(1) == svc.recommend(1, 10, "default")  # deterministic & defaults

def test_empty_candidates_and_ties_and_negative_scores(tmp_path):
    """Synthetic oracle bundle: user 1 saw (4,5,2,1) of a 4-item catalog; user 2 positive on all but one."""
    rows = []
    t = 1000
    def add(u, m, r):
        nonlocal t
        rows.append((u, m, r, t)); t += 1
    for m in (10, 20, 30, 40):
        add(1, m, 5 if m in (10, 20) else 2)  # user 1 observed everything (empty candidates)
    for m in (10, 20, 30):
        add(2, m, 5)  # user 2 -> one candidate (40)
    for m in (10, 20):
        add(3, m, 4)
    add(3, 30, 1)
    add(4, 10, 5)
    add(4, 20, 5)
    add(4, 40, 4)
    # pad so global split has train+validation prefix containing all of the above
    for i in range(10):
        add(50 + i, 10, 3)
    ratings = tmp_path / "raw"
    ratings.mkdir()
    (ratings / "ratings.dat").write_text("".join(f"{u}::{m}::{r}::{ts}\n" for u, m, r, ts in rows))
    # build service straight from a Prefix with the R3 serialisation helpers (prefix covers all rows)
    prefix = Prefix(np.array(rows, dtype=np.int64))
    pop = Popularity().fit(prefix)
    knn = ItemKNN().fit(prefix)
    from personalized_recommender.models import ALSConfig, BPRConfig

    als = LatentFactorRanker(ALSConfig("ALS-01", 32, 10.0)).fit(prefix)
    bpr = LatentFactorRanker(BPRConfig("BPR-02", 32, 0.05)).fit(prefix)
    # force legitimately negative latent scores for user 4 and an exact score tie between items
    als.item_factors[:] = 0.0
    als.item_factors[:, 0] = np.array([-1.0, -1.0, -2.0, -1.0])[: len(prefix.catalog)]
    als.user_factors[:] = 0.0
    als.user_factors[:, 0] = 1.0
    arrays = S._state_arrays(prefix, knn, pop, als, bpr)
    d = tmp_path / "b"
    d.mkdir()
    files = {}
    for fname, arrs in arrays.items():
        np.savez(d / fname, **arrs)
        files[fname] = {"sha256": S._sha_file(d / fname), "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrs.items()}}
    # reuse a valid fixture manifest skeleton
    base = tmp_path / "skel"
    prepare(PKG_FIXTURE, base / "art")
    man, _ = S.export_bundle(base / "art", base / "bundle")
    man = {k: v for k, v in man.items() if k != "reload_parity"}
    man["files"] = files
    man["serving"].update(catalog_count=len(prefix.catalog), history_user_count=len(prefix.users), positive_user_count=len(als.matrix.user_ids))
    man["prefix"].update(events=len(prefix.events), positive_events=len(prefix.positive_events))
    man["manifest_sha256"] = S.manifest_self_hash(man)
    (d / "manifest.json").write_text(canonical_json(man))
    svc = S.RecommenderService.load(d)
    # empty candidate list -> empty recommendations, no error
    for m in S.METHODS:
        r = svc.recommend(1, 10, m)
        assert r["returned_count"] == 0 and r["recommendations"] == []
    # one candidate: top_k larger than candidates returns just that one
    assert [x["movie_id"] for x in svc.recommend(2, 10, "popularity")["recommendations"]] == [40]
    # ALS negative scores: ranked by score DESC then ID ASC, no fallback
    r = svc.recommend(2, 10, "als")
    assert r["fallback_used"] is False and r["recommendations"][0]["score"] < 0
    r = svc.recommend(3, 10, "als")  # candidates {40}? user3 saw 10,20,30 -> only 40
    assert [x["movie_id"] for x in r["recommendations"]] == [40] and r["recommendations"][0]["score"] == -1.0
    # popularity tie -> numeric ID ASC
    r = svc.recommend(10**6, 4, "popularity")["recommendations"]
    pairs = [(-x["score"], x["movie_id"]) for x in r]
    assert pairs == sorted(pairs)
    # nonfinite score in memory is an error, not a fallback
    svc._lat["als"][1][:, :] = np.nan
    with pytest.raises(ModelIntegrityError):
        svc.recommend(2, 10, "als")

# ----------------- reload parity

def test_fixture_same_fit_parity(fx):
    rep = S.verify_reload_parity(fx["svc"], fx["ref"])
    assert rep["status"] == "PASS"
    for m in S.METHODS:
        assert rep["methods"][m]["mismatches"] == 0 and rep["methods"][m]["comparisons"] > 0
    assert fx["manifest"]["reload_parity"]["status"] == "PASS"

def test_parity_detects_difference(fx):
    ref = dict(fx["ref"])
    bad = copy.copy(ref["bpr"])
    bad.item_factors = ref["bpr"].item_factors * -1.0
    ref["bpr"] = bad
    with pytest.raises(ModelIntegrityError):
        S.verify_reload_parity(fx["svc"], ref)

def test_determinism(fx):
    a = [fx["svc"].recommend(u, 20, m) for u in (1, 2, 9999) for m in S.METHODS]
    b = [fx["svc"].recommend(u, 20, m) for u in (1, 2, 9999) for m in S.METHODS]
    assert a == b

# ----------------- CLI

def test_cli_three_commands_and_exit_codes(fx, tmp_path):
    out = tmp_path / "bundle"
    r = _cli("export", "--artifacts", fx["art"], "--output-dir", out)
    assert r.returncode == 0, r.stderr
    j = json.loads(r.stdout)
    assert j["status"] == "exported" and j["reload_parity"]["status"] == "PASS" and j["dataset_kind"] == "synthetic_fixture"
    assert _cli("export", "--artifacts", fx["art"], "--output-dir", out).returncode == 2  # collision
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    moved = tmp_path / "moved"
    shutil.move(str(out), moved)
    for m in ("default", "popularity", "item_knn", "als", "bpr"):
        r = _cli("recommend", "--bundle", moved, "--user-id", 1, "--top-k", 5, "--model", m, cwd=elsewhere)
        assert r.returncode == 0 and r.stderr == ""
        j = json.loads(r.stdout)  # exactly one JSON document on stdout
        assert j["returned_count"] == 5 and j["resolved_model"] == ("item_knn" if m == "default" else m)
    j = json.loads(_cli("recommend", "--bundle", moved, "--user-id", 1, cwd=elsewhere).stdout)
    assert j["top_k"] == 10 and j["requested_model"] == "default"
    for args in (["--user-id", "abc"], ["--user-id", 1, "--top-k", 0], ["--user-id", 1, "--top-k", -1],
                 ["--user-id", 1, "--model", "xyz"], ["--user-id", 1, "--top-k", "x"], []):
        r = _cli("recommend", "--bundle", moved, *args)
        assert r.returncode == 2 and r.stdout == ""
    assert _cli("recommend", "--bundle", tmp_path / "nope", "--user-id", 1).returncode == 2
    bad = tmp_path / "bad"
    shutil.copytree(moved, bad)
    (bad / "als.npz").write_bytes(b"corrupt")
    r = _cli("recommend", "--bundle", bad, "--user-id", 1)
    assert r.returncode == 2 and r.stdout == "" and "BUNDLE_CHECKSUM_MISMATCH" in r.stderr
    assert _cli("latency", "--bundle", bad).returncode == 2
    rep = tmp_path / "lat.json"
    r = _cli("latency", "--bundle", moved, "--output", rep)
    assert r.returncode == 0 and json.loads(r.stdout) == json.loads(rep.read_text())
    assert _cli("serve").returncode == 2

def test_internal_failure_is_exit_1(fx, tmp_path, monkeypatch):
    import personalized_recommender.cli as C

    def boom(*a, **k):
        raise ModelIntegrityError("NONFINITE_SCORE", "x")

    monkeypatch.setattr(S.RecommenderService, "recommend", boom)
    assert C.main(["recommend", "--bundle", str(fx["bundle"]), "--user-id", "1"]) == 1
    def boom2(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(S.RecommenderService, "recommend", boom2)
    assert C.main(["recommend", "--bundle", str(fx["bundle"]), "--user-id", "1"]) == 1

def test_nine_subcommands():
    from personalized_recommender.cli import build_parser

    sub = [a for a in build_parser()._actions if a.dest == "command"][0].choices
    assert sorted(sub) == sorted(["prepare", "validate", "evaluate", "smoke", "train", "benchmark", "export", "recommend", "latency"])

# ----------------- latency

def test_latency_protocol_oracle(fx, monkeypatch):
    calls = []
    real = S.RecommenderService.recommend

    def spy(self, user_id, top_k=10, model="default"):
        calls.append((user_id, top_k, model))
        return real(self, user_id, top_k, model)

    monkeypatch.setattr(S.RecommenderService, "recommend", spy)
    rep = S.run_latency(fx["svc"], warmup=100, measured=1000)
    assert len(calls) == 1100 and rep["warmup_count"] == 100 and rep["measured_count"] == 1000
    cohort = [int(u) for u in fx["svc"].user_ids_with_positives()[:100]]
    assert rep["cohort_size"] == len(cohort) and cohort == sorted(cohort)
    assert [c[0] for c in calls[:100]] == [cohort[i % len(cohort)] for i in range(100)]
    assert [c[0] for c in calls[100:]] == [cohort[i % len(cohort)] for i in range(1000)]
    assert all(c[1:] == (10, "item_knn") for c in calls)
    assert rep["min_ms"] <= rep["p50_ms"] <= rep["p95_ms"] <= rep["p99_ms"] <= rep["max_ms"]
    assert rep["min_ms"] <= rep["mean_ms"] <= rep["max_ms"] and rep["min_ms"] >= 0
    assert rep["concurrency"] == 1 and rep["percentile_definition"].startswith("numpy.percentile")
    for k in ("cpu_model", "os", "logical_cpus", "python", "numpy"):
        assert rep["environment"][k]
    assert "JSON serialization" in rep["timing_excludes"]

def test_percentile_oracle(fx, monkeypatch):
    seq = iter(range(0, 10**9, 1))
    vals = [0.001 * ((i * 7919) % 1000) for i in range(1000)]
    ticks = []
    for v in vals:
        ticks += [0, int(v * 1e6)]
    it = iter(ticks)
    warm = [0] * 0
    real = S.time.perf_counter_ns
    state = {"n": 0}

    def fake():
        state["n"] += 1
        return next(it)

    monkeypatch.setattr(S.time, "perf_counter_ns", fake)
    rep = S.run_latency(fx["svc"], warmup=3, measured=1000)
    arr = np.array([int(v * 1e6) / 1e6 for v in vals])
    for q in (50, 95, 99):
        assert rep[f"p{q}_ms"] == pytest.approx(float(np.percentile(arr, q, method="linear")), abs=1e-12)
    assert rep["mean_ms"] == pytest.approx(arr.mean()) and rep["max_ms"] == arr.max()

# ----------------- no TEST access

def test_legacy_test_seal_and_no_search(fx, tmp_path):
    r = _cli("evaluate", "--artifacts", fx["art"], "--model", "popularity", "--split", "test")
    assert r.returncode == 2 and "TEST_HOLDOUT_SEALED" in r.stderr
    r = _cli("benchmark", "--artifacts", fx["art"], "--experiment-dir", tmp_path / "none", "--output", tmp_path / "o.json")
    assert r.returncode == 2 and not (tmp_path / "o.json").exists()
    import inspect

    src = inspect.getsource(S)
    assert "run_train" not in src and "select_config" not in src and "ALS_GRID[0]" in src and "BPR_GRID[1]" in src

# ----------------------------------------------------------------------------- REAL official ML-1M

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    d, a = os.environ.get("P02_ML1M_DIR"), os.environ.get("P02_ML1M_ARCHIVE")
    if not d or not a:
        pytest.skip("requires official MovieLens-1M (not shipped): set P02_ML1M_DIR and P02_ML1M_ARCHIVE")
    root = tmp_path_factory.mktemp("r3real")
    r = _cli("prepare", "--data-dir", d, "--output-dir", root / "art", "--archive", a)
    assert r.returncode == 0, r.stderr
    r = _cli("export", "--artifacts", root / "art", "--output-dir", root / "bundle")
    assert r.returncode == 0, r.stderr
    return {"root": root, "export": json.loads(r.stdout)}

def test_real_export_and_lineage(real):
    e = real["export"]
    assert e["dataset_kind"] == "official_movielens_1m"
    m = json.loads((real["root"] / "bundle" / "manifest.json").read_text())
    assert m["lineage"]["r1_manifest_fingerprint"] == S.ACCEPTED_R1_MANIFEST_FINGERPRINT
    assert m["lineage"]["split_fingerprint"] == S.ACCEPTED_SPLIT_FINGERPRINT
    assert m["prefix"]["events"] == 900188 and m["prefix"]["max_timestamp"] < 978133414
    assert m["serving"]["catalog_count"] == 3678 and m["serving"]["history_user_count"] == 6011  # 6040 minus the 29 users with no pre-TEST rating
    assert set(m["files"]) == set(S.DATA_FILES)

def test_real_t05_same_fit_parity(real):
    rep = real["export"]["reload_parity"]
    assert rep["status"] == "PASS"
    for m in S.METHODS:
        assert rep["methods"][m]["mismatches"] == 0
        assert rep["methods"][m]["max_abs_score_diff"] <= 1e-9

def test_real_t06_copied_bundle_without_data(real, tmp_path):
    moved = tmp_path / "copied"
    shutil.copytree(real["root"] / "bundle", moved)
    env = {k: v for k, v in os.environ.items() if not k.startswith("P02_")}
    for m in ("default", "popularity", "item_knn", "als", "bpr"):
        r = _cli("recommend", "--bundle", moved, "--user-id", 1, "--top-k", 10, "--model", m, cwd=tmp_path, env=env)
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["returned_count"] == 10
    assert json.loads(_cli("recommend", "--bundle", moved, "--user-id", 999999, cwd=tmp_path, env=env).stdout)["fallback_used"] is True

def test_real_t07_latency_1000(real, tmp_path):
    out = tmp_path / "lat.json"
    r = _cli("latency", "--bundle", real["root"] / "bundle", "--output", out)
    assert r.returncode == 0, r.stderr
    j = json.loads(out.read_text())
    assert j["warmup_count"] == 100 and j["measured_count"] == 1000 and j["cohort_size"] == 100
    assert j["dataset_kind"] == "official_movielens_1m" and j["method"] == "item_knn" and j["top_k"] == 10

"""Deterministic replay, fingerprints, mismatch rejection, sealed TEST holdout;
CLI end-to-end (synthetic offline smoke + real ML-1M when P02_ML1M_DIR is set).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PKG_FIXTURE

from personalized_recommender.artifacts import load_partitions, prepare, validate
from personalized_recommender.cli import main
from personalized_recommender.errors import ArtifactError



def _cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "personalized_recommender", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_replay_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    prepare(PKG_FIXTURE, a)
    prepare(PKG_FIXTURE, b)
    for f in ("manifest.json", "train.npy", "validation.npy", "test.npy"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    va, vb = validate(a), validate(b)
    assert va["split_fingerprint"] == vb["split_fingerprint"]
    ra = _cli("evaluate", "--artifacts", a, "--model", "item_knn", "--split", "validation", "--output", tmp_path / "ra.json")
    rb = _cli("evaluate", "--artifacts", b, "--model", "item_knn", "--split", "validation", "--output", tmp_path / "rb.json")
    assert ra.returncode == rb.returncode == 0
    assert (tmp_path / "ra.json").read_bytes() == (tmp_path / "rb.json").read_bytes()


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: (d / "validation.npy").unlink(), "MISSING_ARTIFACT"),
        (lambda d: (d / "manifest.json").unlink(), "MISSING_ARTIFACT"),
        (lambda d: (d / "manifest.json").write_text("{not json"), "CORRUPT_ARTIFACT"),
        (lambda d: _flip(d / "test.npy"), "CHECKSUM_MISMATCH"),
        (lambda d: _edit_manifest(d, lambda m: m.update(protocol_version="P02-R1-ACCEPTANCE-v0.9")), "PROTOCOL_MISMATCH"),
        (lambda d: _edit_manifest(d, lambda m: m["split"].update(t_val=m["split"]["t_val"] + 1)), "MANIFEST_FINGERPRINT_MISMATCH"),
        (lambda d: _swap_partitions(d), "SPLIT_INTEGRITY"),
    ],
)
def test_mismatch_rejected(tmp_path, mutate, code):
    d = tmp_path / "art"
    prepare(PKG_FIXTURE, d)
    mutate(d)
    with pytest.raises(ArtifactError) as e:
        load_partitions(d)
    assert e.value.code == code
    r = _cli("validate", "--artifacts", d)
    assert r.returncode == 2 and code in r.stderr
    r = _cli("evaluate", "--artifacts", d, "--model", "popularity", "--split", "validation")
    assert r.returncode == 2 and r.stdout == ""


def _flip(p: Path):
    b = bytearray(p.read_bytes())
    b[-3] ^= 0xFF
    p.write_bytes(bytes(b))


def _edit_manifest(d: Path, fn):
    m = json.loads((d / "manifest.json").read_text())
    fn(m)
    (d / "manifest.json").write_text(json.dumps(m))


def _swap_partitions(d: Path):
    """Consistently re-fingerprinted but rule-violating artifact: move a TRAIN row to VALIDATION."""
    import io

    import numpy as np

    from personalized_recommender.artifacts import canonical_json, sha256_bytes, split_fingerprint
    from personalized_recommender.split import events_fingerprint

    tr = np.load(d / "train.npy")
    va = np.load(d / "validation.npy")
    va = np.concatenate([va, tr[:1]])
    tr = tr[1:]
    m = json.loads((d / "manifest.json").read_text())
    for name, arr in (("train", tr), ("validation", va)):
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(arr, dtype="<i8"))
        (d / f"{name}.npy").write_bytes(buf.getvalue())
        m["files"][name].update(rows=len(arr), sha256=sha256_bytes(buf.getvalue()), content_fingerprint=events_fingerprint(arr))
    fps = {p: m["files"][p]["content_fingerprint"] for p in ("train", "validation", "test")}
    m["split"]["split_fingerprint"] = split_fingerprint(m["split"]["t_val"], m["split"]["t_test"], fps)
    del m["manifest_fingerprint"]
    m["manifest_fingerprint"] = sha256_bytes(canonical_json(m).encode())
    (d / "manifest.json").write_text(canonical_json(m))


def test_test_split_sealed(tmp_path, capsys):
    d = tmp_path / "art"
    prepare(PKG_FIXTURE, d)
    for model in ("popularity", "item_knn"):
        r = _cli("evaluate", "--artifacts", d, "--model", model, "--split", "test", "--output", tmp_path / "t.json")
        assert r.returncode == 2
        assert "TEST_HOLDOUT_SEALED" in r.stderr
        assert r.stdout == "" and not (tmp_path / "t.json").exists()
    # sealed even if artifacts are missing: rejected as a phase error before loading anything
    assert main(["evaluate", "--artifacts", str(tmp_path / "none"), "--model", "popularity", "--split", "test"]) == 2


def test_validation_report_contains_no_test_metrics(tmp_path):
    d = tmp_path / "art"
    prepare(PKG_FIXTURE, d)
    r = _cli("evaluate", "--artifacts", d, "--model", "popularity", "--split", "validation")
    rep = json.loads(r.stdout)
    assert rep["stage"] == "validation" and rep["protocol_version"] == "P02-R1-ACCEPTANCE-v1.0"
    assert "test" not in json.dumps(rep["metrics_by_k"]) and "test_sha256" not in rep["split_identity"]
    assert set(rep["metrics_by_k"]) == {"5", "10", "20"}


def test_exit_codes(tmp_path):
    assert _cli().returncode == 2
    assert _cli("evaluate", "--artifacts", tmp_path, "--model", "als", "--split", "validation").returncode == 2
    assert _cli("prepare", "--data-dir", tmp_path / "missing", "--output-dir", tmp_path / "o").returncode == 2
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "ratings.dat").write_text("1::2::9::100\n")
    r = _cli("prepare", "--data-dir", tmp_path / "bad", "--output-dir", tmp_path / "o")
    assert r.returncode == 2 and "INVALID_RATING" in r.stderr


def test_internal_error_exit_1(monkeypatch):
    import personalized_recommender.cli as cli

    def boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "run_smoke", boom)
    assert cli.main(["smoke"]) == 1


def test_smoke_offline_synthetic(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("P02_ML1M_DIR", "P02_ML1M_ARCHIVE")}
    env.update(http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9", no_proxy="")
    r = subprocess.run([sys.executable, "-m", "personalized_recommender", "smoke"], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["smoke"] == "PASS" and out["test_evaluation_rejected"] is True



def test_real_ml1m_cli_workflow(ml1m_dir, tmp_path):
    art = tmp_path / "art"
    args = ["prepare", "--data-dir", ml1m_dir, "--output-dir", art]
    if os.environ.get("P02_ML1M_ARCHIVE"):
        args += ["--archive", os.environ["P02_ML1M_ARCHIVE"]]
    r = _cli(*args)
    assert r.returncode == 0, r.stderr
    prep = json.loads(r.stdout)
    assert prep["dataset"]["official"]["official_reference_match"] is True
    assert prep["dataset"]["counts"]["n_ratings"] == 1_000_209 and prep["dataset"]["counts"]["n_users"] == 6_040
    assert sum(prep["split"]["counts"].values()) == 1_000_209
    r = _cli("validate", "--artifacts", art)
    assert r.returncode == 0, r.stderr
    reports = {}
    for model in ("popularity", "item_knn"):
        out = tmp_path / f"{model}.json"
        r = _cli("evaluate", "--artifacts", art, "--model", model, "--split", "validation", "--output", out)
        assert r.returncode == 0, r.stderr
        reports[model] = json.loads(out.read_text())
        rep = reports[model]
        assert rep["stage"] == "validation" and set(rep["metrics_by_k"]) == {"5", "10", "20"}
        assert rep["eligible_users_count"] > 0
        for k in ("5", "10", "20"):
            for v in rep["metrics_by_k"][k].values():
                assert 0.0 <= v <= 1.0
    assert reports["popularity"]["diagnostics"] == reports["item_knn"]["diagnostics"]
    r = _cli("evaluate", "--artifacts", art, "--model", "item_knn", "--split", "test")
    assert r.returncode == 2 and r.stdout == ""
    # deterministic replay on the real data
    art2 = tmp_path / "art2"
    assert _cli("prepare", "--data-dir", ml1m_dir, "--output-dir", art2).returncode == 0
    for f in ("train.npy", "validation.npy", "test.npy"):
        assert (art / f).read_bytes() == (art2 / f).read_bytes()
    assert validate(art)["split_fingerprint"] == validate(art2)["split_fingerprint"]
    shutil.rmtree(art2)

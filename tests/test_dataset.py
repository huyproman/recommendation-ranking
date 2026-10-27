"""Dataset parsing / provenance / malformed input / cache identity."""

import hashlib
import shutil
import zipfile

import numpy as np
import pytest
from conftest import write_ratings

from personalized_recommender import data
from personalized_recommender.artifacts import prepare
from personalized_recommender.errors import ArtifactError, DataError

GOOD = [(1, 10, 5, 100), (1, 20, 3, 200), (2, 10, 4, 150)]


def _raw(text: str) -> bytes:
    return text.encode()


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "EMPTY_INPUT"),
        ("1::10::5::100", "TRUNCATED_INPUT"),  # no trailing newline
        ("1::10::5::100\n1::20::3\n", "MALFORMED_RECORD"),  # 3 fields
        ("1::10::5::100::7\n", "MALFORMED_RECORD"),  # 5 fields
        ("1:10::5::100\n", "MALFORMED_RECORD"),  # bad separator
        ("1,10,5,100\n", "MALFORMED_RECORD"),
        ("1::10::5::100\n\n", "MALFORMED_RECORD"),  # blank line
        ("1::1a::5::100\n", "INVALID_NUMERIC_FIELD"),
        ("1::10::4.5::100\n", "INVALID_NUMERIC_FIELD"),
        ("1:: 10::5::100\n", "INVALID_NUMERIC_FIELD"),
        ("01::10::5::100\n", "INVALID_NUMERIC_FIELD"),  # leading zero not coerced
        ("0::10::5::100\n", "INVALID_ID"),
        ("1::10::0::100\n", "INVALID_RATING"),
        ("1::10::6::100\n", "INVALID_RATING"),
        ("1::10::5::-5\n", "INVALID_TIMESTAMP"),
        ("1::10::5::0\n", "INVALID_TIMESTAMP"),
        ("1::10::5::99999999999\n", "INVALID_TIMESTAMP"),
        ("1::10::5::100\r\n", "INVALID_NUMERIC_FIELD"),
        ("1::10::5::100\n2::3::4::5\n1::10::2::300\n", "DUPLICATE_PAIR"),
    ],
)
def test_malformed_rejected(text, code):
    with pytest.raises(DataError) as e:
        data.parse_ratings_bytes(_raw(text))
    assert e.value.code == code


def test_duplicate_not_silently_deduplicated():
    with pytest.raises(DataError, match="lines 1 and 3"):
        data.parse_ratings_bytes(b"5::7::4::10\n6::7::4::11\n5::7::4::10\n")


def test_valid_parse_canonical_order_and_exact_values():
    ev = data.parse_ratings_bytes(b"2::10::4::150\n1::20::3::200\n1::10::5::100\n3::9::1::150\n")
    assert ev.dtype == np.int64
    assert ev.tolist() == [[1, 10, 5, 100], [2, 10, 4, 150], [3, 9, 1, 150], [1, 20, 3, 200]]


def test_missing_ratings_file(tmp_path):
    with pytest.raises(DataError) as e:
        data.load_dataset(tmp_path)
    assert e.value.code == "MISSING_RATINGS"
    with pytest.raises(DataError) as e:
        data.load_dataset(tmp_path / "nope")
    assert e.value.code == "MISSING_DATA_DIR"


def test_missing_movie_reference(tmp_path):
    d = write_ratings(tmp_path, GOOD)
    (d / "movies.dat").write_text("10::A (1990)::Drama\n")
    with pytest.raises(DataError) as e:
        data.load_dataset(d)
    assert e.value.code == "MISSING_MOVIE_REFERENCE"


def test_provenance_hash_from_real_bytes(tmp_path):
    d = write_ratings(tmp_path, GOOD)
    _, prov = data.load_dataset(d)
    raw = (d / "ratings.dat").read_bytes()
    assert prov["ratings_file"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert prov["ratings_file"]["size_bytes"] == len(raw)
    assert prov["counts"]["n_ratings"] == 3 and prov["counts"]["n_users"] == 2
    assert prov["counts"]["n_movies_observed"] == 2
    assert (prov["counts"]["min_timestamp"], prov["counts"]["max_timestamp"]) == (100, 200)


def test_archive_cross_check(tmp_path):
    d = write_ratings(tmp_path / "d", GOOD)
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("ml-1m/ratings.dat", (d / "ratings.dat").read_bytes())
    sha = hashlib.sha256((d / "ratings.dat").read_bytes()).hexdigest()
    info = data.verify_archive(z, sha)
    assert info["sha256"] == hashlib.sha256(z.read_bytes()).hexdigest()
    with pytest.raises(DataError) as e:
        data.verify_archive(z, "0" * 64)
    assert e.value.code == "ARCHIVE_RATINGS_MISMATCH"
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DataError) as e:
        data.verify_archive(bad, sha)
    assert e.value.code == "CORRUPT_ARCHIVE"


def test_cache_identity_mismatch_rejected(tmp_path):
    rows = [(u, m, 1 + (u * m) % 5, 1000 + 10 * i) for i, (u, m) in enumerate((u, m) for u in range(1, 6) for m in range(1, 5))]
    d = write_ratings(tmp_path / "d", rows)
    art = tmp_path / "art"
    assert prepare(d, art)["status"] == "prepared"
    assert prepare(d, art)["status"] == "cache_verified"  # identical bytes -> verified reuse
    rows2 = list(rows)
    rows2[0] = (rows[0][0], rows[0][1], 5 if rows[0][2] != 5 else 4, rows[0][3])  # modified data
    d2 = write_ratings(tmp_path / "d2", rows2)
    with pytest.raises(ArtifactError) as e:
        prepare(d2, art)
    assert e.value.code == "CACHE_IDENTITY_MISMATCH"
    # corrupt cached artifact is detected, not regenerated
    p = art / "train.npy"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(ArtifactError) as e:
        prepare(d, art)
    assert e.value.code == "CHECKSUM_MISMATCH"


# ---------------- real official MovieLens-1M ----------------


def test_real_official_ml1m_provenance(ml1m_dir):
    events, prov = data.load_dataset(ml1m_dir)
    assert prov["ratings_file"]["sha256"] == data.sha256_file(ml1m_dir / "ratings.dat")
    assert prov["ratings_file"]["sha256"] == data.OFFICIAL_RATINGS_SHA256
    c = prov["counts"]
    assert c["n_ratings"] == 1_000_209
    assert c["n_users"] == 6_040
    assert c["n_movies_observed"] == len(np.unique(events[:, data.MOVIE]))
    assert set(np.unique(events[:, data.RATING]).tolist()) == {1, 2, 3, 4, 5}
    assert data.official_check({**prov, "archive": None})["official_reference_match"] is True
    assert prov["movies_file"] is not None  # reference integrity check executed


def test_real_official_archive(ml1m_dir, ml1m_archive):
    info = data.verify_archive(ml1m_archive, data.sha256_file(ml1m_dir / "ratings.dat"))
    assert info["sha256"] == data.OFFICIAL_ARCHIVE_SHA256
    assert info["md5"] == data.OFFICIAL_ARCHIVE_MD5


def test_real_modified_copy_rejected_by_official_check(ml1m_dir, tmp_path):
    d = tmp_path / "mod"
    d.mkdir()
    raw = (ml1m_dir / "ratings.dat").read_bytes()
    lines = raw.split(b"\n")
    u, m, r, t = lines[0].split(b"::")
    lines[0] = b"::".join([u, m, b"1" if r != b"1" else b"2", t])  # one modified rating
    (d / "ratings.dat").write_bytes(b"\n".join(lines))
    _, prov = data.load_dataset(d)
    assert data.official_check({**prov, "archive": None})["official_reference_match"] is False
    shutil.rmtree(d)

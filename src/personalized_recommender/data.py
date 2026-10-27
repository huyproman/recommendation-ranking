"""Strict MovieLens-1M ratings ingestion and provenance hashing.

Events are represented as an ``int64`` array of shape ``(N, 4)`` with columns
``(user_id, movie_id, rating, timestamp)`` using raw numeric GroupLens IDs, in the
canonical order ``(timestamp ASC, user_id ASC, movie_id ASC)``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .errors import DataError

USER, MOVIE, RATING, TS = 0, 1, 2, 3

OFFICIAL_DATASET_URL = "https://grouplens.org/datasets/movielens/1m/"
OFFICIAL_ARCHIVE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
OFFICIAL_README_URL = "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"
DATASET_VERSION = "MovieLens 1M (GroupLens stable benchmark, released 2/2003)"

# Reference identity of the official GroupLens archive. These values were computed
# from the real bytes downloaded from OFFICIAL_ARCHIVE_URL; the archive MD5 matches
# the GroupLens-published https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5.
OFFICIAL_ARCHIVE_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
OFFICIAL_ARCHIVE_MD5 = "c4d9eecfca2ab87c1945afe126590906"
OFFICIAL_RATINGS_SHA256 = "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"
OFFICIAL_RATINGS_SIZE = 24594131
OFFICIAL_MOVIES_SHA256 = "0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9"
OFFICIAL_N_RATINGS = 1_000_209
OFFICIAL_N_USERS = 6_040

MAX_ID = 2**31 - 1
MIN_TS = 1
MAX_TS = 2**31 - 1

PARSER_POLICY = (
    "strict: ASCII; every line must match '^UserID::MovieID::Rating::Timestamp\\n$' with positive "
    "integer IDs (no sign/leading zeros, <= 2^31-1), integer rating in 1..5, integer Unix-seconds "
    "timestamp in [1, 2^31-1]; file must be non-empty and newline-terminated; duplicate "
    "(user_id, movie_id) pairs rejected; if movies.dat is present every rated movie must be listed. "
    "No record is skipped, deduplicated or coerced."
)

_LINE = re.compile(rb"^(0|[1-9][0-9]*)::(0|[1-9][0-9]*)::([0-9]+)::(-?[0-9]+)$")
_LOOSE = re.compile(rb"^[^:]*::[^:]*::[^:]*::[^:]*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sort(events: np.ndarray) -> np.ndarray:
    """Sort events by (timestamp ASC, user_id ASC, movie_id ASC)."""
    if len(events) == 0:
        return events.reshape(0, 4).astype(np.int64)
    order = np.lexsort((events[:, MOVIE], events[:, USER], events[:, TS]))
    return np.ascontiguousarray(events[order], dtype=np.int64)


def parse_ratings_bytes(raw: bytes, source: str = "ratings.dat") -> np.ndarray:
    """Strictly parse ratings bytes. Returns canonically sorted int64 (N,4) events."""
    if len(raw) == 0:
        raise DataError("EMPTY_INPUT", f"{source} is empty")
    if not raw.endswith(b"\n"):
        raise DataError("TRUNCATED_INPUT", f"{source} does not end with a newline (truncated/corrupt)")
    lines = raw[:-1].split(b"\n")
    n = len(lines)
    out = np.empty((n, 4), dtype=np.int64)
    for i, line in enumerate(lines):
        lineno = i + 1
        m = _LINE.match(line)
        if m is None:
            if not line.isascii():
                raise DataError("MALFORMED_RECORD", f"{source}:{lineno}: non-ASCII bytes")
            if not _LOOSE.match(line):
                raise DataError(
                    "MALFORMED_RECORD",
                    f"{source}:{lineno}: expected 4 '::'-separated fields, got {line[:80]!r}",
                )
            raise DataError("INVALID_NUMERIC_FIELD", f"{source}:{lineno}: non-integer field in {line[:80]!r}")
        u, mv, r, t = (int(x) for x in m.groups())
        if not (1 <= u <= MAX_ID) or not (1 <= mv <= MAX_ID):
            raise DataError("INVALID_ID", f"{source}:{lineno}: ids must be in [1, 2^31-1]: {line!r}")
        if not (1 <= r <= 5):
            raise DataError("INVALID_RATING", f"{source}:{lineno}: rating {r} outside 1..5")
        if not (MIN_TS <= t <= MAX_TS):
            raise DataError("INVALID_TIMESTAMP", f"{source}:{lineno}: timestamp {t} outside [1, 2^31-1]")
        out[i, 0] = u
        out[i, 1] = mv
        out[i, 2] = r
        out[i, 3] = t
    # Duplicate (user_id, movie_id) detection: never silently deduplicate.
    order = np.lexsort((out[:, MOVIE], out[:, USER]))
    s = out[order]
    dup = np.nonzero((s[1:, USER] == s[:-1, USER]) & (s[1:, MOVIE] == s[:-1, MOVIE]))[0]
    if len(dup):
        a, b = sorted((int(order[dup[0]]) + 1, int(order[dup[0] + 1]) + 1))
        raise DataError(
            "DUPLICATE_PAIR",
            f"{source}: duplicate (user_id={int(s[dup[0], USER])}, movie_id={int(s[dup[0], MOVIE])}) "
            f"at lines {a} and {b}",
        )
    return canonical_sort(out)


def parse_movie_ids(path: Path) -> set[int]:
    """Parse movies.dat (display metadata only). Returns the set of listed movie IDs."""
    raw = path.read_bytes()
    if len(raw) == 0 or not raw.endswith(b"\n"):
        raise DataError("TRUNCATED_INPUT", f"{path.name} is empty or not newline-terminated")
    ids: set[int] = set()
    for i, line in enumerate(raw[:-1].split(b"\n"), start=1):
        parts = line.split(b"::")
        if len(parts) != 3 or not re.fullmatch(rb"[1-9][0-9]*", parts[0]):
            raise DataError("MALFORMED_RECORD", f"{path.name}:{i}: malformed movie record {line[:80]!r}")
        mid = int(parts[0])
        if mid in ids:
            raise DataError("DUPLICATE_MOVIE", f"{path.name}:{i}: duplicate movie_id {mid}")
        ids.add(mid)
    return ids


def load_dataset(data_dir: Path) -> tuple[np.ndarray, dict]:
    """Load and strictly validate a MovieLens-1M style directory.

    Returns (canonically sorted events, provenance dict computed from real bytes).
    """
    data_dir = Path(data_dir)
    rpath = data_dir / "ratings.dat"
    if not data_dir.is_dir():
        raise DataError("MISSING_DATA_DIR", f"data directory not found: {data_dir}")
    if not rpath.is_file():
        raise DataError("MISSING_RATINGS", f"ratings.dat not found in {data_dir}")
    raw = rpath.read_bytes()
    ratings_sha = hashlib.sha256(raw).hexdigest()
    events = parse_ratings_bytes(raw, "ratings.dat")
    prov: dict = {
        "ratings_file": {"name": "ratings.dat", "sha256": ratings_sha, "size_bytes": len(raw)},
        "movies_file": None,
    }
    mpath = data_dir / "movies.dat"
    if mpath.is_file():
        ids = parse_movie_ids(mpath)
        missing = sorted(set(np.unique(events[:, MOVIE]).tolist()) - ids)
        if missing:
            raise DataError(
                "MISSING_MOVIE_REFERENCE",
                f"{len(missing)} rated movie_id(s) absent from movies.dat, first: {missing[:5]}",
            )
        prov["movies_file"] = {
            "name": "movies.dat",
            "sha256": sha256_file(mpath),
            "size_bytes": mpath.stat().st_size,
            "n_movies_listed": len(ids),
            "usage": "reference-integrity check only; not a model feature",
        }
    prov["counts"] = dataset_counts(events)
    prov["parser_policy"] = PARSER_POLICY
    return events, prov


def dataset_counts(events: np.ndarray) -> dict:
    return {
        "n_ratings": int(len(events)),
        "n_users": int(len(np.unique(events[:, USER]))),
        "n_movies_observed": int(len(np.unique(events[:, MOVIE]))),
        "min_timestamp": int(events[:, TS].min()),
        "max_timestamp": int(events[:, TS].max()),
        "rating_histogram": {str(r): int((events[:, RATING] == r).sum()) for r in range(1, 6)},
    }


def verify_archive(archive: Path, ratings_sha256: str) -> dict:
    """Hash a local archive and confirm its ml-1m/ratings.dat is byte-identical to the parsed file."""
    import zipfile

    archive = Path(archive)
    if not archive.is_file():
        raise DataError("MISSING_ARCHIVE", f"archive not found: {archive}")
    try:
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad is not None:
                raise DataError("CORRUPT_ARCHIVE", f"CRC failure in archive member {bad}")
            members = [n for n in z.namelist() if n.endswith("ratings.dat")]
            if len(members) != 1:
                raise DataError("CORRUPT_ARCHIVE", "archive must contain exactly one ratings.dat")
            inner = hashlib.sha256(z.read(members[0])).hexdigest()
    except zipfile.BadZipFile as e:
        raise DataError("CORRUPT_ARCHIVE", f"not a valid zip archive: {e}") from None
    if inner != ratings_sha256:
        raise DataError(
            "ARCHIVE_RATINGS_MISMATCH",
            f"archive ratings.dat sha256 {inner} != data-dir ratings.dat sha256 {ratings_sha256}",
        )
    return {
        "name": archive.name,
        "sha256": sha256_file(archive),
        "md5": md5_file(archive),
        "size_bytes": archive.stat().st_size,
        "inner_ratings_sha256": inner,
    }


def official_check(prov: dict) -> dict:
    """Compare computed identity against the official reference; enforce original count sanity."""
    r = prov["ratings_file"]
    c = prov["counts"]
    match = r["sha256"] == OFFICIAL_RATINGS_SHA256 and r["size_bytes"] == OFFICIAL_RATINGS_SIZE
    if prov.get("archive") is not None:
        match = match and prov["archive"]["sha256"] == OFFICIAL_ARCHIVE_SHA256
    if match and (c["n_ratings"] != OFFICIAL_N_RATINGS or c["n_users"] != OFFICIAL_N_USERS):
        raise DataError("OFFICIAL_COUNT_MISMATCH", "official hash matched but counts differ")
    return {
        "official_reference_match": bool(match),
        "expected_ratings_sha256": OFFICIAL_RATINGS_SHA256,
        "expected_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "expected_archive_md5_published": OFFICIAL_ARCHIVE_MD5,
        "expected_n_ratings": OFFICIAL_N_RATINGS,
        "expected_n_users": OFFICIAL_N_USERS,
        "count_sanity_pass": c["n_ratings"] == OFFICIAL_N_RATINGS and c["n_users"] == OFFICIAL_N_USERS,
    }

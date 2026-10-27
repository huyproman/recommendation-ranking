"""Persisted split artifacts with identity/integrity checking (prepare / validate)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import __version__
from .data import canonical_sort, dataset_counts, load_dataset, official_check, verify_archive
from .errors import ArtifactError
from .split import Split, compute_cutoffs, events_fingerprint, partition_by_cutoffs, temporal_split

# Serialized identifiers (protocol version, "SEALED in R1" status text) are hashed into manifest fingerprints; "R1" means
# the split/evaluation protocol stage. Do not edit without changing artifact identity.
PROTOCOL_VERSION = "P02-R1-ACCEPTANCE-v1.0"
ARTIFACT_SCHEMA = "p02-split-artifacts/1"
MANIFEST = "manifest.json"
PARTITIONS = ("train", "validation", "test")


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype="<i8"), allow_pickle=False)
    return buf.getvalue()


def split_fingerprint(t_val: int, t_test: int, fps: dict[str, str]) -> str:
    return sha256_bytes(canonical_json({"t_val": t_val, "t_test": t_test, "partitions": fps}).encode())


def build_manifest(split: Split, provenance: dict, files: dict[str, dict]) -> dict:
    fps = {p: files[p]["content_fingerprint"] for p in PARTITIONS}
    n = split.n
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "package_version": __version__,
        "dataset": provenance,
        "split": {
            "algorithm": "global temporal; sort (timestamp,user_id,movie_id) ASC; "
            "T_val=r[floor(0.8N)].ts, T_test=r[floor(0.9N)].ts; TRAIN t<T_val, "
            "VALIDATION T_val<=t<T_test, TEST t>=T_test",
            "n_events": n,
            "index_val": split.idx_val,
            "index_test": split.idx_test,
            "t_val": split.t_val,
            "t_test": split.t_test,
            "counts": {p: files[p]["rows"] for p in PARTITIONS},
            "fractions": {p: files[p]["rows"] / n for p in PARTITIONS},
            "split_fingerprint": split_fingerprint(split.t_val, split.t_test, fps),
            "test_partition_status": "SEALED in R1: retained and checksummed for R2; no ranking evaluation",
        },
        "files": files,
    }
    body["manifest_fingerprint"] = sha256_bytes(canonical_json(body).encode())
    return body


def prepare(data_dir: Path, output_dir: Path, archive: Path | None = None) -> dict:
    events, prov = load_dataset(Path(data_dir))
    prov["archive"] = verify_archive(archive, prov["ratings_file"]["sha256"]) if archive else None
    prov["official"] = official_check(prov)
    prov["source"] = {
        "dataset_page": "https://grouplens.org/datasets/movielens/1m/",
        "archive_url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "readme_url": "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt",
        "version": "MovieLens 1M (GroupLens stable benchmark, released 2/2003)",
    }
    split = temporal_split(events)
    output_dir = Path(output_dir)
    blobs = {p: _npy_bytes(getattr(split, p)) for p in PARTITIONS}
    files = {
        p: {
            "file": f"{p}.npy",
            "rows": int(len(getattr(split, p))),
            "sha256": sha256_bytes(blobs[p]),
            "content_fingerprint": events_fingerprint(getattr(split, p)),
        }
        for p in PARTITIONS
    }
    manifest = build_manifest(split, prov, files)
    mpath = output_dir / MANIFEST
    if mpath.exists():
        # Cache reuse: existing artifacts must be intact and identical to what these input bytes produce.
        existing = validate(output_dir)
        if existing["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
            raise ArtifactError(
                "CACHE_IDENTITY_MISMATCH",
                f"{output_dir} holds artifacts with manifest fingerprint {existing['manifest_fingerprint']}, "
                f"but the input data yields {manifest['manifest_fingerprint']}; refusing to overwrite",
            )
        return {"status": "cache_verified", "manifest": manifest}
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in PARTITIONS:
        (output_dir / files[p]["file"]).write_bytes(blobs[p])
    mpath.write_text(canonical_json(manifest))
    return {"status": "prepared", "manifest": manifest}


def _load_manifest(art: Path) -> dict:
    mpath = Path(art) / MANIFEST
    if not mpath.is_file():
        raise ArtifactError("MISSING_ARTIFACT", f"{mpath} not found")
    try:
        m = json.loads(mpath.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtifactError("CORRUPT_ARTIFACT", f"manifest is not valid JSON: {e}") from None
    if not isinstance(m, dict):
        raise ArtifactError("CORRUPT_ARTIFACT", "manifest must be a JSON object")
    return m


def load_partitions(art: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Load and fully verify artifacts. Returns (manifest, partitions)."""
    art = Path(art)
    m = _load_manifest(art)
    if m.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactError("PROTOCOL_MISMATCH", f"protocol_version {m.get('protocol_version')!r} != {PROTOCOL_VERSION!r}")
    if m.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ArtifactError("PROTOCOL_MISMATCH", f"artifact_schema {m.get('artifact_schema')!r} != {ARTIFACT_SCHEMA!r}")
    body = {k: v for k, v in m.items() if k != "manifest_fingerprint"}
    try:
        recomputed = sha256_bytes(canonical_json(body).encode())
    except (TypeError, ValueError) as e:
        raise ArtifactError("CORRUPT_ARTIFACT", f"manifest not canonical-serialisable: {e}") from None
    if recomputed != m.get("manifest_fingerprint"):
        raise ArtifactError("MANIFEST_FINGERPRINT_MISMATCH", "manifest content does not match its fingerprint")
    try:
        files, sp = m["files"], m["split"]
        parts: dict[str, np.ndarray] = {}
        for p in PARTITIONS:
            f = art / files[p]["file"]
            if not f.is_file():
                raise ArtifactError("MISSING_ARTIFACT", f"{f} not found")
            raw = f.read_bytes()
            if sha256_bytes(raw) != files[p]["sha256"]:
                raise ArtifactError("CHECKSUM_MISMATCH", f"{f.name} sha256 does not match manifest")
            try:
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
            except Exception as e:  # noqa: BLE001 - any decode failure is corruption
                raise ArtifactError("CORRUPT_ARTIFACT", f"{f.name} unreadable: {e}") from None
            if arr.dtype != np.dtype("<i8") or arr.ndim != 2 or arr.shape[1] != 4 or len(arr) != files[p]["rows"]:
                raise ArtifactError("CORRUPT_ARTIFACT", f"{f.name} has unexpected dtype/shape {arr.dtype}{arr.shape}")
            if events_fingerprint(arr) != files[p]["content_fingerprint"]:
                raise ArtifactError("CHECKSUM_MISMATCH", f"{f.name} content fingerprint mismatch")
            parts[p] = arr
        # Split integrity: re-derive cutoffs and memberships from the union of partitions.
        union = np.concatenate([parts[p] for p in PARTITIONS])
        if len(union) != sp["n_events"] or dataset_counts(union) != m["dataset"]["counts"]:
            raise ArtifactError("SPLIT_INTEGRITY", "partition union does not match recorded dataset counts")
        su = canonical_sort(union)
        iv, it, tv, tt = compute_cutoffs(su)
        if (iv, it, tv, tt) != (sp["index_val"], sp["index_test"], sp["t_val"], sp["t_test"]):
            raise ArtifactError("SPLIT_INTEGRITY", "recomputed cutoffs differ from manifest")
        re_tr, re_va, re_te = partition_by_cutoffs(su, tv, tt)
        for p, re_ in zip(PARTITIONS, (re_tr, re_va, re_te)):
            if not np.array_equal(re_, parts[p]):
                raise ArtifactError("SPLIT_INTEGRITY", f"{p} membership differs from timestamp-cutoff rule")
        if len(np.unique(su[:, :2], axis=0)) != len(su):
            raise ArtifactError("SPLIT_INTEGRITY", "duplicate (user, movie) across partitions")
        fps = {p: files[p]["content_fingerprint"] for p in PARTITIONS}
        if split_fingerprint(tv, tt, fps) != sp["split_fingerprint"]:
            raise ArtifactError("SPLIT_INTEGRITY", "split fingerprint mismatch")
    except KeyError as e:
        raise ArtifactError("CORRUPT_ARTIFACT", f"manifest missing field {e}") from None
    return m, parts


def validate(art: Path) -> dict:
    m, parts = load_partitions(art)
    return {
        "status": "valid",
        "protocol_version": m["protocol_version"],
        "manifest_fingerprint": m["manifest_fingerprint"],
        "split_fingerprint": m["split"]["split_fingerprint"],
        "ratings_sha256": m["dataset"]["ratings_file"]["sha256"],
        "official_reference_match": m["dataset"]["official"]["official_reference_match"],
        "t_val": m["split"]["t_val"],
        "t_test": m["split"]["t_test"],
        "counts": m["split"]["counts"],
        "checks": [
            "protocol/schema version",
            "manifest fingerprint",
            "per-file sha256 and content fingerprint",
            "dtype/shape",
            "union equals recorded dataset counts",
            "cutoffs re-derived",
            "memberships follow timestamp rule",
            "no duplicate pairs",
            "split fingerprint",
        ],
    }

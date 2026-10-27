"""Versioned four-model serving bundle (``p02-serving-bundle/1``), offline ``RecommenderService``,
same-fit reload parity and the fixed CPU latency protocol.

The bundle is a local directory of numeric ``.npz`` files plus a canonical ``manifest.json``. It is fitted on
TRAIN+VALIDATION only (timestamp < T_test); TEST rows are never decoded. Loading needs nothing but the
bundle directory and NumPy: no MovieLens files, no prepared partitions, no experiment directory.
Arrays are loaded with ``allow_pickle=False``; integrity checking (SHA256) detects corruption, it is not an
authenticity guarantee against someone who can rewrite both files and hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import os
import platform
import shutil
import time
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import __version__
from .artifacts import canonical_json, sha256_bytes
from .baselines import ItemKNN, Popularity
from .data import (
    OFFICIAL_ARCHIVE_SHA256,
    OFFICIAL_N_RATINGS,
    OFFICIAL_N_USERS,
    OFFICIAL_RATINGS_SHA256,
)
from .errors import ArtifactError, BundleError, InvalidRequestError, ModelIntegrityError
from .models import ALS_GRID, BPR_GRID, LatentFactorRanker, config_from_dict
from .stage import Prefix

SCHEMA = "p02-serving-bundle/1"
RESULT_SCHEMA_VERSION = "1.0"
MANIFEST_FILE = "manifest.json"
MODEL_FILES = {"popularity": "popularity.npz", "item_knn": "item_knn.npz", "als": "als.npz", "bpr": "bpr.npz"}
DATA_FILES = ("history.npz",) + tuple(MODEL_FILES.values())
METHODS = ("popularity", "item_knn", "als", "bpr")
MODEL_CHOICES = ("default",) + METHODS
DEFAULT_METHOD = "item_knn"
KNN_NEIGHBORS = 50

# Provenance anchors recorded in every bundle manifest and checked on load; never recomputed.
# The "r1/r2/r3" keys and ids are serialized identifiers (protocol, selection and bundle stages); kept for bundle compatibility.
CONTRACTS = {
    "r1": {"id": "P02-R1-ACCEPTANCE-v1.0", "sha256": "e5b6f10bf35b932693634fcbdd4ec377ece3880da14353102497c4c7f079fe8e",
           "accepted_commit": "ba7901757ac24d3b02ef9ae30e179657fbeb0761"},
    "r2": {"id": "P02-R2-ACCEPTANCE-v1.0", "sha256": "b011ac87afb920cb8ae6f3d9edfe26f67b9e1de8cf21fc8bc85578f44a63ea2b",
           "accepted_commit": "c36d4d5fa7398f69a341ba1ad6302c63bccf0d6a"},
    "r3": {"id": "P02-R3-ACCEPTANCE-v1.0", "sha256": "7eaa87a685992776fb5a6432dde3b1afbfd60f9a0a2ab0ce5bc6484ca82f8d1e"},
}
R2_SELECTION_SELF_HASH = "d0e3316487c05d6255b8304acb774154142b3adc8c7e4212075002b0f20b4f99"
ACCEPTED_R1_MANIFEST_FINGERPRINT = "97b40e93b843d598caf90408aa916afcdcb374d9be454d7229beb58e1312b532"
ACCEPTED_SPLIT_FINGERPRINT = "046dd86418ca325b27f4a3dd41fea24f3a305d7c0fa976ccbd92d2cb05f1fc69"
OFFICIAL_T_TEST = 978133414
OFFICIAL_PREFIX_EVENTS = 900188  # TRAIN 800,164 + VALIDATION 100,024
ALS_CONFIG = ALS_GRID[0]  # ALS-01
BPR_CONFIG = BPR_GRID[1]  # BPR-02

# name -> (dtype, ndim) for every array of every file (strict schema)
ARRAYS = {
    "history.npz": {
        "catalog": ("<i8", 1), "user_ids": ("<i8", 1), "seen_indptr": ("<i8", 1), "seen_items": ("<i8", 1),
        "pos_indptr": ("<i8", 1), "pos_items": ("<i8", 1),
    },
    "popularity.npz": {"counts": ("<i8", 1)},
    "item_knn.npz": {"indptr": ("<i8", 1), "neighbor_index": ("<i8", 1), "similarity": ("<f8", 1)},
    "als.npz": {"user_ids": ("<i8", 1), "item_ids": ("<i8", 1), "user_factors": ("<f8", 2), "item_factors": ("<f8", 2)},
    "bpr.npz": {"user_ids": ("<i8", 1), "item_ids": ("<i8", 1), "user_factors": ("<f8", 2), "item_factors": ("<f8", 2)},
}

# ----------------------------------------------------------------------------- helpers

def _sha_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def manifest_self_hash(body: dict) -> str:
    return sha256_bytes(canonical_json({k: v for k, v in body.items() if k != "manifest_sha256"}).encode())

def _csr(groups: dict[int, np.ndarray], users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = [groups.get(int(u), np.empty(0, dtype=np.int64)) for u in users]
    indptr = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    items = np.concatenate(parts).astype(np.int64) if parts else np.empty(0, dtype=np.int64)
    return indptr, items

def fixture_ratings_sha256() -> str:
    f = resources.files("personalized_recommender") / "fixtures" / "synthetic_ml" / "ratings.dat"
    return hashlib.sha256(f.read_bytes()).hexdigest()

def _git_revision() -> str | None:
    try:
        import subprocess

        root = Path(__file__).resolve().parents[2]
        if not (root / ".git").exists():
            return None
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - best-effort metadata only
        return None

# ----------------------------------------------------------------------------- export

def _state_arrays(prefix: Prefix, knn: ItemKNN, pop: Popularity, als: LatentFactorRanker, bpr: LatentFactorRanker) -> dict:
    users = prefix.users
    seen_indptr, seen_items = _csr(prefix._seen, users)
    pos_indptr, pos_items = _csr(prefix._pos, users)
    w = knn._w
    rows, cols = np.nonzero(w)  # row-major: seeds ascending, neighbour index ascending
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=w.shape[0]))]).astype(np.int64)
    arrays = {
        "history.npz": {"catalog": prefix.catalog.astype("<i8"), "user_ids": users.astype("<i8"),
                        "seen_indptr": seen_indptr, "seen_items": seen_items,
                        "pos_indptr": pos_indptr, "pos_items": pos_items},
        "popularity.npz": {"counts": pop.counts.astype("<i8")},
        "item_knn.npz": {"indptr": indptr, "neighbor_index": cols.astype("<i8"), "similarity": w[rows, cols].astype("<f8")},
    }
    for name, r in (("als.npz", als), ("bpr.npz", bpr)):
        arrays[name] = {"user_ids": r.matrix.user_ids.astype("<i8"), "item_ids": r.matrix.item_ids.astype("<i8"),
                        "user_factors": r.user_factors.astype("<f8"), "item_factors": r.item_factors.astype("<f8")}
    return arrays

def _software() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "package_version": __version__}

def export_bundle(prepared: Path, output_dir: Path) -> tuple[dict, dict]:
    """Fit the four selected methods on TRAIN+VALIDATION and write one verified bundle.

    Returns ``(manifest, reference)`` where ``reference`` holds the SAME fitted in-memory rankers (for parity).
    Never decodes TEST rows, never runs a validation search or reselection.
    """
    from .r2 import code_identity, load_validation_partitions, _lineage  # noqa: PLC0415 - reuse the verified partition loader

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ArtifactError("BUNDLE_EXISTS", f"{output_dir} already exists; refusing to overwrite an existing bundle")
    r1, parts = load_validation_partitions(prepared)  # TRAIN/VALIDATION decoded; TEST checked by byte hash only
    lineage = _lineage(r1)
    if lineage["official_reference_match"]:
        dataset_kind = "official_movielens_1m"
        if lineage["r1_manifest_fingerprint"] != ACCEPTED_R1_MANIFEST_FINGERPRINT \
                or lineage["split_fingerprint"] != ACCEPTED_SPLIT_FINGERPRINT:
            raise ArtifactError("LINEAGE_MISMATCH", "official prepared artifacts differ from the expected official ML-1M identity")
    elif lineage["ratings_sha256"] == fixture_ratings_sha256():
        dataset_kind = "synthetic_fixture"
    else:
        raise ArtifactError("LINEAGE_MISMATCH", "prepared artifacts are neither the official MovieLens-1M nor the bundled fixture")
    events = np.concatenate([parts["train"], parts["validation"]])
    t_test = int(r1["split"]["t_test"])
    if len(events) and int(events[:, 3].max()) >= t_test:
        raise ArtifactError("SPLIT_INTEGRITY", "TRAIN+VALIDATION contains rows with timestamp >= T_test")
    # The selected ALS/BPR configurations are fixed here, not re-selected.
    if config_from_dict(ALS_CONFIG.as_dict()) is not ALS_CONFIG or config_from_dict(BPR_CONFIG.as_dict()) is not BPR_CONFIG:
        raise ModelIntegrityError("R2_DECISION_MISMATCH", "ALS-01/BPR-02 are not in the fixed grid")
    prefix = Prefix(events)
    pop = Popularity().fit(prefix)
    knn = ItemKNN().fit(prefix)
    als = LatentFactorRanker(ALS_CONFIG).fit(prefix)
    bpr = LatentFactorRanker(BPR_CONFIG).fit(prefix)
    arrays = _state_arrays(prefix, knn, pop, als, bpr)

    tmp = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        files = {}
        for fname, arrs in arrays.items():
            np.savez(tmp / fname, **arrs)
            files[fname] = {
                "sha256": _sha_file(tmp / fname),
                "size_bytes": (tmp / fname).stat().st_size,
                "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrs.items()},
            }
        body = {
            "schema": SCHEMA,
            "dataset_kind": dataset_kind,
            "export_environment": {**_software(), "code_identity": code_identity(), "git_revision": _git_revision()},
            "contracts": CONTRACTS,
            "r2_provenance": {
                "accepted_commit": CONTRACTS["r2"]["accepted_commit"],
                "accepted_selection_manifest_sha256": R2_SELECTION_SELF_HASH,
                "selected_als": ALS_CONFIG.as_dict(),
                "selected_bpr": BPR_CONFIG.as_dict(),
                "default_method": DEFAULT_METHOD,
                "note": "recorded from the selection step; no validation search, no reselection, no TEST access in R3",
            },
            "lineage": lineage,
            "prefix": {
                "rule": "TRAIN+VALIDATION = timestamp < T_test",
                "t_test": t_test,
                "events": int(len(prefix.events)),
                "positive_events": int(len(prefix.positive_events)),
                "max_timestamp": int(prefix.events[:, 3].max()) if len(prefix.events) else None,
            },
            "serving": {
                "catalog_count": int(len(prefix.catalog)),
                "history_user_count": int(len(prefix.users)),
                "positive_user_count": int(len(als.matrix.user_ids)),
                "methods": list(METHODS),
                "default_method": DEFAULT_METHOD,
                "model_configs": {
                    "popularity": {"definition": "positive count DESC, movie_id ASC"},
                    "item_knn": {"n_neighbors": KNN_NEIGHBORS, "definition": "binary cosine, top-50 positive neighbours"},
                    "als": ALS_CONFIG.as_dict(),
                    "bpr": BPR_CONFIG.as_dict(),
                },
            },
            "files": files,
        }
        body["manifest_sha256"] = manifest_self_hash(body)
        (tmp / MANIFEST_FILE).write_text(canonical_json(body))
        service = RecommenderService.load(tmp)  # verified load after writing
        reference = {"prefix": prefix, "popularity": pop, "item_knn": knn, "als": als, "bpr": bpr}
        parity = verify_reload_parity(service, reference)
        os.rename(tmp, output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    body["reload_parity"] = parity
    return body, reference

# ----------------------------------------------------------------------------- loader

def _bad(msg: str, code: str = "BUNDLE_CORRUPT") -> BundleError:
    return BundleError(code, msg)

def _load_npz(path: Path, expected: dict) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            names = set(z.files)
            if names != set(expected):
                raise _bad(f"{path.name}: arrays {sorted(names)} != expected {sorted(expected)}")
            out = {n: z[n] for n in z.files}
    except BundleError:
        raise
    except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile) as e:
        raise _bad(f"{path.name} unreadable: {e}") from None
    for n, (dt, nd) in expected.items():
        a = out[n]
        if a.dtype != np.dtype(dt) or a.ndim != nd:
            raise _bad(f"{path.name}:{n} has dtype/ndim {a.dtype.str}/{a.ndim}, expected {dt}/{nd}")
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            raise _bad(f"{path.name}:{n} contains nonfinite values", "BUNDLE_NONFINITE")
    return out

def _check_csr(name: str, indptr: np.ndarray, n_rows: int, n_items: int) -> None:
    if len(indptr) != n_rows + 1 or indptr[0] != 0 or indptr[-1] != n_items or np.any(np.diff(indptr) < 0):
        raise _bad(f"{name}: invalid CSR offsets")

def _strictly_increasing_rows(name: str, indptr: np.ndarray, items: np.ndarray) -> None:
    if len(items) < 2:
        return
    d = np.diff(items)
    boundary = np.zeros(len(items) - 1, dtype=bool)
    inner = indptr[1:-1]
    inner = inner[(inner > 0) & (inner < len(items))]
    boundary[inner - 1] = True
    if np.any((d <= 0) & ~boundary):
        raise _bad(f"{name}: items within a user are not strictly increasing")

class RecommenderService:
    """Dataset-independent, deterministic offline recommender over a verified ``p02-serving-bundle/1``."""

    @classmethod
    def load(cls, bundle_dir) -> "RecommenderService":
        d = Path(bundle_dir)
        mpath = d / MANIFEST_FILE
        if not d.is_dir() or not mpath.is_file():
            raise _bad(f"{mpath} not found", "BUNDLE_MISSING")
        try:
            m = json.loads(mpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise _bad(f"manifest is not valid JSON: {e}") from None
        if not isinstance(m, dict):
            raise _bad("manifest must be a JSON object")
        if m.get("schema") != SCHEMA:
            raise _bad(f"unsupported bundle schema {m.get('schema')!r}", "BUNDLE_UNSUPPORTED_SCHEMA")
        try:
            if manifest_self_hash(m) != m.get("manifest_sha256"):
                raise _bad("manifest self-hash mismatch", "BUNDLE_MANIFEST_TAMPERED")
        except (TypeError, ValueError) as e:
            raise _bad(f"manifest not canonical-serialisable: {e}") from None
        try:
            return cls._build(d, m)
        except BundleError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
            raise _bad(f"manifest/bundle structurally invalid: {type(e).__name__}: {e}") from None

    @classmethod
    def _build(cls, d: Path, m: dict) -> "RecommenderService":
        files = m["files"]
        if set(files) != set(DATA_FILES):
            raise _bad(f"manifest file inventory {sorted(files)} != {sorted(DATA_FILES)}", "BUNDLE_INVENTORY")
        present = {p.name for p in d.iterdir()}
        if present != set(DATA_FILES) | {MANIFEST_FILE}:
            raise _bad(f"bundle directory inventory {sorted(present)} is not the expected file set", "BUNDLE_INVENTORY")
        data = {}
        for fname in DATA_FILES:
            p = d / fname
            if _sha_file(p) != files[fname]["sha256"]:
                raise _bad(f"{fname}: SHA256 mismatch", "BUNDLE_CHECKSUM_MISMATCH")
            data[fname] = _load_npz(p, ARRAYS[fname])
            for n, a in data[fname].items():
                decl = files[fname]["arrays"][n]
                if list(a.shape) != decl["shape"] or a.dtype.str != decl["dtype"]:
                    raise _bad(f"{fname}:{n} shape/dtype differs from manifest")
        sv, ex = m["serving"], m["r2_provenance"]
        # lineage / decision coherence
        if sv["methods"] != list(METHODS) or sv["default_method"] != DEFAULT_METHOD or ex["default_method"] != DEFAULT_METHOD:
            raise _bad("methods/default method are not the expected ones", "BUNDLE_INCOMPATIBLE")
        cfgs = sv["model_configs"]
        if cfgs["als"] != ALS_CONFIG.as_dict() or cfgs["bpr"] != BPR_CONFIG.as_dict() \
                or ex["selected_als"] != ALS_CONFIG.as_dict() or ex["selected_bpr"] != BPR_CONFIG.as_dict() \
                or cfgs["item_knn"].get("n_neighbors") != KNN_NEIGHBORS:
            raise _bad("model configuration differs from the expected ALS-01/BPR-02/Item-KNN", "BUNDLE_INCOMPATIBLE")
        if m["contracts"] != CONTRACTS or ex["accepted_selection_manifest_sha256"] != R2_SELECTION_SELF_HASH \
                or ex["accepted_commit"] != CONTRACTS["r2"]["accepted_commit"]:
            raise _bad("accepted contract/selection provenance anchors differ", "BUNDLE_INCOMPATIBLE")
        lin, pre = m["lineage"], m["prefix"]
        kind = m["dataset_kind"]
        if kind == "official_movielens_1m":
            if lin["ratings_sha256"] != OFFICIAL_RATINGS_SHA256 or lin["split_fingerprint"] != ACCEPTED_SPLIT_FINGERPRINT \
                    or lin["r1_manifest_fingerprint"] != ACCEPTED_R1_MANIFEST_FINGERPRINT or lin["official_reference_match"] is not True \
                    or pre["t_test"] != OFFICIAL_T_TEST or pre["events"] != OFFICIAL_PREFIX_EVENTS:
                raise _bad("official lineage fields are incoherent", "BUNDLE_LINEAGE")
        elif kind == "synthetic_fixture":
            if lin["official_reference_match"] is not False:
                raise _bad("synthetic fixture bundle claims official identity", "BUNDLE_LINEAGE")
        else:
            raise _bad(f"unknown dataset_kind {kind!r}", "BUNDLE_LINEAGE")
        if pre["t_test"] != lin["t_test"]:
            raise _bad("prefix t_test differs from lineage", "BUNDLE_LINEAGE")

        h = data["history.npz"]
        catalog, users = h["catalog"], h["user_ids"]
        n_items, n_users = len(catalog), len(users)
        if n_items == 0 or n_items != sv["catalog_count"] or n_users != sv["history_user_count"]:
            raise _bad("catalog/user counts differ from manifest")
        if np.any(np.diff(catalog) <= 0) or np.any(np.diff(users) <= 0):
            raise _bad("catalog/user mappings are not sorted unique")
        _check_csr("seen", h["seen_indptr"], n_users, len(h["seen_items"]))
        _check_csr("pos", h["pos_indptr"], n_users, len(h["pos_items"]))
        if len(h["seen_items"]) != pre["events"] or len(h["pos_items"]) != pre["positive_events"]:
            raise _bad("history sizes differ from manifest prefix")
        _strictly_increasing_rows("seen", h["seen_indptr"], h["seen_items"])
        _strictly_increasing_rows("pos", h["pos_indptr"], h["pos_items"])
        seen_idx = np.searchsorted(catalog, h["seen_items"])
        pos_idx = np.searchsorted(catalog, h["pos_items"])
        if (len(seen_idx) and (seen_idx.max() >= n_items or np.any(catalog[seen_idx] != h["seen_items"]))) or \
                (len(pos_idx) and (pos_idx.max() >= n_items or np.any(catalog[pos_idx] != h["pos_items"]))):
            raise _bad("history references movies outside the catalog")
        # positives must be a subset of seen (per user), and every catalog item must have been observed
        seen_rows = np.repeat(np.arange(n_users), np.diff(h["seen_indptr"]))
        pos_rows = np.repeat(np.arange(n_users), np.diff(h["pos_indptr"]))
        if not np.all(np.isin(pos_rows * n_items + pos_idx, seen_rows * n_items + seen_idx)):
            raise _bad("positive history is not a subset of the observed history")
        if len(np.unique(seen_idx)) != n_items:
            raise _bad("catalog contains items that no user observed")
        counts = data["popularity.npz"]["counts"]
        if len(counts) != n_items or not np.array_equal(counts, np.bincount(pos_idx, minlength=n_items)):
            raise _bad("popularity counts are inconsistent with the positive history")
        k = data["item_knn.npz"]
        _check_csr("item_knn", k["indptr"], n_items, len(k["neighbor_index"]))
        if len(k["similarity"]) != len(k["neighbor_index"]):
            raise _bad("item_knn arrays have different lengths")
        if len(k["neighbor_index"]):
            if k["neighbor_index"].min() < 0 or k["neighbor_index"].max() >= n_items or np.any(k["similarity"] <= 0) \
                    or np.any(k["similarity"] > 1.0 + 1e-12):
                raise _bad("item_knn neighbour indices/similarities are out of range")
            seeds = np.repeat(np.arange(n_items), np.diff(k["indptr"]))
            if np.any(seeds == k["neighbor_index"]) or np.any(np.diff(k["indptr"]) > KNN_NEIGHBORS):
                raise _bad("item_knn neighbour lists violate the top-50 / no-self rules")
        pos_user_rows = np.flatnonzero(np.diff(h["pos_indptr"]) > 0)
        lat = {}
        for name, cfg in (("als", ALS_CONFIG), ("bpr", BPR_CONFIG)):
            a = data[f"{name}.npz"]
            if not np.array_equal(a["item_ids"], catalog):
                raise _bad(f"{name}: item mapping differs from catalog")
            if not np.array_equal(a["user_ids"], users[pos_user_rows]) or sv["positive_user_count"] != len(a["user_ids"]):
                raise _bad(f"{name}: factor user mapping differs from users with positive history")
            if a["user_factors"].shape != (len(a["user_ids"]), cfg.factors) or a["item_factors"].shape != (n_items, cfg.factors):
                raise _bad(f"{name}: wrong factor dimensions", "BUNDLE_SHAPE")
            lat[name] = (a["user_ids"], a["user_factors"], a["item_factors"])
        self = cls.__new__(cls)
        self.manifest = m
        self._catalog, self._users = catalog, users
        self._seen_indptr, self._seen_idx = h["seen_indptr"], seen_idx
        self._pos_indptr, self._pos_idx = h["pos_indptr"], pos_idx
        self._counts = counts
        self._knn_indptr, self._knn_nbr, self._knn_sim = k["indptr"], k["neighbor_index"], k["similarity"]
        self._lat = lat
        self.catalog_size = n_items
        return self

    # ------------------------------------------------------------------ queries
    @property
    def default_method(self) -> str:
        return DEFAULT_METHOD

    def user_ids_with_positives(self) -> np.ndarray:
        return self._users[np.diff(self._pos_indptr) > 0]

    def _row(self, user_id: int) -> int:
        r = int(np.searchsorted(self._users, user_id))
        return r if r < len(self._users) and self._users[r] == user_id else -1

    @staticmethod
    def _check_request(user_id, top_k, model) -> None:
        if isinstance(user_id, (bool, np.bool_)) or not isinstance(user_id, numbers.Integral):
            raise InvalidRequestError("INVALID_USER_ID", f"user_id must be an integer, got {user_id!r}")
        if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, numbers.Integral) or int(top_k) < 1:
            raise InvalidRequestError("INVALID_TOP_K", f"top_k must be a positive integer, got {top_k!r}")
        if not isinstance(model, str) or model not in MODEL_CHOICES:
            raise InvalidRequestError("INVALID_MODEL", f"model must be one of {MODEL_CHOICES}, got {model!r}")

    def recommend(self, user_id, top_k=10, model="default") -> dict:
        self._check_request(user_id, top_k, model)
        user_id, top_k = int(user_id), int(top_k)
        resolved = DEFAULT_METHOD if model == "default" else model
        row = self._row(user_id)
        mask = np.ones(self.catalog_size, dtype=bool)  # candidates = catalog minus ALL observed prefix items
        pos = np.empty(0, dtype=np.int64)
        if row >= 0:
            mask[self._seen_idx[self._seen_indptr[row] : self._seen_indptr[row + 1]]] = False
            pos = self._pos_idx[self._pos_indptr[row] : self._pos_indptr[row + 1]]
        cand = np.flatnonzero(mask)  # ascending catalog index == ascending numeric movie id
        pop = self._counts[cand]
        fallback, semantics = False, "popularity_positive_count"
        if resolved == "popularity":
            scores, order = pop.astype(np.float64), np.lexsort((cand, -pop))
        elif resolved == "item_knn":
            s = None
            if len(pos):
                lo, hi = self._knn_indptr[pos], self._knn_indptr[pos + 1]
                sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(pos) else np.empty(0, np.int64)
                s = np.bincount(self._knn_nbr[sel], weights=self._knn_sim[sel], minlength=self.catalog_size)[cand]
            if s is None or not np.any(s > 0):
                fallback = True
                scores, order = pop.astype(np.float64), np.lexsort((cand, -pop))
            else:
                semantics = "item_knn_similarity_sum"
                scores, order = s, np.lexsort((cand, -pop, -s))
        else:
            uids, uf, itf = self._lat[resolved]
            ur = int(np.searchsorted(uids, user_id))
            if ur >= len(uids) or uids[ur] != user_id or len(pos) == 0:
                fallback = True
                scores, order = pop.astype(np.float64), np.lexsort((cand, -pop))
            else:
                scores = itf[cand] @ uf[ur]
                if not np.all(np.isfinite(scores)):
                    raise ModelIntegrityError("NONFINITE_SCORE", f"{resolved} produced NaN/Inf scores for user {user_id}")
                semantics = f"{resolved}_latent_dot_product"
                order = np.lexsort((cand, -scores))
        top = order[:top_k]
        movies = self._catalog[cand[top]].tolist()
        sc = scores[top].tolist()
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "user_id": user_id,
            "requested_model": model,
            "resolved_model": resolved,
            "top_k": top_k,
            "returned_count": len(movies),
            "fallback_used": fallback,
            "score_semantics": semantics,
            "recommendations": [{"movie_id": int(a), "rank": i + 1, "score": float(b)} for i, (a, b) in enumerate(zip(movies, sc))],
        }

# ----------------------------------------------------------------------------- same-fit reload parity

def reference_recommend(reference: dict, method: str, user_id: int, top_k: int) -> tuple[list[int], list[float], bool]:
    """Reference ranking from the SAME in-memory fit, using the baseline and model ranker classes (independent of the service path)."""
    prefix, ranker = reference["prefix"], reference[method]
    cands = prefix.candidates(user_id)
    ranked = ranker.rank(user_id, cands)
    pop = reference["popularity"]
    fallback = False
    if method == "popularity":
        score_of = {int(i): float(c) for i, c in zip(cands, pop.pop_of(cands))}
    elif method == "item_knn":
        s = ranker.score(user_id, cands)
        if len(prefix.positives(user_id)) == 0 or not np.any(s > 0):
            fallback = True
            score_of = {int(i): float(c) for i, c in zip(cands, pop.pop_of(cands))}
        else:
            score_of = {int(i): float(v) for i, v in zip(cands, s)}
    else:
        s = ranker.score(user_id, cands)
        if s is None:
            fallback = True
            score_of = {int(i): float(c) for i, c in zip(cands, pop.pop_of(cands))}
        else:
            score_of = {int(i): float(v) for i, v in zip(cands, s)}
    top = [int(i) for i in ranked[:top_k]]
    return top, [score_of[i] for i in top], fallback

def parity_users(prefix: Prefix) -> dict[str, list[int]]:
    pos_users = [int(u) for u in prefix.users if len(prefix.positives(u))]
    zero_pos = [int(u) for u in prefix.users if len(prefix.positives(u)) == 0]
    spread = pos_users[:: max(1, len(pos_users) // 8)][:8]
    known = sorted(set(pos_users[:12]) | set(spread))
    unknown = [int(prefix.users.max()) + 1_000_003]
    return {"known_positive": known, "zero_positive": zero_pos[:10], "unknown": unknown}

def verify_reload_parity(service: RecommenderService, reference: dict, ks=(5, 10, 20), rtol=1e-9, atol=1e-9) -> dict:
    """Exact ID/order/rank equality and score tolerance for every method, K and user category."""
    groups = parity_users(reference["prefix"])
    report: dict = {"rtol": rtol, "atol": atol, "k_values": list(ks), "user_groups": {g: len(u) for g, u in groups.items()},
                    "methods": {}}
    for method in METHODS:
        checks = mismatches = 0
        max_err = 0.0
        for group, users in groups.items():
            for u in users:
                for k in ks:
                    got = service.recommend(u, k, method)
                    ids, scores, fb = reference_recommend(reference, method, u, k)
                    rec = got["recommendations"]
                    ok = [r["movie_id"] for r in rec] == ids and [r["rank"] for r in rec] == list(range(1, len(ids) + 1)) \
                        and got["fallback_used"] == fb
                    if ok and ids:
                        err = float(np.max(np.abs(np.array([r["score"] for r in rec]) - np.array(scores))))
                        max_err = max(max_err, err)
                        ok = all(math.isclose(r["score"], s, rel_tol=rtol, abs_tol=atol) and math.isfinite(r["score"])
                                 for r, s in zip(rec, scores))
                    checks += 1
                    mismatches += 0 if ok else 1
        report["methods"][method] = {"comparisons": checks, "mismatches": mismatches, "max_abs_score_diff": max_err}
        if mismatches:
            raise ModelIntegrityError("RELOAD_PARITY_FAILED", f"{method}: {mismatches}/{checks} comparisons differ")
    report["status"] = "PASS"
    return report

# ----------------------------------------------------------------------------- latency

def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"

def run_latency(service: RecommenderService, warmup: int = 100, measured: int = 1000, cohort: int = 100) -> dict:
    """Fixed protocol: default item_knn, top_k=10, first 100 ascending positive users, 100 warmups + 1000 timed calls."""
    users = [int(u) for u in service.user_ids_with_positives()[:cohort]]
    if not users:
        raise ModelIntegrityError("LATENCY_NO_USERS", "bundle has no users with positive history")
    method, k = DEFAULT_METHOD, 10
    lat = np.empty(measured, dtype=np.float64)
    with threadpool_limits(limits=1):
        for i in range(warmup):
            service.recommend(users[i % len(users)], k, method)
        for i in range(measured):
            u = users[i % len(users)]
            t0 = time.perf_counter_ns()
            res = service.recommend(u, k, method)
            t1 = time.perf_counter_ns()
            recs = res["recommendations"]
            if res["returned_count"] != len(recs) or [r["rank"] for r in recs] != list(range(1, len(recs) + 1)) or len(recs) > k:
                raise ModelIntegrityError("LATENCY_INVALID_OUTPUT", f"invalid recommendation output for user {u}")
            lat[i] = (t1 - t0) / 1e6
    if not np.all(np.isfinite(lat)) or np.any(lat < 0):
        raise ModelIntegrityError("LATENCY_INVALID", "non-finite or negative latency observed")
    p = {f"p{q}_ms": float(np.percentile(lat, q, method="linear")) for q in (50, 95, 99)}
    stats = {"mean_ms": float(lat.mean()), "min_ms": float(lat.min()), "max_ms": float(lat.max()), **p}
    if not (stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]
            and stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]):
        raise ModelIntegrityError("LATENCY_INCOHERENT", "latency statistics are internally inconsistent")
    return {
        "schema": "p02-latency-report/1",
        "method": method,
        "top_k": k,
        "dataset_kind": service.manifest["dataset_kind"],
        "bundle_manifest_sha256": service.manifest["manifest_sha256"],
        "catalog_size": service.catalog_size,
        "warmup_count": warmup,
        "measured_count": measured,
        "user_selection_rule": f"first {cohort} ascending numeric user IDs with >=1 positive (rating>=4) TRAIN+VALIDATION "
                               f"observation; request i uses user_ids[i % {len(users)}]",
        "cohort_size": len(users),
        "cohort_first_last": [users[0], users[-1]],
        "percentile_definition": "numpy.percentile(method='linear')",
        **stats,
        "timing_scope": "time.perf_counter_ns() around RecommenderService.recommend() only",
        "timing_excludes": ["package import", "bundle loading", "CLI startup", "model fitting", "JSON serialization",
                            "printing", "disk I/O"],
        "concurrency": 1,
        "single_process": True,
        "environment": {
            "cpu_model": _cpu_model(), "architecture": platform.machine(), "os": platform.platform(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "blas_threads_during_timing": 1,
            "env": {k_: os.environ.get(k_) for k_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "no_sla_note": "descriptive measurement on this machine; no latency target or SLA is claimed",
    }

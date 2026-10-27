"""Experiment workflow: fixed 8-trial validation search + selection lock (``train``) and the
single authorized four-method sealed-TEST campaign (``benchmark``).

``train`` never decodes TEST rows: it verifies TRAIN/VALIDATION content and only the byte-level
checksum of ``test.npy`` (identity metadata only). ``benchmark`` opens TEST only after the
selection manifest's canonical hash, protocol, code identity and data lineage are verified.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import stat
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import (
    ARTIFACT_SCHEMA,
    MANIFEST,
    PROTOCOL_VERSION as R1_PROTOCOL_VERSION,
    _load_manifest,
    canonical_json,
    load_partitions,
    sha256_bytes,
    split_fingerprint,
)
from .baselines import ItemKNN, Popularity
from .errors import ArtifactError, ModelIntegrityError
from .evaluator import evaluate
from .models import ALS_GRID, BPR_GRID, BASE_SEED, LatentFactorRanker, config_from_dict
from .split import events_fingerprint
from .stage import build_stage, build_test_stage

R2_PROTOCOL_VERSION = "P02-R2-ACCEPTANCE-v1.0"
SELECTION_SCHEMA = "p02-r2-selection-manifest/1"
BENCHMARK_SCHEMA = "p02-r2-final-benchmark/1"
SELECTION_FILE = "selection_manifest.json"
METHOD_PREFERENCE = ("popularity", "item_knn", "als", "bpr")
SELECTION_RULE = {
    "per_family": "max NDCG@10, then max Recall@10, then max HitRate@10, then earliest config ID (exact equality only)",
    "default_method": "max NDCG@10, then max Recall@10, then max HitRate@10, then preference popularity<item_knn<als<bpr",
    "data": "VALIDATION metrics only",
}
# Reference VALIDATION values of the Popularity/Item-KNN baselines on official ML-1M (6-decimal rounding).
R1_REPORTED_VALIDATION_K10 = {
    "split_fingerprint": "046dd86418ca325b27f4a3dd41fea24f3a305d7c0fa976ccbd92d2cb05f1fc69",
    "popularity": {"recall": 0.060736, "hitrate": 0.662738, "ndcg": 0.260403, "catalog_coverage": 0.033042},
    "item_knn": {"recall": 0.062482, "hitrate": 0.655485, "ndcg": 0.265665, "catalog_coverage": 0.055161},
    "eligible_users_count": 1103,
    "rounding_half_width": 5e-7,
}


# ----------------------------------------------------------------------------- identity helpers


def code_identity() -> dict:
    """SHA256 over the package's Python sources (sorted relative path + bytes): stable, non-circular."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))
    for rel in files:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return {"package_version": __version__, "source_sha256": h.hexdigest(), "files": len(files)}


def software_versions() -> dict:
    def v(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "threadpoolctl": v("threadpoolctl"),
        "personalized_recommender": __version__,
        "model_implementation": "personalized_recommender.models (pure NumPy CPU ALS/BPR; no external recsys library)",
    }


def thread_settings() -> dict:
    return {
        "model_training_threads": 1,
        "blas_threads_during_fit": 1,
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def _lineage(manifest: dict) -> dict:
    d, s, f = manifest["dataset"], manifest["split"], manifest["files"]
    return {
        "r1_protocol_version": manifest["protocol_version"],
        "r1_artifact_schema": manifest["artifact_schema"],
        "r1_manifest_fingerprint": manifest["manifest_fingerprint"],
        "ratings_sha256": d["ratings_file"]["sha256"],
        "archive_sha256": (d.get("archive") or {}).get("sha256"),
        "official_reference_match": d["official"]["official_reference_match"],
        "counts": d["counts"],
        "t_val": s["t_val"],
        "t_test": s["t_test"],
        "split_counts": s["counts"],
        "split_fingerprint": s["split_fingerprint"],
        "file_sha256": {p: f[p]["sha256"] for p in ("train", "validation", "test")},
    }


def load_validation_partitions(art: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Verified TRAIN and VALIDATION rows for ``train``; TEST is checked by byte hash only (never decoded)."""
    art = Path(art)
    m = _load_manifest(art)
    if m.get("protocol_version") != R1_PROTOCOL_VERSION or m.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ArtifactError("PROTOCOL_MISMATCH", "prepared artifacts do not use the expected protocol version")
    body = {k: v for k, v in m.items() if k != "manifest_fingerprint"}
    if sha256_bytes(canonical_json(body).encode()) != m.get("manifest_fingerprint"):
        raise ArtifactError("MANIFEST_FINGERPRINT_MISMATCH", "manifest content does not match its fingerprint")
    try:
        files, sp = m["files"], m["split"]
        parts = {}
        for p in ("train", "validation"):
            raw = _read(art / files[p]["file"])
            if sha256_bytes(raw) != files[p]["sha256"]:
                raise ArtifactError("CHECKSUM_MISMATCH", f"{p}.npy sha256 does not match manifest")
            arr = _decode_partition(p, raw)
            if arr.dtype != np.dtype("<i8") or arr.ndim != 2 or arr.shape[1] != 4 or len(arr) != files[p]["rows"]:
                raise ArtifactError("CORRUPT_ARTIFACT", f"{p}.npy has unexpected dtype/shape")
            if events_fingerprint(arr) != files[p]["content_fingerprint"]:
                raise ArtifactError("CHECKSUM_MISMATCH", f"{p}.npy content fingerprint mismatch")
            parts[p] = arr
        if sha256_bytes(_read(art / files["test"]["file"])) != files["test"]["sha256"]:  # identity only
            raise ArtifactError("CHECKSUM_MISMATCH", "test.npy sha256 does not match manifest")
        tv, tt = sp["t_val"], sp["t_test"]
        tr, va = parts["train"], parts["validation"]
        if (len(tr) and tr[:, 3].max() >= tv) or (len(va) and (va[:, 3].min() < tv or va[:, 3].max() >= tt)):
            raise ArtifactError("SPLIT_INTEGRITY", "TRAIN/VALIDATION rows violate the recorded cutoffs")
        fps = {p: files[p]["content_fingerprint"] for p in ("train", "validation", "test")}
        if split_fingerprint(tv, tt, fps) != sp["split_fingerprint"]:
            raise ArtifactError("SPLIT_INTEGRITY", "split fingerprint mismatch")
    except KeyError as e:
        raise ArtifactError("CORRUPT_ARTIFACT", f"manifest missing field {e}") from None
    return m, parts


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise ArtifactError("MISSING_ARTIFACT", f"{path} not found")
    return path.read_bytes()


def _decode_partition(name: str, raw: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise ArtifactError("CORRUPT_ARTIFACT", f"{name}.npy unreadable: {e}") from None


# ----------------------------------------------------------------------------- selection


def _k10(row: dict) -> tuple[float, float, float]:
    m = row["metrics_by_k"]["10"]
    return (m["ndcg"], m["recall"], m["hitrate"])


def select_config(trials: list[dict]) -> dict:
    """Per-family winner: max (NDCG@10, Recall@10, HitRate@10); exact ties -> earliest config order."""
    best = None
    for order, t in enumerate(trials):
        key = _k10(t) + (-order,)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def select_default(candidates: dict[str, dict]) -> str:
    """Cross-method default: max (NDCG@10, Recall@10, HitRate@10); ties -> popularity<item_knn<als<bpr."""
    best = None
    for pref, method in enumerate(METHOD_PREFERENCE):
        key = _k10(candidates[method]) + (-pref,)
        if best is None or key > best[0]:
            best = (key, method)
    return best[1]


def _summary(result: dict, stage_diag: dict) -> dict:
    return {
        "metrics_by_k": result["metrics_by_k"],
        "eligible_users_count": result["eligible_users_count"],
        "excluded_users_count": stage_diag["excluded_users_count"],
        "cold_start": result["cold_start"],
    }


def manifest_hash(body: dict) -> str:
    return sha256_bytes(canonical_json({k: v for k, v in body.items() if k != "manifest_sha256"}).encode())


def run_train(artifacts: Path, output_dir: Path, grid=None) -> dict:
    """Execute the fixed eight-trial TRAIN->VALIDATION search and lock the selection manifest."""
    output_dir = Path(output_dir)
    target = output_dir / SELECTION_FILE
    if target.exists():
        raise ArtifactError("SELECTION_ALREADY_LOCKED", f"{target} exists; a locked selection is never overwritten")
    manifest, parts = load_validation_partitions(artifacts)
    stage = build_stage("validation", parts["train"], parts["validation"])  # prefix = TRAIN only
    grid = grid if grid is not None else (ALS_GRID, BPR_GRID)

    baselines = {}
    for name, cls in (("popularity", Popularity), ("item_knn", ItemKNN)):
        t0 = time.perf_counter()
        r = cls().fit(stage.prefix)
        fit_s = time.perf_counter() - t0
        res = evaluate(stage, r)
        baselines[name] = {**_summary(res, stage.diagnostics), "train_seconds": fit_s,
                           "eval_seconds": time.perf_counter() - t0 - fit_s}

    trials = []
    for family_grid in grid:
        for cfg in family_grid:
            ranker = LatentFactorRanker(cfg).fit(stage.prefix)  # fresh mapping + fresh model per trial
            t0 = time.perf_counter()
            res = evaluate(stage, ranker)
            trials.append({
                "config": cfg.as_dict(),
                "status": "completed",
                "train_seconds": ranker.train_seconds,
                "eval_seconds": time.perf_counter() - t0,
                "user_factor_shape": list(ranker.user_factors.shape),
                "item_factor_shape": list(ranker.item_factors.shape),
                "final_epoch_loss": ranker.fit_info["loss_trace"][-1] if ranker.fit_info["loss_trace"] else None,
                "fallback_users": ranker.fallback_users,
                "personalized_users": ranker.personalized_users,
                **_summary(res, stage.diagnostics),
            })
    als_trials = [t for t in trials if t["config"]["family"] == "als"]
    bpr_trials = [t for t in trials if t["config"]["family"] == "bpr"]
    best_als, best_bpr = select_config(als_trials), select_config(bpr_trials)
    default = select_default({"popularity": baselines["popularity"], "item_knn": baselines["item_knn"],
                              "als": best_als, "bpr": best_bpr})
    lineage = _lineage(manifest)
    body = {
        "schema": SELECTION_SCHEMA,
        "protocol_version": R2_PROTOCOL_VERSION,
        "r1_protocol_version": R1_PROTOCOL_VERSION,
        "status": "FINALIZED",
        "test_labels_accessed": False,
        "code_identity": code_identity(),
        "software_versions": software_versions(),
        "base_seed": BASE_SEED,
        "thread_settings": thread_settings(),
        "data_lineage": lineage,
        "validation_stage": {"prefix": "TRAIN (t < T_val)", "heldout": "VALIDATION", "diagnostics": stage.diagnostics},
        "trials": trials,
        "selected": {
            "als": best_als["config"],
            "bpr": best_bpr["config"],
            "default_method": default,
            "rejected_configs": [t["config"]["config_id"] for t in trials
                                 if t["config"]["config_id"] not in (best_als["config"]["config_id"], best_bpr["config"]["config_id"])],
        },
        "selection_rule": SELECTION_RULE,
        "baseline_reference": baselines,
        "r1_baseline_reconciliation": reconcile_baselines(baselines, lineage, stage.diagnostics),
    }
    body["manifest_sha256"] = manifest_hash(body)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = canonical_json(body)
    target.write_text(text)
    os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only: finalized
    return body


def reconcile_baselines(baselines: dict, lineage: dict, diag: dict) -> dict:
    ref = R1_REPORTED_VALIDATION_K10
    if lineage["split_fingerprint"] != ref["split_fingerprint"]:
        return {"status": "NOT_APPLICABLE", "reason": "artifacts are not the official ML-1M split"}
    out = {"status": "MATCH", "reference": "recorded baseline VALIDATION values (rounded to 6 decimals)", "methods": {}}
    for m in ("popularity", "item_knn"):
        rec = {}
        for k, v in ref[m].items():
            new = baselines[m]["metrics_by_k"]["10"][k]
            rec[k] = {"recomputed": new, "r1_reported": v, "abs_diff": abs(new - v)}
            if abs(new - v) > ref["rounding_half_width"]:
                out["status"] = "DIFFERS"
        out["methods"][m] = rec
    if diag["eligible_users_count"] != ref["eligible_users_count"]:
        out["status"] = "DIFFERS"
    return out


# ----------------------------------------------------------------------------- benchmark


def verify_selection(experiment_dir: Path, r1_manifest: dict) -> dict:
    path = Path(experiment_dir) / SELECTION_FILE
    if not path.is_file():
        raise ArtifactError("SELECTION_MISSING", f"{path} not found; run `train` first (benchmark before selection)")
    try:
        sel = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtifactError("SELECTION_CORRUPT", f"selection manifest is not valid JSON: {e}") from None
    if not isinstance(sel, dict):
        raise ArtifactError("SELECTION_CORRUPT", "selection manifest must be a JSON object")
    try:
        if manifest_hash(sel) != sel.get("manifest_sha256"):
            raise ArtifactError("SELECTION_CHECKSUM_MISMATCH", "selection manifest content does not match its SHA256")
        if sel.get("schema") != SELECTION_SCHEMA or sel.get("protocol_version") != R2_PROTOCOL_VERSION \
                or sel.get("r1_protocol_version") != R1_PROTOCOL_VERSION:
            raise ArtifactError("PROTOCOL_MISMATCH", "selection manifest protocol/schema mismatch")
        if sel.get("status") != "FINALIZED" or sel.get("test_labels_accessed") is not False:
            raise ArtifactError("SELECTION_NOT_FINALIZED", "selection manifest is not finalized")
        if sel["code_identity"]["source_sha256"] != code_identity()["source_sha256"]:
            raise ArtifactError("SOURCE_IDENTITY_MISMATCH", "selection was produced by different source code")
        if sel["data_lineage"] != _lineage(r1_manifest):
            raise ArtifactError("DATA_LINEAGE_MISMATCH", "selection manifest lineage differs from the prepared artifacts")
        ids = [t["config"]["config_id"] for t in sel["trials"]]
        expected = [c.config_id for c in ALS_GRID + BPR_GRID]
        if ids != expected or [t["config"] for t in sel["trials"]] != [c.as_dict() for c in ALS_GRID + BPR_GRID]:
            raise ArtifactError("SELECTION_INCONSISTENT", f"trial table {ids} is not the fixed eight-config grid")
        if any(t["status"] != "completed" for t in sel["trials"]):
            raise ArtifactError("SELECTION_INCONSISTENT", "not all eight trials completed")
        als = select_config([t for t in sel["trials"] if t["config"]["family"] == "als"])
        bpr = select_config([t for t in sel["trials"] if t["config"]["family"] == "bpr"])
        default = select_default({**{m: sel["baseline_reference"][m] for m in ("popularity", "item_knn")},
                                  "als": als, "bpr": bpr})
        if (sel["selected"]["als"], sel["selected"]["bpr"], sel["selected"]["default_method"]) != \
                (als["config"], bpr["config"], default):
            raise ArtifactError("SELECTION_INCONSISTENT", "recorded selection does not follow the selection rules")
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, ArtifactError):
            raise
        raise ArtifactError("SELECTION_CORRUPT", f"selection manifest malformed: {e}") from None
    return sel


class RecordingRanker:
    """Transparent proxy that records each eligible user's ordinary Top-10 (same output the evaluator scores)."""

    def __init__(self, inner, k: int = 10):
        self.inner, self.k, self.name, self.top = inner, k, inner.name, {}

    def fit(self, prefix):
        self.inner.fit(prefix)
        return self

    def rank(self, user_id, candidates):
        r = self.inner.rank(user_id, candidates)
        self.top[int(user_id)] = np.asarray(r[: self.k], dtype=np.int64)
        return r


def head_tail(prefix) -> dict:
    """Head = first ceil(0.2*|C|) items by (pre-test positive count DESC, movie_id ASC); tail = rest."""
    pop = Popularity().fit(prefix)
    order = np.lexsort((pop.catalog, -pop.counts))
    n = len(pop.catalog)
    n_head = math.ceil(0.20 * n)
    head = pop.catalog[order[:n_head]]
    tail = pop.catalog[order[n_head:]]
    return {"head": np.sort(head), "tail": np.sort(tail), "n_head": int(n_head), "n_tail": int(n - n_head),
            "tail_zero_positive_items": int((pop.counts[np.searchsorted(pop.catalog, tail)] == 0).sum())}


def tail_metrics(stage, top10: dict[int, np.ndarray], tail: np.ndarray) -> dict:
    tail_set = set(tail.tolist())
    per_user, tail_pos, total_pos = [], 0, 0
    for u in stage.eligible_users.tolist():
        rec = top10[u].tolist()
        total_pos += len(rec)
        tail_pos += sum(1 for m in rec if m in tail_set)
        gt_tail = [m for m in stage.ground_truth[u].tolist() if m in tail_set]
        if gt_tail:
            per_user.append(sum(1 for m in rec if m in set(gt_tail)) / len(gt_tail))
    return {
        "tail_eligible_users": len(per_user),
        "tail_recall_at_10": (math.fsum(per_user) / len(per_user)) if per_user else None,
        "tail_positions_at_10": tail_pos,
        "emitted_positions_at_10": total_pos,
        "tail_exposure_at_10": (tail_pos / total_pos) if total_pos else None,
    }


def run_benchmark(artifacts: Path, experiment_dir: Path) -> dict:
    """The single authorized four-method TEST campaign (requires a verified locked selection)."""
    r1_manifest = _load_manifest(Path(artifacts))
    if r1_manifest.get("protocol_version") != R1_PROTOCOL_VERSION:
        raise ArtifactError("PROTOCOL_MISMATCH", "prepared artifacts do not use the expected protocol version")
    try:
        sel = verify_selection(experiment_dir, r1_manifest)  # BEFORE any TEST row is decoded
    except KeyError as e:
        raise ArtifactError("CORRUPT_ARTIFACT", f"prepared manifest missing field {e}") from None
    verified_at = time.time()
    manifest, parts = load_partitions(artifacts)  # full artifact verification; TEST opened here
    if sel["data_lineage"] != _lineage(manifest):
        raise ArtifactError("DATA_LINEAGE_MISMATCH", "prepared artifacts changed during verification")
    stage = build_test_stage(parts["train"], parts["validation"], parts["test"])  # prefix = TRAIN+VALIDATION
    ht = head_tail(stage.prefix)
    methods = {
        "popularity": (Popularity(), {"name": "popularity"}),
        "item_knn": (ItemKNN(), {"name": "item_knn", "n_neighbors": 50}),
        "als": (LatentFactorRanker(config_from_dict(sel["selected"]["als"])), sel["selected"]["als"]),
        "bpr": (LatentFactorRanker(config_from_dict(sel["selected"]["bpr"])), sel["selected"]["bpr"]),
    }
    results = {}
    populations = {}
    for name, (model, cfg) in methods.items():
        rec = RecordingRanker(model)
        t0 = time.perf_counter()
        rec.fit(stage.prefix)  # fresh fit on TRAIN+VALIDATION
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        res = evaluate(stage, rec)
        eval_s = time.perf_counter() - t1
        populations[name] = sorted(rec.top)
        fb = getattr(model, "fallback_users", None)
        results[name] = {
            "configuration": cfg,
            "train_seconds": train_s,
            "test_eval_seconds": eval_s,
            "metrics_by_k": res["metrics_by_k"],
            "eligible_users_count": res["eligible_users_count"],
            "ranked_candidate_pairs": res["ranked_candidate_pairs"],
            "cold_start": res["cold_start"],
            "fallback_users": fb,
            "personalized_users": getattr(model, "personalized_users", None),
            "long_tail": tail_metrics(stage, rec.top, ht["tail"]),
        }
        if name in ("als", "bpr"):
            results[name]["user_factor_shape"] = list(model.user_factors.shape)
            results[name]["item_factor_shape"] = list(model.item_factors.shape)
    if len({tuple(v) for v in populations.values()}) != 1 or populations["popularity"] != stage.eligible_users.tolist():
        raise ModelIntegrityError("POPULATION_MISMATCH", "methods were evaluated on different user populations")
    if len({r["ranked_candidate_pairs"] for r in results.values()}) != 1:
        raise ModelIntegrityError("POPULATION_MISMATCH", "methods ranked different candidate sets")
    comparisons = {}
    for model_name in ("als", "bpr"):
        for base in ("popularity", "item_knn"):
            comparisons[f"{model_name}_vs_{base}"] = {
                f"{metric}@{k}_delta": results[model_name]["metrics_by_k"][k][metric] - results[base]["metrics_by_k"][k][metric]
                for k in ("5", "10", "20") for metric in ("ndcg", "recall", "hitrate", "catalog_coverage")
            }
    return {
        "schema": BENCHMARK_SCHEMA,
        "protocol_version": R2_PROTOCOL_VERSION,
        "r1_protocol_version": R1_PROTOCOL_VERSION,
        "stage": "test",
        "refit_prefix": "TRAIN+VALIDATION (t < T_test); TEST rows never used for fitting, mappings, catalog, histories or popularity",
        "heldout": "TEST (t >= T_test)",
        "selection_manifest_sha256": sel["manifest_sha256"],
        "selection_verified_before_test_open": True,
        "selection_verified_at_unix": verified_at,
        "selected": sel["selected"],
        "default_method": sel["selected"]["default_method"],
        "code_identity": code_identity(),
        "software_versions": software_versions(),
        "thread_settings": thread_settings(),
        "data_lineage": _lineage(manifest),
        "test_file_sha256": manifest["files"]["test"]["sha256"],
        "k_values": [5, 10, 20],
        "methods": list(methods),
        "stage_diagnostics": stage.diagnostics,
        "head_tail_definition": {
            "rule": "pre-test (TRAIN+VALIDATION) positive count DESC, movie_id ASC; head = first ceil(0.20*|C|)",
            "catalog_size": len(stage.prefix.catalog),
            "head_item_count": ht["n_head"],
            "tail_item_count": ht["n_tail"],
            "tail_zero_positive_items": ht["tail_zero_positive_items"],
        },
        "results": results,
        "comparisons": comparisons,
    }

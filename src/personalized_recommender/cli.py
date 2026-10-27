"""Command-line interface: prepare | validate | evaluate | smoke | train | benchmark | export | recommend | latency.

Exit codes: 0 success; 2 malformed arguments / invalid or missing data / corrupt artifacts /
protocol mismatch / prohibited TEST evaluation; 1 unexpected internal failure.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

from .errors import PhaseError, RecError

EVAL_SPLITS = ("validation", "test")


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse default already exits 2; keep a uniform message
        self.print_usage(sys.stderr)
        print(f"ERROR INVALID_ARGUMENTS: {message}", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m personalized_recommender", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    a = sub.add_parser("prepare", help="parse/validate ML-1M ratings and persist the temporal split")
    a.add_argument("--data-dir", required=True, type=Path)
    a.add_argument("--output-dir", required=True, type=Path)
    a.add_argument("--archive", type=Path, default=None, help="optional ml-1m.zip to hash and cross-check")
    v = sub.add_parser("validate", help="verify persisted artifact identity, checksums and split integrity")
    v.add_argument("--artifacts", required=True, type=Path)
    e = sub.add_parser("evaluate", help="full-catalog Top-K evaluation of Popularity or Item-KNN (validation split only)")
    e.add_argument("--artifacts", required=True, type=Path)
    e.add_argument("--model", required=True, choices=("popularity", "item_knn"))
    e.add_argument("--split", required=True, choices=EVAL_SPLITS)
    e.add_argument("--output", type=Path, default=None)
    sub.add_parser("smoke", help="offline end-to-end run on bundled synthetic fixtures")
    t = sub.add_parser("train", help="fixed 8-trial ALS/BPR validation search; locks selection_manifest.json")
    t.add_argument("--artifacts", required=True, type=Path)
    t.add_argument("--output-dir", required=True, type=Path)
    b = sub.add_parser("benchmark", help="refit the four selected methods on TRAIN+VALIDATION and evaluate once on TEST (requires `train` output)")
    b.add_argument("--artifacts", required=True, type=Path)
    b.add_argument("--experiment-dir", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    x = sub.add_parser("export", help="fit the four selected methods on TRAIN+VALIDATION and write a serving bundle")
    x.add_argument("--artifacts", required=True, type=Path)
    x.add_argument("--output-dir", required=True, type=Path)
    r = sub.add_parser("recommend", help="offline Top-K recommendations from a serving bundle (JSON on stdout)")
    r.add_argument("--bundle", required=True, type=Path)
    r.add_argument("--user-id", required=True, type=_int_arg)
    r.add_argument("--top-k", type=_int_arg, default=10)
    r.add_argument("--model", default="default")
    la = sub.add_parser("latency", help="fixed 100-warmup + 1000-call CPU latency protocol (default item_knn, K=10)")
    la.add_argument("--bundle", required=True, type=Path)
    la.add_argument("--output", type=Path, default=None)
    return p


def _int_arg(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def _emit(obj: dict, output: Path | None = None) -> None:
    from .artifacts import canonical_json

    text = canonical_json(obj)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    sys.stdout.write(text)


def run_evaluate(artifacts: Path, model: str, split: str) -> dict:
    if split != "validation":
        # TEST is sealed: refused before any artifact/label is loaded.
        raise PhaseError(
            "TEST_HOLDOUT_SEALED",
            "`evaluate` accepts the validation split only; TEST is sealed and is used only by `benchmark`",
        )
    from .artifacts import PROTOCOL_VERSION, load_partitions
    from .baselines import MODELS
    from .evaluator import evaluate
    from .stage import build_stage

    manifest, parts = load_partitions(artifacts)
    stage = build_stage("validation", parts["train"], parts["validation"])  # prefix = TRAIN only
    ranker = MODELS[model]().fit(stage.prefix)
    result = evaluate(stage, ranker)
    files = manifest["files"]
    model_info: dict = {"name": model}
    if model == "popularity":
        model_info["definition"] = "pop(i)=#positive(>=4) TRAIN events; rank pop DESC, movie_id ASC"
    else:
        model_info["definition"] = (
            "binary-cosine on TRAIN positives; top-50 strictly-positive neighbours per seed "
            "(sim DESC, movie_id ASC); s(u,i)=sum sim; rank s DESC, pop DESC, movie_id ASC; "
            "Popularity fallback when no positive history or all scores zero"
        )
        model_info["n_neighbors"] = ranker.n_neighbors
        model_info["popularity_fallback_users"] = ranker.fallback_users
        model_info["personalized_users"] = ranker.personalized_users
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "validation",
        "snapshot": {"prefix": "TRAIN (t < T_val)", "heldout": "VALIDATION (T_val <= t < T_test)"},
        "model": model_info,
        "dataset_identity": {
            "ratings_sha256": manifest["dataset"]["ratings_file"]["sha256"],
            "archive_sha256": (manifest["dataset"].get("archive") or {}).get("sha256"),
            "official_reference_match": manifest["dataset"]["official"]["official_reference_match"],
        },
        "split_identity": {
            "t_val": manifest["split"]["t_val"],
            "t_test": manifest["split"]["t_test"],
            "split_fingerprint": manifest["split"]["split_fingerprint"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "train_sha256": files["train"]["sha256"],
            "validation_sha256": files["validation"]["sha256"],
        },
        "k_values": [5, 10, 20],
        "candidate_protocol": "full known-item catalog C_s = items rated in TRAIN; candidates = C_s minus all "
        "items the user rated in TRAIN; no sampling",
        "diagnostics": stage.diagnostics,
        **result,
    }


def run_smoke() -> dict:
    from importlib import resources

    from .artifacts import prepare, validate

    fx = resources.files("personalized_recommender") / "fixtures" / "synthetic_ml"
    with tempfile.TemporaryDirectory() as tmp, resources.as_file(fx) as data_dir:
        art = Path(tmp) / "artifacts"
        prep = prepare(Path(data_dir), art)
        val = validate(art)
        pop = run_evaluate(art, "popularity", "validation")
        knn = run_evaluate(art, "item_knn", "validation")
        try:
            run_evaluate(art, "popularity", "test")
            sealed = False
        except PhaseError:
            sealed = True
    if not sealed:
        raise RuntimeError("smoke: TEST holdout was not sealed")
    return {
        "smoke": "PASS",
        "fixture": "bundled synthetic_ml (synthetic, not MovieLens data)",
        "prepare_status": prep["status"],
        "validate_status": val["status"],
        "split": {"t_val": val["t_val"], "t_test": val["t_test"], "counts": val["counts"]},
        "popularity_validation_k10": pop["metrics_by_k"]["10"],
        "item_knn_validation_k10": knn["metrics_by_k"]["10"],
        "eligible_users_count": pop["eligible_users_count"],
        "test_evaluation_rejected": sealed,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            from .artifacts import prepare

            res = prepare(args.data_dir, args.output_dir, args.archive)
            m = res["manifest"]
            _emit(
                {
                    "status": res["status"],
                    "protocol_version": m["protocol_version"],
                    "artifacts": str(args.output_dir),
                    "dataset": m["dataset"],
                    "split": m["split"],
                    "manifest_fingerprint": m["manifest_fingerprint"],
                }
            )
        elif args.command == "validate":
            from .artifacts import validate

            _emit(validate(args.artifacts))
        elif args.command == "evaluate":
            _emit(run_evaluate(args.artifacts, args.model, args.split), args.output)
        elif args.command == "smoke":
            _emit(run_smoke())
        elif args.command == "train":
            from .r2 import run_train

            sel = run_train(args.artifacts, args.output_dir)
            _emit(
                {
                    "status": "selection_locked",
                    "selection_manifest": str(args.output_dir / "selection_manifest.json"),
                    "manifest_sha256": sel["manifest_sha256"],
                    "selected": sel["selected"],
                    "trials": [
                        {"config_id": t["config"]["config_id"], "status": t["status"],
                         "train_seconds": t["train_seconds"], "metrics_k10": t["metrics_by_k"]["10"]}
                        for t in sel["trials"]
                    ],
                    "baseline_reconciliation": sel["r1_baseline_reconciliation"]["status"],
                    "test_labels_accessed": sel["test_labels_accessed"],
                }
            )
        elif args.command == "benchmark":
            from .r2 import run_benchmark

            _emit(run_benchmark(args.artifacts, args.experiment_dir), args.output)
        elif args.command == "export":
            from .serving import export_bundle

            manifest, _ = export_bundle(args.artifacts, args.output_dir)
            _emit({"status": "exported", "bundle": str(args.output_dir), "schema": manifest["schema"],
                   "dataset_kind": manifest["dataset_kind"], "manifest_sha256": manifest["manifest_sha256"],
                   "serving": manifest["serving"], "files": {k: v["sha256"] for k, v in manifest["files"].items()},
                   "reload_parity": manifest["reload_parity"]})
        elif args.command == "recommend":
            from .serving import RecommenderService

            _emit(RecommenderService.load(args.bundle).recommend(args.user_id, args.top_k, args.model))
        elif args.command == "latency":
            from .serving import RecommenderService, run_latency

            _emit(run_latency(RecommenderService.load(args.bundle)), args.output)
        return 0
    except RecError as e:
        print(f"ERROR {e.code}: {e.message}", file=sys.stderr)
        return e.exit_code
    except Exception:  # noqa: BLE001 - unexpected internal failure
        traceback.print_exc()
        print("ERROR INTERNAL: unexpected internal failure", file=sys.stderr)
        return 1

# personalized_recommender: offline recommendation and ranking benchmark (MovieLens-1M)

**Problem.** Rank unseen movies for each user from implicit feedback (rating >= 4 is a positive) and measure the
ranking honestly: with a global temporal split and an exact full-catalog evaluation, not sampled negatives.

**Approach.** Four rankers implemented in NumPy (no recsys library): Popularity, Item-KNN, implicit ALS and BPR-MF,
evaluated under one protocol (split, candidate set, metrics), plus a saved-model path for offline recommendations.
It is **not** an online service: no HTTP, UI or database.

**Principal result (single seed, no significance tests).** On the held-out TEST window (1,187 eligible users)
Item-KNN has the highest NDCG@10 (0.2431), ALS the highest Recall@10 (0.0709) and coverage, and BPR-MF is close to
Item-KNN on NDCG@10 (0.2400). See [Results](#results-on-the-test-split).

**Key limitations.** One seed and 8 validation configurations, no confidence intervals; offline metrics only;
ML-1M only; most VALIDATION users have no TRAIN history (see [Fallback rates](#fallback-rates-validation-vs-test)).
See [Known limitations](#known-limitations).

**Install and test.** `pip install '.[test]'` then `python -m pytest -q` (see [Install](#install-python-312) and [Tests](#tests)).
MovieLens data are not included; the terms are in [Dataset and license](#dataset-and-license).

**What it contains**
- Strict `ratings.dat` ingestion, a global temporal split (train / validation / sealed test), and an exact
  **full-catalog** Top-K evaluator (no sampled negatives; Recall, HitRate, NDCG, Coverage at K = 5/10/20).
- Four ranking methods (`--model` values):

  | Method | `--model` | Notes |
  |---|---|---|
  | Popularity | `popularity` | positive-count ranking; also the cold-start fallback |
  | Item-KNN | `item_knn` | binary-cosine, top-50 neighbours; **default** |
  | Implicit ALS | `als` | Hu–Koren–Volinsky, ALS-01 (32 factors, α=10) |
  | BPR-MF | `bpr` | BPR-02 (32 factors, lr=0.05) |

- A fixed 8-trial validation search (4 ALS + 4 BPR configs, one seed) with a locked selection manifest and one
  sealed TEST benchmark.
- **Saved model bundles** (`export`): an integrity-checked `p02-serving-bundle/1` directory holding all four fitted
  models, **reloadable** without MovieLens files, with same-fit reload parity checks.
- **Offline ranking / inference** (`recommend`, `RecommenderService.load(...).recommend(...)`) for any of the four
  methods or `default`.
- CLI with nine subcommands: `prepare`, `validate`, `evaluate`, `smoke`, `train`, `benchmark`, `export`, `recommend`, `latency`.

**Default method.** `--model default` resolves to `item_knn`. It was chosen on VALIDATION NDCG@10
(Item-KNN 0.265665 vs BPR-02 0.265618). This 0.000047 difference is a **near-tie**: one seed, no confidence
intervals, no paired significance test. `item_knn` is the default because the fixed selection rule picked it,
not because it is shown to generalize better than BPR. On TEST the methods trade places by metric (see Results).

**Fallback caveat.** On VALIDATION 611 of 1,103 eligible users (55.4%) have no TRAIN history and get the same Popularity
ranking under every method, versus 31 of 1,187 (2.6%) on TEST. See [Fallback rates](#fallback-rates-validation-vs-test).

**Quick start (runs on the bundled synthetic fixture; no MovieLens needed).**

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install .
FIX=$(python -c "import personalized_recommender,pathlib;print(pathlib.Path(personalized_recommender.__file__).parent/'fixtures/synthetic_ml')")
python -m personalized_recommender smoke
python -m personalized_recommender prepare --data-dir "$FIX" --output-dir /tmp/p02_art
python -m personalized_recommender export  --artifacts /tmp/p02_art --output-dir /tmp/p02_bundle
python -m personalized_recommender recommend --bundle /tmp/p02_bundle --user-id 1 --top-k 5 --model als
```

## Dataset and license

**Code license: MIT.** Original project code, documentation and synthetic fixtures are licensed under the [MIT License](LICENSE), Copyright (c) 2026 Nguyen Quang Huy. This source-only grant excludes upstream datasets, models, dependencies and excluded derived artifacts; their separate terms remain applicable.

**Dataset terms.**

You must obtain the official GroupLens **MovieLens 1M** dataset yourself:
<https://grouplens.org/datasets/movielens/1m/> (archive
<https://files.grouplens.org/datasets/movielens/ml-1m.zip>, README/license
<https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>). The GroupLens usage license
has these terms:
- Research use only; do **not** redistribute the data (or derived row-level data) without separate permission from GroupLens.
- Commercial or revenue-bearing use needs prior permission.
- Cite the dataset: F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context", ACM TiiS 5(4), 2015.
- Always check the current terms in the GroupLens README linked above; they govern, not this summary.

This repository contains **no MovieLens data**. Raw files and prepared row-level artifacts are
git-ignored and must stay local.

Placement: extract the archive so that `PATH_TO_ML_1M/ratings.dat` exists, e.g. `~/data/ml-1m/`.
`movies.dat`, if present, is used only for a referential-integrity check. `users.dat` is never read.
`prepare` records the SHA256 and size of the files plus the counts it observed. It also compares
them against the reference identity of the official archive:
- archive SHA256 `a6898adb…4e20` (MD5 `c4d9eecf…0906`, matching the GroupLens-published `.md5`);
- `ratings.dat` SHA256 `506d64ca…6e0f`;
- 1,000,209 ratings from 6,040 users.

Pass `--archive ml-1m.zip` to also hash the archive and check that its `ratings.dat` is byte-identical.

**Parser policy (strict).**
- Every line must be `UserID::MovieID::Rating::Timestamp\n`.
- IDs must be positive integers with no leading zeros. Rating must be 1..5. Timestamp is Unix seconds.
- These are rejected with exit code 2:
  - malformed records or separators;
  - non-integer fields;
  - out-of-range values;
  - an empty file or one without a trailing newline (treated as truncated);
  - duplicate `(user_id, movie_id)` pairs;
  - a missing `movies.dat` reference.
- Nothing is skipped, deduplicated or coerced.

## Protocol

Identifiers such as `P02-R1-ACCEPTANCE-v1.0`, `p02-r2-selection-manifest/1` and "R1/R2/R3" keys in manifests are serialized stage labels (protocol, selection and serving stages); they are hashed into fingerprints and are kept for artifact compatibility.

- **Feedback:** a rating ≥ 4 is a POSITIVE implicit interaction. A rating of 1–3 is OBSERVED
  NON-POSITIVE. An unobserved pair is UNKNOWN and is never treated as a negative.
- **Split:**
  - Sort the N events by `(timestamp, user_id, movie_id)`.
  - `T_val = r[floor(0.8N)].ts`, `T_test = r[floor(0.9N)].ts` (0-based), with `T_val < T_test` required; otherwise the run fails with `UNSPLITTABLE_CUTOFFS`.
  - TRAIN `t < T_val`; VALIDATION `T_val ≤ t < T_test`; TEST `t ≥ T_test`.
  - Partitions are by timestamp, so tied timestamps never straddle two partitions.
- **Snapshots:**
  - The validation stage uses the prefix TRAIN only. All statistics, the catalog, seen sets and model fits come from TRAIN.
  - The test stage (used by `benchmark`) uses TRAIN+VALIDATION.
  - TEST is retained and checksummed but is **sealed**: `evaluate --split test` exits 2.
- **Candidates:**
  - `C_s` = items with ≥ 1 observed rating in the prefix. This includes items with only non-positive ratings; items that appear only in the future are excluded.
  - `Seen_s(u)` = all items the user rated in the prefix, at any rating.
  - `Candidates_s(u) = C_s \ Seen_s(u)`. All of them are ranked; there is no sampling.
  - `GT_s(u)` = the user's held-out positives that lie in `Candidates_s(u)`. Held-out positives on items not in the catalog are counted as `out_of_catalog_positive_count`.
- **Eligibility:** `|GT| ≥ 1` and `|Candidates| ≥ 1`. Exclusions are counted by reason. Users with no prefix history can be eligible (they get the Popularity fallback).
- **Cold-start:** based on the count of *all* prior observed prefix ratings. Cohorts are `0`, `1..5` (and `0..5`), plus `6+` for warm users. An empty cohort reports `n_users: 0` with `null` metrics.
- **Metrics:**
  - K ∈ {5, 10, 20}; `L_K` = top `min(K, |Candidates|)`.
  - Recall = hits/|GT|. HitRate = 1[hits > 0].
  - NDCG = DCG/IDCG, with `DCG = Σ 1/log2(rank+1)` over hits and `IDCG = Σ_{r ≤ min(K,|GT|)} 1/log2(r+1)`.
  - Metrics are **macro-averaged** over eligible users.
  - Coverage@K = |∪ top-K over eligible users| / |C_s|.
  - No eligible users ⇒ `NO_EVALUABLE_USERS`.
  - General ranking order is score DESC, then numeric movie_id ASC.
- **Popularity:** `pop(i)` = number of positive prefix events for item i. Ranked by pop DESC, then movie_id ASC.
- **Item-KNN:**
  - Binary cosine over positive prefix user sets: `|U_i∩U_j| / sqrt(|U_i||U_j|)`. The value is 0 if either set is empty, and `sim(i,i) = 0`.
  - Each seed j keeps its top-50 neighbors with strictly positive similarity, ordered by sim DESC, then id ASC.
  - Score: `s(u,i) = Σ_{j∈Pos(u), i∈N50(j)} sim(i,j)`. Ranked by s DESC, then pop DESC, then movie_id ASC.
  - Users with no positive history, or with all scores at 0, get the exact Popularity ranking.

Evaluator API: any object with `fit(prefix)` and `rank(user_id, candidates) -> full permutation`
can be passed to `evaluator.evaluate(stage, ranker)`. The evaluator rejects any ranking that is
not a permutation of the full candidate set.

## Install (Python 3.12)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install .            # runtime: numpy, threadpoolctl
pip install '.[test]'    # adds pytest
```

Dependency ranges are in `pyproject.toml` (`numpy>=2.2,<2.3`, `threadpoolctl>=3.6,<3.7`, `pytest>=8.3,<8.4` for tests).
No exact-pin lock file is provided; results were produced with NumPy 2.2.6 on CPython 3.12.3 and factor values
can differ across BLAS builds.

## CLI

Baseline commands (need MovieLens-1M locally, except `smoke`):

```bash
python -m personalized_recommender prepare  --data-dir PATH_TO_ML_1M --output-dir ARTIFACT_DIR [--archive ml-1m.zip]
python -m personalized_recommender validate --artifacts ARTIFACT_DIR
python -m personalized_recommender evaluate --artifacts ARTIFACT_DIR --model popularity --split validation --output popularity.json
python -m personalized_recommender evaluate --artifacts ARTIFACT_DIR --model item_knn   --split validation --output item_knn.json
python -m personalized_recommender smoke     # bundled synthetic fixture only; no network/real data
```

Exit codes:
- `0`: success.
- `2`: malformed arguments, invalid or missing data, corrupt artifacts, protocol/identity mismatch, or `--split test` (`TEST_HOLDOUT_SEALED`).
- `1`: unexpected internal failure.

Errors print as `ERROR <CODE>: message` on stderr.

**Artifacts** (`ARTIFACT_DIR`):
- `train.npy`, `validation.npy`, `test.npy`: int64 `(user, movie, rating, ts)` arrays in canonical order. These are local derived data; do not publish them.
- `manifest.json`: protocol version, dataset provenance, cutoffs, counts, per-file SHA256 plus content fingerprints, a split fingerprint and a self-fingerprint.

`validate` re-verifies all of these. It also re-derives the cutoffs and memberships from the
partition union. It never regenerates anything.

If `prepare` targets an existing artifact directory, the directory must verify and must match the
input exactly; otherwise it fails with `CACHE_IDENTITY_MISMATCH`.

**Report JSON (`evaluate`):**
- `protocol_version`, `stage`, `snapshot`, `model`;
- `dataset_identity`, `split_identity` (cutoffs, split/manifest fingerprints, train/validation SHA256);
- `k_values`, `candidate_protocol`;
- `diagnostics` (catalog size, eligible/excluded counts by reason, out-of-catalog positives);
- `metrics_by_k` (keys `"5"`, `"10"`, `"20"`, each with recall/hitrate/ndcg/catalog_coverage);
- `eligible_users_count`, `cold_start` (per-cohort `n_users` + `metrics_by_k`).

Output is canonical (sorted keys) with no timestamps, so reruns are byte-identical.

## Tests

Tests spawn `python -m personalized_recommender`; `tests/conftest.py` puts `src/` on `PYTHONPATH`, so they also run from a fresh
checkout without installing (needs only numpy, threadpoolctl, pytest).

```bash
pip install '.[test]'
python -m pytest -q                                   # synthetic tests (offline)
P02_ML1M_DIR=~/data/ml-1m P02_ML1M_ARCHIVE=~/data/ml-1m.zip python -m pytest -q   # + real-data tests (needs MovieLens-1M)
```

Without the variables, 12 tests skip, each with an explicit reason: they need the official MovieLens-1M data (not shipped).
All other tests run offline on synthetic data.

## ALS, BPR-MF, selection and TEST benchmark

These build on the protocol above without changing it: same split, `Stage`/`Prefix`, candidates,
evaluator, metrics and baselines; `evaluate --split test` stays sealed.

**Models.** Both are pure-NumPy CPU code in `models.py`; no external recsys library is used.

- **Positive matrix and mappings.**
  - The positive matrix holds the prefix's rating ≥ 4 events, as a binary matrix.
  - Item index: every prefix catalog item in ascending id order, including items with zero positives.
  - User index: users with ≥ 1 prefix positive, in ascending id order.
  - A fresh mapping and a fresh model are built for every fit.
- **Implicit ALS (Hu–Koren–Volinsky).**
  - Preference: p=1 at prefix positives and 0 elsewhere.
  - Confidence: c=1+α at positives and exactly 1 elsewhere. Ratings 1–3 get no extra confidence.
  - Each half-step solves exactly `(FᵀF + α·F_SᵀF_S + λI) x = (1+α)·Σ_{s∈S} f_s` with λ=0.05.
  - α is applied as-is: the binary matrix is not rescaled.
  - Initialization is N(0, 0.01²) with seed 42; one BLAS thread.
- **BPR-MF.**
  - Minibatch SGD (batch 1024) on `−ln σ(x_u·(y_i−y_j)) + λ(‖x_u‖²+‖y_i‖²+‖y_j‖²)`, with λ=0.01.
  - Each epoch visits every prefix (u, i⁺) pair once, in a seeded random order.
  - The contrast j is drawn uniformly from the same prefix catalog and rejected if it is one of u's positives.
  - An observed 1–3 item may be a contrast. It is a training sample, not a verified negative.
- **Scoring.**
  - Score = `x_u·y_i` over the full candidate set; ranking is score DESC, then movie_id ASC.
  - Users with no factor (no prefix positive, or unknown) get the exact stage Popularity ranking.
  - Negative scores are valid and do not trigger the fallback.
  - NaN/Inf factors or scores, or a wrong factor shape, raise `ModelIntegrityError` (exit 1).

**Fixed validation grid (seed 42, 15 iterations, CPU).**

| Config | factors | alpha / lr | reg |
|---|---|---|---|
| ALS-01 | 32 | α=10 | 0.05 |
| ALS-02 | 32 | α=40 | 0.05 |
| ALS-03 | 64 | α=10 | 0.05 |
| ALS-04 | 64 | α=40 | 0.05 |
| BPR-01 | 32 | lr=0.01 | 0.01 |
| BPR-02 | 32 | lr=0.05 | 0.01 |
| BPR-03 | 64 | lr=0.01 | 0.01 |
| BPR-04 | 64 | lr=0.05 | 0.01 |

**Selection** uses VALIDATION only, and only exact equality counts as a tie.
- Per family: max NDCG@10, then max Recall@10, then max HitRate@10, then the earliest config ID.
- Default method: the same ordering over {Popularity, Item-KNN, best ALS, best BPR}, with final ties broken by the preference order popularity < item_knn < als < bpr.

**Commands.**

```bash
python -m personalized_recommender train     --artifacts ARTIFACT_DIR --output-dir EXPERIMENT_DIR
python -m personalized_recommender benchmark --artifacts ARTIFACT_DIR --experiment-dir EXPERIMENT_DIR --output FINAL_BENCHMARK.json
```

- **`train`**
  1. Verifies the TRAIN/VALIDATION artifacts. `test.npy` is checked only by its byte SHA256 and is never decoded.
  2. Recomputes the Popularity and Item-KNN baselines and reconciles them with their recorded VALIDATION values.
  3. Runs all 8 trials.
  4. Writes `selection_manifest.json` (read-only). The manifest contains:
     - lineage, code identity (SHA256 of the package's `.py` sources), versions, seed and threads;
     - all 8 trials;
     - the selections and the rules used;
     - a canonical SHA256 that excludes its own field.

  An existing manifest is never overwritten (`SELECTION_ALREADY_LOCKED`).
- **`benchmark`**
  - Before decoding any TEST row it checks:
    - the manifest SHA256, protocol/schema and `FINALIZED` status;
    - the code identity and data lineage;
    - that the table is exactly the fixed 8-trial grid;
    - that the recorded choices follow the rules.
  - Then it builds `build_test_stage(train, validation, test)` (prefix = TRAIN+VALIDATION).
  - It freshly refits Popularity, Item-KNN, the selected ALS and the selected BPR, and evaluates all four on the same users, candidates and ground truth.
  - It accepts no hyperparameter or method overrides.
  - Output includes:
    - K=5/10/20 metrics, coverage, cold-start cohorts and fallback counts;
    - the head/tail split (pre-test positive count DESC, id ASC; head = first ⌈0.2·|C|⌉ items);
    - tail Recall@10 (macro over eligible users with tail positives) and tail exposure@10 (tail positions / emitted Top-10 positions);
    - ALS/BPR minus baseline deltas;
    - runtimes and versions.
- **Exit codes.**
  - 2: missing, corrupt or mismatched manifest or artifacts, or `benchmark` run before `train`.
  - 1: model fitting failure or non-finite output.
  - The legacy `evaluate --split test` still exits 2.

Experiment outputs (manifest, benchmark JSON) belong in an ignored local directory and are never committed.

**Results discipline.** The TEST table below is one run, one seed, with no confidence intervals or significance tests.
Recall@10 (~0.06) and NDCG@10 (~0.24) differ in scale because Recall is divided by all held-out positives (many per user)
while NDCG's ideal DCG is limited to min(K, |GT|).

**Limitations of the search and benchmark.**
- The search covers 8 configurations with one seed each. No significance testing is done.
- Fallback users have no prefix positives; under the global-time split they are a large share of eligible users.
- Runtimes are machine-dependent metadata.
- Factor values may differ across BLAS builds or hardware.

## Serving bundle, offline recommendations, CPU latency

`export` adds no new model, search or evaluation. It refits the four selected methods once on **TRAIN+VALIDATION** (timestamp < `T_test`), saves them, and serves them offline.

**Selection (fixed, not recomputed by `export`).** Default method **`item_knn`**; ALS-01
(32 factors, α=10, λ=0.05, 15 it., seed 42); BPR-02 (32 factors, lr=0.05, λ=0.01, 15 it., seed 42, batch 1024).
The selection manifest self-hash recorded in every bundle is `d0e3316487c05d6255b8304acb774154142b3adc8c7e4212075002b0f20b4f99`.
The default was chosen on VALIDATION NDCG@10 (item_knn 0.265665 vs BPR-02 0.265618, a 0.000047 margin).

## Results on the TEST split

Single run on the sealed TEST split, 1,187 eligible users; offline, observational, no significance test, no online-uplift claim.

| Method | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Popularity | 0.0574 | 0.2185 | 0.038 |
| Item-KNN (default) | 0.0636 | 0.2431 | 0.079 |
| ALS-01 | 0.0709 | 0.2136 | 0.262 |
| BPR-02 | 0.0676 | 0.2400 | 0.176 |

Item-KNN and BPR-02 differ by 0.003 NDCG@10 on TEST (single seed, untested); read the pair as close rather than ranked. ALS had the highest Recall@10 and coverage; Item-KNN the highest NDCG@10. `export` does not rerun TEST, and the default was not changed after seeing TEST.

```bash
python -m personalized_recommender export    --artifacts ARTIFACT_DIR --output-dir BUNDLE_DIR      # refuses to overwrite
python -m personalized_recommender recommend --bundle BUNDLE_DIR --user-id 42 --top-k 10 --model default
python -m personalized_recommender latency   --bundle BUNDLE_DIR --output LATENCY.json
```

```python
from personalized_recommender.serving import RecommenderService
service = RecommenderService.load("/path/to/p02_serving_bundle")
result = service.recommend(user_id=42, top_k=10, model="default")   # default | popularity | item_knn | als | bpr
```

The CLI has **nine** subcommands: `prepare`, `validate`, `evaluate`, `smoke`, `train`, `benchmark`, `export`, `recommend`, `latency`.
Exit codes: `0` success; `2` malformed arguments, invalid user/top-k/model, missing/corrupt/incompatible bundle or prepared input;
`1` unexpected internal failure or non-finite scoring. `recommend` prints exactly one JSON document on stdout.

**Bundle `p02-serving-bundle/1`** (a local directory; git-ignored; never put it in a source archive, it contains row-level histories):
`manifest.json`, `history.npz`, `popularity.npz`, `item_knn.npz`, `als.npz`, `bpr.npz`. Numeric `.npz` only, loaded with
`allow_pickle=False`; no pickle/joblib.
- `history.npz`: sorted pre-TEST catalog, sorted history users, CSR of **all observed** items (ratings 1–5) and CSR of positives (≥4).
- `popularity.npz`: positive counts for every catalog item (including zero). `item_knn.npz`: the exact top-50 positive-similarity neighbour lists (CSR).
- `als.npz`, `bpr.npz`: ordered raw user/item IDs and the learned float64 user/item factors.
- `manifest.json`: schema, `dataset_kind` (`official_movielens_1m` or `synthetic_fixture`), environment, provenance anchors (protocol/identity IDs and hashes), selection provenance,
  dataset/split lineage, prefix, counts, per-file SHA256 + array shapes/dtypes, and a manifest self-hash (SHA256 of canonical JSON without that field).
- `RecommenderService.load` verifies schema, self-hash, file inventory, every file hash, dtypes/shapes/ranges, sorted unique mappings, finite values,
  history/popularity/neighbour coherence and the selected configs. Anything else is rejected with an identified error (exit 2); nothing is repaired or refit.
  SHA256 detects corruption; it is not authenticity against someone who rewrites files and hashes.
- Loading needs only NumPy and the bundle directory: no MovieLens files, prepared `.npy`, experiment directory or network. A moved or copied bundle loads.

**Semantics.** Candidates = pre-TEST catalog minus every item the user observed in TRAIN+VALIDATION (any rating); always on.
- Popularity: positive count DESC, movie_id ASC.
- Item-KNN: similarity sum as in the protocol, score DESC, popularity DESC, movie_id ASC. Falls back to Popularity if the user has no positive history or all scores are 0.
- ALS/BPR: `dot(user_factor, item_factor)` DESC, movie_id ASC; Popularity fallback for users without a trained factor. Negative scores are valid.
- Unknown user IDs get the full catalog with Popularity ranking. If `top_k` exceeds the candidates all are returned; no candidates → empty list.
- Output: `schema_version`, `user_id`, `requested_model`, `resolved_model`, `top_k`, `returned_count`, `fallback_used`, `score_semantics`,
  `recommendations[{movie_id, rank, score}]`. **Scores are model-specific ranking signals** (positive counts, similarity sums or latent dot products), not
  probabilities, star ratings or labels. When the Popularity fallback is used the scores are the Popularity counts.

**Same-fit reload parity.** `export` keeps the in-memory fitted rankers and, after writing, compares them with the freshly loaded bundle for all four methods,
K ∈ {5,10,20}, known-positive, zero-positive and unknown users: exact movie IDs, order, ranks and fallback flags, scores within rtol=atol=1e-9.
Independently retrained weights are not compared.

**Latency protocol (`latency`; descriptive, no SLA).** Default `item_knn`, `top_k=10`, one process, concurrency 1, BLAS threads 1. The first 100 ascending
user IDs with ≥1 positive observation; request *i* uses `user_ids[i % 100]`; 100 warmup calls, then exactly 1,000 timed `service.recommend` calls with
`time.perf_counter_ns()`. Excluded: import, bundle load, CLI start, fitting, JSON serialization, printing, disk I/O. Percentiles: `numpy.percentile(method="linear")`.
On the development machine (AMD Ryzen Threadripper PRO 5955WX, Python 3.12.3, NumPy 2.2.6) one real ML-1M run gave
mean 0.274 ms, p50 0.245 ms, p95 0.456 ms, p99 0.758 ms (max 1.106 ms). These are single-machine numbers, not a throughput or production-latency claim.

**Fixture-only run (synthetic fixture, no MovieLens needed).** `prepare --data-dir <package>/fixtures/synthetic_ml` then `export`/`recommend` exercise the same pipeline; those
bundles are marked `dataset_kind="synthetic_fixture"` and never claim official identity.

**Real-data run (needs MovieLens-1M locally; about a minute of CPU for `export`).**
```bash
python -m personalized_recommender prepare --data-dir ~/data/ml-1m --output-dir ART --archive ~/data/ml-1m.zip
python -m personalized_recommender export --artifacts ART --output-dir BUNDLE     # ~1 min, CPU
cp -r BUNDLE /elsewhere && cd / && python -m personalized_recommender recommend --bundle /elsewhere/BUNDLE --user-id 1
```
Real-data tests read `P02_ML1M_DIR` and `P02_ML1M_ARCHIVE`.

**Serving limitations.** Offline fixed snapshot: no online profile updates, no content features for new items, cold users get Popularity; the bundle is a
one-machine CPU artifact whose factor values may differ across BLAS builds when refit; Item-KNN export is the sparse form of a dense 3.7k² fit-time
matrix; no concurrency, throughput or production-serving claims; ML-1M only, with an implicit signal derived from explicit star ratings. The bundle holds
row-level histories derived from MovieLens and must not be redistributed (GroupLens license).

## Fallback rates (validation vs test)

Fallback = the user has no positive (rating ≥ 4) interaction in the prefix, so ALS, BPR and Item-KNN all return the exact
Popularity ranking. Counts are over **eligible** users (≥ 1 held-out positive and ≥ 1 candidate), recomputed from the prepared
ML-1M artifacts with the unchanged split:

| Stage (prefix → held-out) | Eligible | Fallback | % | Zero prefix history | Only ratings 1–3 | Personalized |
|---|---|---|---|---|---|---|
| VALIDATION (TRAIN → VALIDATION) | 1,103 | 611 | 55.4 | 611 | 0 | 492 |
| TEST (TRAIN+VALIDATION → TEST) | 1,187 | 31 | 2.6 | 29 | 2 | 1,156 |

No eligible user lacks candidates on either stage. The measured cause is the global temporal split: validation users are
mostly people whose first ratings occur after the TRAIN cutoff, whereas by TEST most active users already have history in
TRAIN+VALIDATION. The repo does not establish anything beyond this (e.g. why the user mix changes over time). Consequences:
the VALIDATION selection signal (including the Item-KNN/BPR near-tie) comes from 492 personalized users, while the TEST
population is mostly warm. The split and protocol were not changed.

## Known limitations

- The global-time split is not a causal online experiment, and user participation in each window is uneven. On ML-1M, most validation users have no TRAIN history.
- The prefix-only catalog makes new items non-recommendable by design; the excluded positives are reported.
- A non-positive rating is not proof of dislike, and an unobserved pair is not a verified negative.
- Offline Recall/NDCG do not guarantee online lift. No quality or latency thresholds are implied.
- Item-KNN materialises a dense item×item matrix (≈ 3.7k² floats for ML-1M). This is fine for ML-1M but is not built for much larger catalogs.

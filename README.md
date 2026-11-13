# Amazon ML Challenge 2026: Business Entity Resolution

For each **Source-1 (S1)** business record, find every matching record in **Source 2 / Source 3**.
Scoring is **macro F0.5 per S1 entity**. Singletons count: an S1 with no true match scores 1.0 only
for an empty prediction.

**Best submission: v6, public leaderboard F0.5 = 0.971.**

| Version | What changed | Validation F0.5 (US/India, held out) | Leaderboard |
| --- | --- | --- | --- |
| v1 | MinHash-LSH blocking + two-stage XGBoost | 0.9669 | 0.952 |
| v3 | + name × house-number blocking pass, + acronym / rare-token features | 0.9735 | 0.958 |
| v5 | + name × address-token blocking pass, + per-country percentile features | 0.9766 | not submitted |
| **v6** | **v5 + Ditto cross-encoder re-scoring of uncertain pairs** | **0.9800** | **0.971** |

The pipeline runs on a 16 GB laptop (Apple M4). It uses no external data or APIs. All models are
MIT/Apache-2.0: XGBoost, and the team's fine-tuned XLM-RoBERTa "Ditto" model.

---

## 1. Pipeline (v6)

```
S1/S2/S3 ─ normalize ─► blocking (same country only), union of 5 passes:
                         MinHash-LSH: name · name consonant-skeleton · address
                         exact keys:  name token × house number · rare name token × rare address token
                ─► stage-1 XGBoost ranker ─► top-25 candidates per S1         (= candidate_pairs.tsv)
                ─► 124 features: string/IDF/house-number/competition + per-country percentiles
                ─► stage-2 XGBoost  p(match)
                ─► Ditto (XLM-R cross-encoder) re-scores pairs with p in [0.05, 0.95):
                   p ← logit-average(p, p_ditto)
                ─► each S2/S3 record → its best-scoring S1 only;  keep p ≥ 0.8 ─► matching_results.tsv
```

1. **Normalization** (`src/normalize.py`, `src/data.py`)
   - NFC, lowercase, Latin accent stripping.
   - **Romanisation of all 9 Indic scripts** through their shared ISCII layout
     (`नोवो इंश्योरेंस` → `novo inshyorens`).
   - Leetspeak repair, domain stems (`foo.com` → `foo`), street-type and state canonicalisation.
   - **Data-driven stop tokens** per country and split. French legal forms (`sarl`, `sas`, `eurl`)
     were found automatically on the test set.
   - A **consonant skeleton** (`management` / `mainejament` → `mngmnt`).
2. **Blocking** (`src/lsh.py`, `src/blocking.py`)
   - numba MinHash over within-token character 3-grams, which makes it order-invariant.
   - The country id is hashed into every band key.
   - Two exact-key passes rescue **generic names**, whose LSH buckets are too large to use:
     - `(name token, house number)`: candidate recall 94.95% → 96.49%;
     - `(rare name token, rare address token)`: candidate recall → **97.18%**.
3. **Stage-1 ranker** (`src/candidates.py`): XGBoost on the cheap blocking signals; keeps the top 25 per S1.
4. **Stage-2 features** (`src/features.py`)
   - rapidfuzz similarities and IDF token overlap.
   - House-number features; blank addresses are treated as *missing*, not 0.
   - **Competition features**: each S2/S3 record belongs to at most one S1, so the features say how this
     S1 ranks among all S1s competing for the record. These are the most important features.
   - Acronym and rare-token name similarity.
   - **Per-country percentile copies** of all continuous features, with tables built from each split's
     own unlabeled pairs. This is robust to the unseen country (France).
5. **Stage-2 XGBoost** (`src/train.py`): threshold tuned for macro F0.5 on an S1-level held-out fold.
6. **Ditto re-scoring** (`src/ditto_rescore.py`): your teammate's fine-tuned XLM-R cross-encoder, used
   only where XGBoost is unsure (1.34M of 43M test pairs).
   - On validation it added **+0.39**: AUC in the band 0.947 vs 0.922 for XGBoost.
   - On the leaderboard it added **≈ +1.0**. It is multilingual, so it also helped **France**, which has
     no training data.

## 2. Reproduce v6

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install libomp                                   # macOS only, for xgboost
ln -s /path/to/student_resource/dataset data           # challenge data (not in this repo)
# Ditto model (not in this repo, 556 MB): unzip ditto_er_model.zip so that
#   ditto/ditto_er_model/{ditto_infer.py, model/, serializer.json} exists
```

```bash
export ER_TAG=v5                                       # all v5/v6 steps use this tag
.venv/bin/python -m src.data train test                # normalize + cache            (~3 min)
.venv/bin/python -m src.train                          # blocking, stage 1+2, threshold (~3.5 h on M4)
.venv/bin/python -m src.predict                        # v5 test predictions          (~1 h)
.venv/bin/python -m src.ditto_rescore v5 v6            # Ditto on uncertain pairs -> output_v6/ (~3.7 h on M4)
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching output_v6/matching_results.tsv --candidate output_v6/candidate_pairs.tsv \
    --test-dir data/test --check-ids
```

- `models_v5/` contains the trained models used for v6. If they are present, `src.train` reuses stage 1;
  delete them to retrain.
- Expensive steps write resumable chunks to `cache/`.
- The submitted file had SHA-256 `590adc5773800a22dd9840b0d4ec761e49a94fd0e73da784876bd9eb7680cfce`.

## 3. Key findings from the data (EDA)

- **Sizes:**

  | Split | S1 | S2 | S3 |
  | --- | --- | --- | --- |
  | Train | 2,206,821 | 5,034,616 | 5,285,603 |
  | Test | 1,732,544 | 4,887,273 | 5,082,316 |

  Test adds **France** (259k S1), which never appears in training.
- **Every S2/S3 record belongs to at most one S1.**
- S1s have 3.46 matches on average; **5.6% are singletons**; 26% of S2/S3 records are unmatched decoys.
- **Country agrees on 100% of true pairs**, so country is a hard block.
- **Only addresses are ever blank** (~3.4% of S2/S3). 97.7% of blank-address records are true
  matches of *some* S1, but branches with identical names make many of them impossible to attribute.
  The model's recall on them is 44%, and every rule we tried to raise it lowered F0.5.
- **Noise:** legal-form swaps, injected accents, leetspeak, `[#123]` tags, token reordering, websites,
  Indic scripts, dropped address parts, perturbed house numbers.
- **France** (test only): region vs department names, acronyms (`Club de Secours` ↔ `CS`), swapped
  category words.
- **Leaderboard vs validation gap:** about 1.5 points for XGBoost-only versions, 0.9 for v6.
  - Simulating orphan clusters explains only 0.14 of it, and the country mix about 0.3.
  - The rest is mostly France.

## 4. Tried and rejected (measured, so they don't need repeating)

| Idea | Result |
| --- | --- |
| Accept the best-name S1 for blank-address records | validation −0.0005 to −0.0010 |
| Give blank-address records to the branch with the fewest matches | 27.9% vs 28.0% random |
| Expected-F0.5 subset selection / per-country thresholds | no gain |
| Self-training on the unseen country (leave-one-country-out bench) | US→India +1.3, India→US −0.5: inconsistent |
| Synthetic training data for the unseen country | US→India 0.8905 → 0.8818: hurts (synthetic noise too easy) |
| Percentile-only features | worse; raw + percentile was kept |
| 2× stage-2 training data | +0.1 per doubling (learning curve), not worth 16 GB of RAM |
| Laya (fine-tuned mmBERT decision model) on top of v6 | AUC 0.61 on uncertain pairs; validation 0.9800 → 0.9798 |
| "One word swapped at the same address" removal (France) | our implementation flagged 99.6%-true pairs: validation −0.01 |
| v7: Ditto also on [0.95, 0.98) + threshold 0.75 | validation +0.0004; not submitted |
| Qwen3-4B LoRA re-scoring | passes its sanity check, but ~5 pairs/s on an M4: needs a GPU |

## 5. Repository layout

```
src/normalize.py      Unicode/Indic normalization        src/data.py         loading, caching, stop tokens
src/lsh.py            numba MinHash + LSH joins          src/blocking.py     5 blocking passes
src/candidates.py     stage-1 ranker, competition feats  src/features.py     stage-2 features (+ percentiles)
src/train.py          folds, training, threshold          src/predict.py      test inference, TSV writing
src/ditto_rescore.py  Ditto re-scoring (v6, v7)           src/metrics.py      official macro F0.5
src/bench_blocking.py blocking recall benchmark           bench/eval_model.py evaluate any extra model vs v6
src/synth*.py, src/laya_*.py   rejected experiments (kept for reference)
models_v5/            trained v5 models used by v6       docs/METHODOLOGY.md  methodology write-up
```

**Validation protocol:**
- S1s are split by a hash of their id: fold 0 → stage 1, fold 1 → validation, folds 2–9 → stage 2.
- Ditto settings were chosen on one half of a validation subset and reported on the other half.
- Ground truth is used only as labels.

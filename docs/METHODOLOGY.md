# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Final submission:** v6, public leaderboard macro F0.5 = **0.971**

---

## 1. Executive Summary

The pipeline has four stages:
1. Unicode/Indic-aware normalization.
2. A five-pass blocking stage: three character-3-gram MinHash-LSH passes plus two exact compound-key
   passes for generic names.
3. A two-stage XGBoost matcher with cross-entity "competition" features and per-country percentile
   features.
4. A fine-tuned multilingual cross-encoder (Ditto, XLM-RoBERTa base) that re-scores the pairs XGBoost is
   unsure about.

Each S2/S3 record is assigned to at most one S1. Held-out macro F0.5 on US/India is **0.980**; the public
leaderboard score is **0.971**.

## 2. Methodology

### 2.1 Problem Analysis

- **Sizes:** train 2.2M S1 / 5.0M S2 / 5.3M S3 (7.6M true pairs); test 1.73M / 4.89M / 5.08M.
  The test set adds **France**, which has no training data.
- **Each S2/S3 record belongs to at most one S1.** S1s average 3.46 matches; 5.6% are singletons;
  26% of S2/S3 records are decoys.
- **Country agrees on 100% of true pairs.**
- **Only addresses are ever blank** (~3.4% of S2/S3). 97.7% of blank-address records are true matches of
  some S1, but identical-name branches make many of them impossible to attribute without an address.
- **Noise:**
  - names: legal forms, injected accents, leetspeak, tags, reordering, websites, nine Indic scripts;
  - addresses: dropped components, perturbed house numbers, state and region variants;
  - France: acronyms and swapped category words.

### 2.2 Solution Strategy

**Approach type:** blocking + two-stage gradient boosting + cross-encoder re-scoring + exclusive assignment.

**Core innovations:**
- order-invariant, script-agnostic MinHash-LSH with a consonant-skeleton pass;
- compound exact keys for generic names;
- competition features that exploit the "at most one S1 per record" structure;
- per-country percentile features for the unseen country;
- cross-encoder re-scoring limited to the uncertain band.

## 3. Candidate Generation (Blocking)

**Normalization:**
- NFC, lowercase, Latin accent stripping.
- ISCII-offset romanisation of Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada
  and Malayalam.
- Leetspeak repair, domain stems, legal-form, street-type and state canonicalisation.
- Data-driven per-country stop tokens: >1.5% of names, >0.4% of addresses, never numbers.
- A consonant skeleton (g/j, c/k/q, v/w merged; vowels dropped).

**Passes** (all restricted to the same country):

| Pass | Key | Notes |
| --- | --- | --- |
| name LSH | name without stop tokens, within-token 3-grams | 48 permutations, bands of 2 rows |
| skeleton LSH | consonant skeleton | 32 permutations, bands of 2 rows |
| address LSH | address without stop tokens, numbers kept | 36 permutations, bands of 3 rows |
| name × number | (name-skeleton token, house number) | buckets ≤ 100 records |
| name × address token | (2 rarest name tokens, 4 rarest address tokens) | buckets ≤ 100 records |

- Pairs with fewer than 2 LSH band hits and no exact-key hit are pruned.
- A stage-1 XGBoost ranker (MinHash similarities, hit counts, within-S1 ranks) keeps the **top 25 per S1**.
- **Candidate pairs:** 43.0M on test (24.8 per S1).
- **Validation candidate recall: 97.18%** (v1: 94.95%).

## 4. Matching Model

**Features (124):**
- **Name:** fuzzy ratios, token-sort and token-set ratios, partial ratio, Jaro-Winkler, core-name and
  skeleton ratios, IDF-weighted token overlap, acronym match, rare-token similarity, website flags.
- **Address:** fuzzy ratios on the full and core address, IDF overlap; house-number Jaccard,
  first-number equality, share of S1 numbers present. Address similarities are set to missing when an
  address is blank.
- **Competition:** how many S1s compete for the record, this S1's rank among them, and the gap to the
  best rival; the within-S1 rank and gap.
- **Per-country percentiles:** a percentile copy of each continuous feature, computed from each split's
  own unlabeled candidate pairs per country.

**Models:**
- **Stage 2:** XGBoost (depth 8, lr 0.08, early stopping on an S1-disjoint split), trained on 400k S1s.
- **Re-scoring:** the team's fine-tuned **Ditto** cross-encoder (XLM-RoBERTa base, Apache/MIT). It runs
  on pairs with XGBoost p in [0.05, 0.95), and the two are combined by a logit average with weight 0.5.

**Decision:** each S2/S3 record goes to its best-scoring S1, then pairs with p ≥ 0.8 are kept.

**Threshold selection:**
- The XGBoost threshold was grid-searched for macro F0.5 on the held-out S1 fold.
- The Ditto band, weight and threshold were chosen on one half of a validation subset and reported on
  the other half.

**Leakage control:** S1-level folds; ground truth used only as labels; no external data or APIs.

## 5. Results & Error Analysis

| Version | Validation F0.5 | Leaderboard |
| --- | --- | --- |
| v1 (LSH + XGBoost) | 0.9669 | 0.952 |
| v3 (+ name × number blocking) | 0.9735 | 0.958 |
| v5 (+ name × address-token blocking, percentile features) | 0.9766 | — |
| **v6 (+ Ditto re-scoring)** | **0.9800** | **0.971** |

- **Ditto's effect:** +0.4 on validation but about +1.0 on the leaderboard. Being multilingual, it
  helped France most.
- **False positives:** mostly another branch of the same company, or a nearby house number.
- **False negatives:**
  - true pairs never proposed by blocking (2.8%);
  - blank-address records of identical-name branches, which are not resolvable from the text;
  - Indic-script names with minimal addresses.

**Tried and rejected** (all measured):
- rules to accept or route blank-address records;
- expected-F0.5 subset selection; per-country thresholds;
- self-training and synthetic data for the unseen country (tested on a leave-one-country-out bench);
- Laya re-scoring (AUC 0.61 on uncertain pairs);
- removing French "one-word-swap" pairs;
- 2× training data (+0.1 per doubling).

## 6. Conclusion

The two largest gains came from **blocking recall**: compound keys that rescue generic names
(+0.65 and +0.31 validation points) and from **multilingual cross-encoder re-scoring**, which also
narrowed the gap on the unseen country. The remaining headroom is in France, which has no labels to
validate against, and in candidate recall for Indic-script names.

## Appendix: Code

Entry points, all run with `ER_TAG=v5`:
1. `python -m src.data train test`
2. `python -m src.train`
3. `python -m src.predict`
4. `python -m src.ditto_rescore v5 v6`

These produce `output_v6/matching_results.tsv` and `output_v6/candidate_pairs.tsv`.

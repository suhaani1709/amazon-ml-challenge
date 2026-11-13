"""End-to-end training + validation on the training split.

S1 entities are split by a hash of their row id (never by pair):
  fold 0     -> stage-1 (blocking ranker) training sample
  fold 1     -> held-out validation (all metrics reported here)
  folds 2..9 -> stage-2 (matcher) training sample
Blocking is run for *all* train S1 so the cross-S1 competition features see the
same competition they will see on the test set. Ground truth is used only as labels.
"""
import json
import os
import time

import numpy as np
import polars as pl
import psutil
import xgboost as xgb

from . import candidates, features
from .data import BASE_TAG, ROOT, TAG, VCACHE, load_gt, prepare
from .metrics import macro_f05

MODELS = os.path.join(ROOT, f"models_{TAG}")
N_STAGE1, N_STAGE2 = 60_000, 400_000


def fold_of(rid):
    return (rid.cast(pl.UInt64) * 2654435761 % 4294967296) % 10


def truth_pairs(df):
    ids = df.select("entity_id", "rid")
    return (load_gt("train").join(ids.rename({"entity_id": "s1_id", "rid": "q"}), on="s1_id")
              .join(ids.rename({"entity_id": "m_id", "rid": "s"}), on="m_id").select("q", "s"))


def decide(scored, thr, rel=0.0):
    """Final matches: each S2/S3 record goes to at most its best-scoring S1, then
    keep pairs with p >= thr and p >= rel * (best p of that S1)."""
    x = scored.filter(pl.col("p").rank("ordinal", descending=True).over("s") == 1)
    x = x.filter(pl.col("p") >= thr)
    if rel > 0:
        x = x.filter(pl.col("p") >= rel * pl.col("p").max().over("q"))
    return x.select("q", "s")


def evaluate(pred, truth, q_ids):
    as_ids = lambda d: d.select(pl.col("q").alias("s1_id"), pl.col("s").alias("m_id"))
    return macro_f05(as_ids(pred), as_ids(truth), q_ids)


def main():
    t0 = time.time()
    os.makedirs(MODELS, exist_ok=True)
    df = prepare("train")
    truth = truth_pairs(df)
    s1 = df.filter(pl.col("src") == 1).select("rid").with_columns(fold_of(pl.col("rid")).alias("fold"))
    rng = np.random.default_rng(0)
    f0 = s1.filter(pl.col("fold") == 0)["rid"].to_numpy()
    val_q = np.sort(s1.filter(pl.col("fold") == 1)["rid"].to_numpy())
    tr_pool = s1.filter(pl.col("fold") >= 2)["rid"].to_numpy()
    st1_q = np.sort(rng.choice(f0, N_STAGE1, replace=False))
    st2_q = np.sort(rng.choice(tr_pool, N_STAGE2, replace=False))

    # ---------------- stage 1 (blocking ranker)
    p1 = os.path.join(ROOT, f"models_{BASE_TAG}", "stage1.json")  # shared with base version
    if os.path.exists(p1):
        m1 = xgb.XGBClassifier(); m1.load_model(p1)
    else:
        print("== training stage-1 ranker", flush=True)
        m1 = candidates.train_stage1(df, "train", st1_q, truth.filter(pl.col("q").is_in(st1_q)))
        m1.save_model(p1)

    # ---------------- candidates for all train S1
    print("== candidate generation (all train S1)", flush=True)
    all_q = np.sort(s1["rid"].to_numpy())
    cdir = candidates.generate_candidates(df, "train", m1, all_q)
    cands = candidates.load_candidates(cdir, np.concatenate([st2_q, val_q]))
    cands = cands.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q", "s"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    vt = truth.filter(pl.col("q").is_in(val_q))
    vc = cands.filter(pl.col("q").is_in(val_q))
    print(f"validation candidate recall: {vc['y'].sum()/vt.height:.4f}  "
          f"({vc.height/len(val_q):.1f} cands/S1)  pairs total {cands.height:,}", flush=True)

    # ---------------- stage 2 features
    if features.ACTIVE is not features.FEATS:
        df = features.add_rare_names(df)
    idf_n = features.token_idf(df, "name_n")
    idf_a = features.token_idf(df, "addr_n")
    fpath = os.path.join(VCACHE, "train_feats_{}.parquet")
    feats = {}
    for name, qs in (("st2", st2_q), ("val", val_q)):
        if os.path.exists(fpath.format(name)):
            feats[name] = pl.read_parquet(fpath.format(name))
        else:
            print(f"== features: {name}", flush=True)
            feats[name] = features.pair_features(cands.filter(pl.col("q").is_in(qs)), df, idf_n, idf_a)
            feats[name].write_parquet(fpath.format(name))
    tr, va = feats["st2"], feats["val"]
    if features.QN:  # per-country percentile features, tables from this split's candidates
        ctry = df.select(pl.col("rid").alias("q"), "country")
        tr, va = tr.join(ctry, on="q", how="left"), va.join(ctry, on="q", how="left")
        pool = pl.concat([tr.select(features.BASE_ACTIVE + ["country"]).sample(min(tr.height, 3_000_000), seed=0),
                          va.select(features.BASE_ACTIVE + ["country"])], how="vertical_relaxed")
        tables = features.build_qn_tables(pool, tr["country"].unique().to_list())
        del pool
        tables.write_parquet(os.path.join(MODELS, "qn_tables_train.parquet"))
        json.dump(features.QN_COLS, open(os.path.join(MODELS, "qn_cols.json"), "w"))
        tr, va = features.add_qn(tr), features.add_qn(va)
        print(f"  percentile features: {len(features.QN_COLS)}  (total features {len(features.ACTIVE)})", flush=True)

    # ---------------- stage 2 model
    print("== training stage-2 matcher", flush=True)
    es_mask = (tr["q"].to_numpy() % 10) == 0  # small S1-disjoint early-stopping set
    Xtr, ytr = features.matrix(tr), tr["y"].to_numpy()
    del tr, feats
    m2 = xgb.XGBClassifier(n_estimators=1500, max_depth=8, learning_rate=0.08, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=10,
                           eval_metric="logloss", early_stopping_rounds=50)
    m2.fit(Xtr[~es_mask], ytr[~es_mask], eval_set=[(Xtr[es_mask], ytr[es_mask])], verbose=200)
    m2.save_model(os.path.join(MODELS, "stage2.json"))
    va = va.with_columns(pl.Series("p", m2.predict_proba(features.matrix(va))[:, 1]))
    va.select("q", "s", "p", "y").write_parquet(os.path.join(VCACHE, "val_scored.parquet"))

    # ---------------- decision tuning on validation (macro F0.5)
    best = None
    for thr in np.arange(0.2, 0.91, 0.05):
        for rel in (0.0, 0.2, 0.4):
            r = evaluate(decide(va, thr, rel), vt, val_q)
            if best is None or r["macro_f05"] > best[0]["macro_f05"]:
                best = (r, float(thr), rel)
    r, thr, rel = best
    no_assign = evaluate(va.filter((pl.col("p") >= thr)).select("q", "s"), vt, val_q)
    oracle = evaluate(vc.filter(pl.col("y") == 1).select("q", "s"), vt, val_q)
    print(f"\nVALIDATION ({len(val_q):,} held-out S1)")
    print(f"  best thr={thr:.2f} rel={rel}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})}")
    print(f"  without one-S1-per-record assignment: macro F0.5 {no_assign['macro_f05']:.4f}")
    print(f"  oracle matcher on these candidates (blocking ceiling): {oracle['macro_f05']:.4f}")
    imp = sorted(zip(features.ACTIVE, m2.feature_importances_), key=lambda z: -z[1])[:15]
    print("  top features:", ", ".join(f"{k}={v:.3f}" for k, v in imp))
    json.dump({"thr": thr, "rel": rel, "k": candidates.K, "val": r,
               "val_candidate_recall": vc["y"].sum() / vt.height},
              open(os.path.join(MODELS, "decision.json"), "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s  rss {psutil.Process().memory_info().rss/1e9:.1f} GB")


if __name__ == "__main__":
    main()

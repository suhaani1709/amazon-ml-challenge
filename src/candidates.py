"""Stage A: LSH blocking -> stage-1 ranker -> top-K candidates per S1.

The stage-1 ranker is a small XGBoost model over the *cheap* blocking signals
(MinHash similarities, band-hit counts and their within-S1 ranks). Its top-K
output is exactly what is written to candidate_pairs.tsv and scored by stage 2.
"""
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

from . import blocking
from .data import BASE_VCACHE

K = 25
S1_FEATS = ["mh_name", "mh_skel", "mh_addr", "hits_name", "hits_skel", "hits_addr",
            "r_n", "r_s", "r_a", "mx_n", "mx_s", "mx_a", "n_raw", "s_src",
            "q_addr_missing", "s_addr_missing", "hits_nn"] + (["hits_tk"] if blocking.TOKKEY else [])


def stage1_features(raw, meta):
    """meta: DataFrame(rid, src, addr_missing) for all records."""
    m = meta.select("rid", "src", "addr_missing")
    raw = (raw.join(m.rename({"rid": "q", "src": "_", "addr_missing": "q_addr_missing"}).drop("_"),
                    on="q", how="left")
              .join(m.rename({"rid": "s", "src": "s_src", "addr_missing": "s_addr_missing"}),
                    on="s", how="left"))
    return raw.with_columns(
        pl.col("mh_name").rank("ordinal", descending=True).over("q").alias("r_n"),
        pl.col("mh_skel").rank("ordinal", descending=True).over("q").alias("r_s"),
        pl.col("mh_addr").rank("ordinal", descending=True).over("q").alias("r_a"),
        pl.col("mh_name").max().over("q").alias("mx_n"),
        pl.col("mh_skel").max().over("q").alias("mx_s"),
        pl.col("mh_addr").max().over("q").alias("mx_a"),
        pl.len().over("q").alias("n_raw"),
    )


def _X(df, feats):
    return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


def train_stage1(df, split, q_rids, truth, P=blocking.PARAMS):
    """truth: DataFrame(q, s) of true pairs."""
    raw, _ = blocking.generate(df, split, q_rids, P, keep_raw=True)
    raw = stage1_features(raw, df)
    raw = raw.join(truth.with_columns(pl.lit(1).alias("y")), on=["q", "s"], how="left") \
             .with_columns(pl.col("y").fill_null(0))
    model = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, n_jobs=10,
                              tree_method="hist", subsample=0.8, colsample_bytree=0.9)
    model.fit(_X(raw, S1_FEATS), raw["y"].to_numpy())
    return model


def competition_features(c):
    """Cross-S1 features: each S2/S3 record belongs to at most one S1 (ground-truth
    structure), so how this S1 ranks among all S1s competing for the same record is
    informative. Computed over *all* S1 in the split (available at inference)."""
    return c.with_columns(
        pl.len().over("s").alias("s_nq"),
        pl.col("s1").rank("ordinal", descending=True).over("s").alias("s_rank"),
        pl.col("s1").max().over("s").alias("s_max"),
        pl.col("s1").rank("ordinal", descending=True).over("q").alias("q_rank"),
        pl.col("s1").max().over("q").alias("q_max"),
        pl.col("s1").sum().over("q").alias("q_sum"),
    ).with_columns(
        (pl.col("s_max") - pl.col("s1")).alias("s_gap"),
        (pl.col("q_max") - pl.col("s1")).alias("q_gap"),
        # best competing *other* S1 score for this record
        pl.when(pl.col("s_rank") == 1)
          .then(pl.col("s1").sort(descending=True).slice(1, 1).first().over("s"))
          .otherwise(pl.col("s_max")).fill_null(0).alias("s_other"),
    )


def generate_candidates(df, split, model, q_rids, k=K, P=blocking.PARAMS):
    """Writes top-k candidates per S1 to cache/<TAG>/{split}_cands_k{k}/part-*.parquet."""
    d = os.path.join(BASE_VCACHE, f"{split}_cands_k{k}")
    if os.path.exists(os.path.join(d, "_DONE")):
        return d
    os.makedirs(d, exist_ok=True)
    t = time.time()
    sigs = blocking.get_signatures(df, split)
    idx = blocking.build_indexes(df, sigs)
    q_rids = np.asarray(q_rids, dtype=np.int32)
    n = 0
    for j, i in enumerate(range(0, len(q_rids), P["chunk"])):
        part = os.path.join(d, f"part-{j:04d}.parquet")
        if os.path.exists(part):  # resume after an interruption; redo unreadable parts
            try:
                n += pl.scan_parquet(part).select(pl.len()).collect().item()
                continue
            except Exception:
                os.remove(part)
        raw = stage1_features(blocking.raw_pairs(idx, sigs, q_rids[i:i + P["chunk"]], P), df)
        raw = raw.with_columns(pl.Series("s1", model.predict_proba(_X(raw, S1_FEATS))[:, 1]))
        raw = raw.filter(pl.col("s1").rank("ordinal", descending=True).over("q") <= k)
        raw.write_parquet(part + ".tmp")
        os.replace(part + ".tmp", part)  # atomic: a killed run never leaves a half-written part
        n += raw.height
        print(f"  cands {i+len(q_rids[i:i+P['chunk']]):,}/{len(q_rids):,}  [{time.time()-t:.0f}s]",
              flush=True)
    open(os.path.join(d, "_DONE"), "w").write(str(n))
    print(f"  {n:,} candidates ({n/len(q_rids):.1f}/S1) in {time.time()-t:.0f}s")
    return d


def load_candidates(d, q_filter=None):
    """Competition features are computed over ALL candidates of the split (only
    q, s, s1 in memory), then full rows are loaded for the requested S1s only."""
    files = os.path.join(d, "part-*.parquet")
    comp = competition_features(pl.scan_parquet(files).select("q", "s", "s1").collect())
    lf = pl.scan_parquet(files)
    if q_filter is not None:
        qs = pl.Series("q", np.asarray(q_filter, dtype=np.int32))
        lf = lf.filter(pl.col("q").is_in(qs))
        comp = comp.filter(pl.col("q").is_in(qs))
    return lf.collect().join(comp.drop("s1"), on=["q", "s"], how="left")

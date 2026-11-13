"""Test inference: writes output/candidate_pairs.tsv and output/matching_results.tsv.

Uses only the test source files plus the trained models; no labels are involved.
"""
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

from . import candidates, features
from .data import BASE_TAG, ROOT, TAG, VCACHE, prepare
from .train import MODELS, decide

OUT = os.path.join(ROOT, f"output_{TAG}")


def write_id_lists(pairs, df, col, path):
    """One row per S1 entity (empty list allowed), unique S2/S3 ids, tab-separated."""
    ids = df.select("rid", "entity_id")
    s1 = df.filter(pl.col("src") == 1).select(pl.col("rid").alias("q"), pl.col("entity_id").alias("source1_entity_id"))
    lists = (pairs.select("q", "s").unique()
                  .join(ids.rename({"rid": "s", "entity_id": "mid"}), on="s")
                  .sort("q", "mid").group_by("q").agg(pl.col("mid").str.join(",").alias(col)))
    out = s1.join(lists, on="q", how="left").with_columns(pl.col(col).fill_null("")) \
            .sort("q").select("source1_entity_id", col)
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"  wrote {path}: {out.height:,} rows, {(out[col] != '').sum():,} non-empty")


def main(q_chunk=300_000):
    t = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = prepare("test")
    m1 = xgb.XGBClassifier(); m1.load_model(os.path.join(ROOT, f"models_{BASE_TAG}", "stage1.json"))
    m2 = xgb.XGBClassifier(); m2.load_model(os.path.join(MODELS, "stage2.json"))
    dec = json.load(open(os.path.join(MODELS, "decision.json")))
    all_q = np.sort(df.filter(pl.col("src") == 1)["rid"].to_numpy())

    print("== candidate generation", flush=True)
    cdir = candidates.generate_candidates(df, "test", m1, all_q, k=dec["k"])
    files = os.path.join(cdir, "part-*.parquet")
    cand_qs = pl.scan_parquet(files).select("q", "s").collect()
    write_id_lists(cand_qs, df, "candidate_entity_ids", os.path.join(OUT, "candidate_pairs.tsv"))

    print("== stage-2 scoring", flush=True)
    if features.ACTIVE is not features.FEATS:
        df = features.add_rare_names(df)
    idf_n = features.token_idf(df, "name_n")
    idf_a = features.token_idf(df, "addr_n")
    comp = candidates.competition_features(pl.scan_parquet(files).select("q", "s", "s1").collect())
    ctry = df.select(pl.col("rid").alias("q"), "country")

    def cand_feats(qs):
        c = (pl.scan_parquet(files).filter(pl.col("q").is_in(qs)).collect()
               .join(comp.filter(pl.col("q").is_in(qs)).drop("s1"), on=["q", "s"], how="left"))
        return features.pair_features(c, df, idf_n, idf_a).join(ctry, on="q", how="left")

    if features.QN:
        # percentile tables from the TEST split's own (unlabeled) candidate pairs, per country
        qn_cols = json.load(open(os.path.join(MODELS, "qn_cols.json")))
        rng = np.random.default_rng(0)
        sample = cand_feats(pl.Series("q", np.sort(rng.choice(all_q, min(300_000, len(all_q)), replace=False)).astype(np.int32)))
        features.build_qn_tables(sample, sample["country"].unique().to_list(), cols=qn_cols)
        print(f"  test percentile tables from {sample.height:,} pairs; countries {sorted(sample['country'].unique().to_list())}", flush=True)
        del sample
    scored = []
    for i in range(0, len(all_q), q_chunk):
        qs = pl.Series("q", all_q[i:i + q_chunk].astype(np.int32))
        f = cand_feats(qs)
        if features.QN:
            f = features.add_qn(f)
        scored.append(f.select("q", "s", pl.Series("p", m2.predict_proba(features.matrix(f))[:, 1])))
        print(f"  scored S1 {min(i+q_chunk, len(all_q)):,}/{len(all_q):,} [{time.time()-t:.0f}s]", flush=True)
    scored = pl.concat(scored)
    scored.write_parquet(os.path.join(VCACHE, "test_scored.parquet"))

    pred = decide(scored, dec["thr"], dec["rel"])
    write_id_lists(pred, df, "matched_entity_ids", os.path.join(OUT, "matching_results.tsv"))
    print(f"done in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()

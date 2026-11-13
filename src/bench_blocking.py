"""Measure candidate recall / size of the blocking stage on a sample of train S1."""
import sys
import time

import numpy as np
import polars as pl
import psutil

from . import blocking
from .data import prepare, load_gt


def gt_pairs(df):
    gt = load_gt("train")
    ids = df.select("entity_id", "rid")
    return (gt.join(ids.rename({"entity_id": "s1_id", "rid": "q"}), on="s1_id")
              .join(ids.rename({"entity_id": "m_id", "rid": "s"}), on="m_id").select("q", "s"))


def main(n_sample=20000):
    P = dict(blocking.PARAMS)
    t = time.time()
    df = prepare("train")
    q_rids = np.sort(df.filter(pl.col("src") == 1).sample(n_sample, seed=42)["rid"].to_numpy())
    truth = gt_pairs(df).filter(pl.col("q").is_in(q_rids)).with_columns(pl.lit(True).alias("y"))
    raw, _ = blocking.generate(df, "train", q_rids, P, keep_raw=True)
    raw = raw.join(truth, on=["q", "s"], how="left").with_columns(pl.col("y").fill_null(False))
    n = truth.height
    print(f"true pairs: {n:,}  raw union recall: {raw['y'].sum()/n:.4f} ({raw.height/n_sample:.0f}/S1)")
    for bit, f in ((1, "name"), (2, "skel"), (4, "addr")):
        print(f"   only-{f} recall: {raw.filter((pl.col('via') & bit) > 0)['y'].sum()/n:.4f}")
    hm = raw.with_columns(pl.max_horizontal("hits_name", "hits_skel", "hits_addr").alias("hm"))
    for mh in (2, 3):
        k = hm.filter(pl.col("hm") >= mh)
        print(f"   min_hits={mh}: recall {k['y'].sum()/n:.4f}  raw/S1 {k.height/n_sample:.0f}")
    for kn, ka, ks in [(5, 5, 8), (8, 8, 12), (10, 10, 15), (15, 15, 25), (25, 25, 40)]:
        sel = blocking.select_topk(raw, {**P, "k_name": kn, "k_addr": ka, "k_sum": ks})
        print(f"  k=({kn},{ka},{ks}): recall {sel['y'].sum()/n:.4f}  cands/S1 {sel.height/n_sample:.1f}")
    truth.join(raw.select("q", "s"), on=["q", "s"], how="anti").write_parquet("cache/bench_missed.parquet")
    print(f"time {time.time()-t:.0f}s  rss {psutil.Process().memory_info().rss/1e9:.1f} GB")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))

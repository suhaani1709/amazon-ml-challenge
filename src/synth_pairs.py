"""Run a synthetic split (src/synth.py) through blocking + stage-1 + stage-2 features
and return labelled candidate pairs for stage-2 training."""
import os

import numpy as np
import polars as pl

from . import candidates, features
from .data import VCACHE, load_gt, prepare


def labelled_pairs(split, m1):
    path = os.path.join(VCACHE, f"{split}_feats.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    df = prepare(split)
    ids = df.select("entity_id", "rid")
    truth = (load_gt(split).join(ids.rename({"entity_id": "s1_id", "rid": "q"}), on="s1_id")
               .join(ids.rename({"entity_id": "m_id", "rid": "s"}), on="m_id").select("q", "s"))
    all_q = np.sort(df.filter(pl.col("src") == 1)["rid"].to_numpy())
    cdir = candidates.generate_candidates(df, split, m1, all_q)
    c = candidates.load_candidates(cdir)
    c = c.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q", "s"], how="left") \
         .with_columns(pl.col("y").fill_null(0))
    print(f"[{split}] candidate recall {c['y'].sum()/truth.height:.4f}  ({c.height/len(all_q):.1f} cands/S1)", flush=True)
    if features.ACTIVE is not features.FEATS:
        df = features.add_rare_names(df)
    f = features.pair_features(c, df, features.token_idf(df, "name_n"), features.token_idf(df, "addr_n"))
    f = f.join(df.select(pl.col("rid").alias("q"), "country"), on="q", how="left")
    f.write_parquet(path)
    return f

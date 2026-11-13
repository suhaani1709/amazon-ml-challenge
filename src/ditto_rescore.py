"""Stage-2b: re-score uncertain pairs with the team's fine-tuned Ditto cross-encoder.

Ditto (XLM-RoBERTa base, fine-tuned by a teammate; see ditto/ditto_er_model/README.md)
reads both raw records as text. It is ~100 pairs/s on an Apple M4, far too slow for all
42M test pairs, so it only re-scores pairs whose XGBoost probability p is in [LO, HI).
There the two opinions are combined by a logit average; everywhere else p is kept.

Validated on 1/3 of the validation S1s (settings chosen on one half, reported on the other):
  v5 0.9761 -> 0.9800 macro F0.5 (band [0.05, 0.95), w = 0.5, threshold 0.8);
  within the band Ditto AUC 0.947 vs XGBoost 0.922.

Progress is written in chunks (resumable). Output: output_<TAG>/matching_results.tsv, with the
same candidate_pairs.tsv as the base version (Ditto scores only pairs inside that set).
"""
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

from .data import ROOT, VCACHE, prepare
from .predict import write_id_lists
from .train import decide

LO, HI, W, THR = 0.05, 0.95, 0.5, 0.8      # v6
# v7: additionally re-score the upper edge [0.95, 0.98) and use threshold 0.75
# (validation, held-out half: 0.9800 -> 0.9804); earlier Ditto scores are reused.
BANDS_V7 = [(0.05, 0.95, "ditto_test_parts"), (0.95, 0.98, "ditto_test_parts_upper")]
THR_V7 = 0.75
CHUNK = 20_000
DITTO_DIR = os.path.join(ROOT, "ditto", "ditto_er_model")


def combine(p, pd, w=W):
    p, pd = np.clip(p, 1e-6, 1 - 1e-6), np.clip(pd, 1e-6, 1 - 1e-6)
    z = w * np.log(pd / (1 - pd)) + (1 - w) * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z))


def main(base_tag="v5", out_tag="v6"):
    t = time.time()
    sys.path.insert(0, DITTO_DIR)
    from ditto_infer import Ditto

    scored = pl.read_parquet(os.path.join(ROOT, "cache", base_tag, "test_scored.parquet"))
    band = scored.filter((pl.col("p") >= LO) & (pl.col("p") < HI))
    df = prepare("test").select("rid", "business_name", "business_address")
    x = (band.join(df.rename({"rid": "q", "business_name": "na", "business_address": "aa"}), on="q")
             .join(df.rename({"rid": "s", "business_name": "nb", "business_address": "ab"}), on="s")
             .with_columns(pl.col("aa").fill_null(""), pl.col("ab").fill_null(""), pl.col("nb").fill_null(""))
             .sort("q", "s"))
    print(f"test pairs {scored.height:,}; in band [{LO},{HI}): {x.height:,}", flush=True)
    parts = os.path.join(VCACHE, "ditto_test_parts")
    os.makedirs(parts, exist_ok=True)
    model = None
    for j, i in enumerate(range(0, x.height, CHUNK)):
        path = os.path.join(parts, f"part-{j:05d}.parquet")
        if os.path.exists(path):
            continue
        if model is None:
            model = Ditto()
        c = x.slice(i, CHUNK)
        pd = model.score(list(zip(c["na"], c["aa"])), list(zip(c["nb"], c["ab"])), batch=128)
        c.select("q", "s").with_columns(pl.Series("pd", pd, dtype=pl.Float32)).write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        done = min(i + CHUNK, x.height)
        print(f"  ditto {done:,}/{x.height:,}  [{time.time()-t:.0f}s]", flush=True)
    d = pl.read_parquet(os.path.join(parts, "part-*.parquet"))
    assert d.height == x.height, "missing Ditto chunks"
    s = scored.join(d, on=["q", "s"], how="left")
    newp = combine(s["p"].to_numpy(), s["pd"].fill_null(0.5).to_numpy())
    s = s.with_columns(pl.when(pl.col("pd").is_not_null()).then(pl.Series(newp)).otherwise(pl.col("p")).alias("p"))
    s.select("q", "s", "p").write_parquet(os.path.join(VCACHE, "test_scored_ditto.parquet"))
    out = os.path.join(ROOT, f"output_{out_tag}")
    os.makedirs(out, exist_ok=True)
    full = prepare("test")
    write_id_lists(decide(s, THR), full, "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
    shutil.copyfile(os.path.join(ROOT, f"output_{base_tag}", "candidate_pairs.tsv"),
                    os.path.join(out, "candidate_pairs.tsv"))
    print(f"done in {time.time()-t:.0f}s -> {out}", flush=True)


def score_band(scored, df, lo, hi, parts_name):
    """Ditto scores for all pairs with XGBoost p in [lo, hi), in resumable chunks."""
    sys.path.insert(0, DITTO_DIR)
    from ditto_infer import Ditto
    t = time.time()
    x = (scored.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
               .join(df.rename({"rid": "q", "business_name": "na", "business_address": "aa"}), on="q")
               .join(df.rename({"rid": "s", "business_name": "nb", "business_address": "ab"}), on="s")
               .with_columns(pl.col("aa").fill_null(""), pl.col("ab").fill_null(""), pl.col("nb").fill_null(""))
               .sort("q", "s"))
    parts = os.path.join(VCACHE, parts_name)
    os.makedirs(parts, exist_ok=True)
    print(f"band [{lo},{hi}): {x.height:,} pairs -> {parts}", flush=True)
    model = None
    for j, i in enumerate(range(0, x.height, CHUNK)):
        path = os.path.join(parts, f"part-{j:05d}.parquet")
        if os.path.exists(path):
            continue
        model = model or Ditto()
        c = x.slice(i, CHUNK)
        pd = model.score(list(zip(c["na"], c["aa"])), list(zip(c["nb"], c["ab"])), batch=128)
        c.select("q", "s").with_columns(pl.Series("pd", pd, dtype=pl.Float32)).write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"  ditto {min(i + CHUNK, x.height):,}/{x.height:,}  [{time.time()-t:.0f}s]", flush=True)
    d = pl.read_parquet(os.path.join(parts, "part-*.parquet"))
    assert d.height == x.height, "missing Ditto chunks"
    return d


def main_v7(base_tag="v5", out_tag="v7"):
    t = time.time()
    scored = pl.read_parquet(os.path.join(ROOT, "cache", base_tag, "test_scored.parquet"))
    df = prepare("test").select("rid", "business_name", "business_address")
    d = pl.concat([score_band(scored, df, lo, hi, name) for lo, hi, name in BANDS_V7]).unique(["q", "s"])
    s = scored.join(d, on=["q", "s"], how="left")
    newp = combine(s["p"].to_numpy(), s["pd"].fill_null(0.5).to_numpy())
    s = s.with_columns(pl.when(pl.col("pd").is_not_null()).then(pl.Series(newp)).otherwise(pl.col("p")).alias("p"))
    s.select("q", "s", "p").write_parquet(os.path.join(VCACHE, f"test_scored_{out_tag}.parquet"))
    out = os.path.join(ROOT, f"output_{out_tag}")
    os.makedirs(out, exist_ok=True)
    write_id_lists(decide(s, THR_V7), prepare("test"), "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
    shutil.copyfile(os.path.join(ROOT, f"output_{base_tag}", "candidate_pairs.tsv"),
                    os.path.join(out, "candidate_pairs.tsv"))
    print(f"re-scored pairs {d.height:,}; done in {time.time()-t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["v7"]:
        main_v7(*sys.argv[2:])
    else:
        main(*sys.argv[1:])

"""Stage-2c: add the team's fine-tuned Laya matcher on top of v6 (XGBoost + Ditto).

v6 score (logit average of XGBoost and Ditto, band [0.05, 0.95)) is rebuilt from the saved
parts; Laya then re-scores only the pairs Ditto was unsure about (Ditto p in [0.1, 0.9]),
which is what fits in the time budget on an Apple M4 (~60 pairs/s). The combination weight
and threshold are the ones chosen on the validation bench (bench/eval_model.py).

    python -m src.laya_rescore <weight> <threshold>
"""
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

from .data import ROOT, prepare
from .ditto_rescore import combine
from .predict import write_id_lists
from .train import decide

CHUNK = 20_000


def main(w, thr, base_tag="v5", out_tag="v8"):
    w, thr = float(w), float(thr)
    from . import laya_score as ls
    t = time.time()
    scored = pl.read_parquet(os.path.join(ROOT, "cache", base_tag, "test_scored.parquet"))
    d = pl.read_parquet(os.path.join(ROOT, "cache", base_tag, "ditto_test_parts", "part-*.parquet"))
    s = scored.join(d, on=["q", "s"], how="left")
    v6 = combine(s["p"].to_numpy(), s["pd"].fill_null(0.5).to_numpy())
    s = s.with_columns(pl.when(pl.col("pd").is_not_null()).then(pl.Series(v6)).otherwise(pl.col("p")).alias("p6"))
    unsure = s.filter((pl.col("pd") >= 0.1) & (pl.col("pd") <= 0.9)).select("q", "s").sort("q", "s")
    df = prepare("test")
    r = df.select("rid", "business_name", "business_address", "country")
    x = (unsure.join(r.rename({"rid": "q", "business_name": "na", "business_address": "aa", "country": "c"}), on="q")
               .join(r.drop("country").rename({"rid": "s", "business_name": "nb", "business_address": "ab"}), on="s")
               .sort("q", "s"))
    print(f"Laya on {x.height:,} Ditto-unsure test pairs", flush=True)
    parts = os.path.join(ROOT, "cache", base_tag, "laya_test_parts")
    os.makedirs(parts, exist_ok=True)
    agent = max_len = None
    for j, i in enumerate(range(0, x.height, CHUNK)):
        path = os.path.join(parts, f"part-{j:05d}.parquet")
        if os.path.exists(path):
            continue
        if agent is None:
            agent, max_len = ls.load()
        c = x.slice(i, CHUNK)
        p = ls.score(agent, max_len, [ls.rec(a, b, k) for a, b, k in zip(c["na"], c["aa"], c["c"])],
                     [ls.rec(a, b, k) for a, b, k in zip(c["nb"], c["ab"], c["c"])])
        c.select("q", "s").with_columns(pl.Series("pl", p)).write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"  laya {min(i + CHUNK, x.height):,}/{x.height:,}  [{time.time()-t:.0f}s]", flush=True)
    la = pl.read_parquet(os.path.join(parts, "part-*.parquet"))
    assert la.height == x.height, "missing Laya chunks"
    s = s.join(la, on=["q", "s"], how="left")
    z = combine(s["p6"].to_numpy(), s["pl"].fill_null(0.5).to_numpy(), w=w)
    s = s.with_columns(pl.when(pl.col("pl").is_not_null()).then(pl.Series(z)).otherwise(pl.col("p6")).alias("p"))
    out = os.path.join(ROOT, f"output_{out_tag}")
    os.makedirs(out, exist_ok=True)
    write_id_lists(decide(s.select("q", "s", "p"), thr), df, "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
    shutil.copyfile(os.path.join(ROOT, f"output_{base_tag}", "candidate_pairs.tsv"), os.path.join(out, "candidate_pairs.tsv"))
    print(f"done in {time.time()-t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

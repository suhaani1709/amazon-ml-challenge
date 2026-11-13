"""Loading, normalization and caching of the three sources.

Output per split (train/test): cache/{split}_records.parquet with one row per
record of S1, S2 and S3 and columns:
  rid (int32 row id), entity_id, src (1/2/3), country, business_name, business_address,
  name_n, addr_n (normalized), name_core, addr_core (stop tokens removed),
  is_domain, name_missing, addr_missing
"""
import os
import sys
import time

import polars as pl

from .normalize import normalize_many

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ER_DATA", os.path.join(ROOT, "data"))
CACHE = os.path.join(ROOT, "cache")
# Pipeline version: model-dependent artefacts (candidates, features, scores, models,
# outputs) are kept per version so a new experiment never overwrites a previous one.
TAG = os.environ.get("ER_TAG", "v2")
VCACHE = os.path.join(CACHE, TAG)
# A version may reuse the blocking stage (stage-1 model + candidates) of a base version.
BASE_TAG = os.environ.get("ER_BASE_TAG", TAG)
BASE_VCACHE = os.path.join(CACHE, BASE_TAG)
os.makedirs(VCACHE, exist_ok=True)

# A token is a "stop token" for blocking when it occurs in more than this share of
# the names (addresses) of its country. Computed on each split's own records, so it
# adapts to unseen countries (e.g. French legal forms SARL/SAS) without hard-coding.
NAME_STOP_DF = 0.015
ADDR_STOP_DF = 0.004


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def data_dir(split):
    """Synthetic splits (name starts with 'syn') live in data_syn/, see src/synth.py."""
    return os.path.join(ROOT, "data_syn") if split.startswith("syn") else DATA


def load_raw(split):
    frames = []
    for s in (1, 2, 3):
        d = read_tsv(os.path.join(data_dir(split), split, f"{split}_source{s}.tsv"))
        frames.append(d.with_columns(pl.lit(s, pl.Int8).alias("src")))
    return pl.concat(frames)


def load_gt(split="train"):
    gt = read_tsv(os.path.join(data_dir(split), split, f"{split}_ground_truth.tsv"))
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
              .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))


def _stop_tokens(df, col, thr):
    """Per-country set of tokens whose document frequency exceeds thr."""
    tok = (df.select("country", pl.col(col).str.split(" ").list.unique().alias("t"))
             .explode("t").filter(pl.col("t") != ""))
    cnt = tok.group_by("country", "t").len()
    tot = df.group_by("country").len().rename({"len": "n"})
    # tokens containing digits (house / plot / postal numbers) are never stop tokens
    return (cnt.join(tot, on="country").filter(pl.col("len") > thr * pl.col("n"))
               .filter(~pl.col("t").str.contains(r"\d")).select("country", "t"))


def _core(df, col, stop, out):
    """Remove stop tokens; fall back to the full string if nothing is left."""
    ex = (df.select("rid", "country", pl.col(col).str.split(" ").alias("t"))
            .with_columns(pl.int_ranges(pl.col("t").list.len()).alias("pos")).explode("t", "pos"))
    ex = ex.join(stop.with_columns(pl.lit(True).alias("stop")), on=["country", "t"], how="left")
    kept = (ex.filter(pl.col("stop").is_null() & (pl.col("t") != "")).sort("rid", "pos")
              .group_by("rid", maintain_order=True).agg(pl.col("t").str.join(" ").alias(out)))
    df = df.join(kept, on="rid", how="left")
    return df.with_columns(pl.coalesce(pl.col(out), pl.col(col)).alias(out))


def skeleton(col):
    """Consonant skeleton: robust to vowel noise from typos and transliteration
    (surya management / surya mainejament -> sry mngmnt / sry mngmnt)."""
    e = col.str.replace_all("ph", "f").str.replace_all("[jz]", "g").str.replace_all("[cq]", "k")
    e = e.str.replace_all("w", "v").str.replace_all("x", "ks").str.replace_all("[aeiouyh]", "")
    for ch in "bdfgklmnprstv":
        e = e.str.replace_all(ch + ch + "+", ch)
    return e.str.replace_all(" +", " ").str.strip_chars()


def prepare(split):
    path = os.path.join(CACHE, f"{split}_records.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    t = time.time()
    df = load_raw(split)
    df = df.with_columns(pl.col("country").fill_null("UNK"))
    df = df.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int32))
    print(f"[{split}] loaded {df.height:,} rows in {time.time()-t:.0f}s", flush=True)
    names, dom, addrs = normalize_many(df["business_name"].to_list(), df["business_address"].to_list())
    df = df.with_columns(pl.Series("name_n", names), pl.Series("is_domain", dom),
                         pl.Series("addr_n", addrs))
    print(f"[{split}] normalized in {time.time()-t:.0f}s", flush=True)
    ns = _stop_tokens(df, "name_n", NAME_STOP_DF)
    ads = _stop_tokens(df, "addr_n", ADDR_STOP_DF)
    print(f"[{split}] name stop tokens: {ns.sort('country')['t'].to_list()[:80]}")
    print(f"[{split}] addr stop tokens ({ads.height}): {ads['t'].to_list()[:60]}")
    df = _core(df, "name_n", ns, "name_core")
    df = _core(df, "addr_n", ads, "addr_core")
    df = df.with_columns(skeleton(pl.col("name_core")).alias("name_skel"))
    df = df.sort("rid").with_columns(
        (pl.col("name_n") == "").alias("name_missing"),
        (pl.col("addr_n") == "").alias("addr_missing"))
    df.write_parquet(path)
    ns.write_parquet(os.path.join(CACHE, f"{split}_name_stop.parquet"))
    ads.write_parquet(os.path.join(CACHE, f"{split}_addr_stop.parquet"))
    print(f"[{split}] prepared in {time.time()-t:.0f}s -> {path}", flush=True)
    return df


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        prepare(s)

"""Candidate generation: union of three MinHash-LSH passes, then top-K selection.

Passes (all over within-token character 3-grams):
  name  : name_core  (legal-form / very common tokens removed)
  skel  : name_skel  (consonant skeleton; robust to typos + Indic transliteration)
  addr  : addr_core  (common city/state/street-type tokens removed, numbers kept)
plus two exact-key passes:
  numname: (country, house/plot number, name-skeleton token). Generic names
           ("White Industries", "Womens Health Clinic") produce huge name buckets that
           the LSH passes must skip; pairing each name token with the address numbers
           makes the key specific again (recovers ~1/3 of pairs the LSH passes miss).
  tokkey : (country, one of the 2 rarest name-skeleton tokens, one of the 4 rarest
           address tokens incl. city/locality words). Catches generic names at addresses
           without house numbers ("New Properties" + "surat"); validation candidate
           recall 96.49% -> 97.47%.

Country is a hard block (it agrees on 100% of training matches): the country id is
mixed into every band key, so an unseen country (e.g. France) is just another id.
"""
import os
import time

import numpy as np
import polars as pl

from . import lsh
from .data import BASE_TAG, CACHE

TOKKEY = BASE_TAG not in ("v1", "v2", "v3", "v4")  # tokkey pass from v5 on

FIELDS = {  # field: (column, num_perm, rows per band, seed, via-bit)
    "name": ("name_core", 48, 2, 11, 1),
    "skel": ("name_skel", 32, 2, 17, 2),
    "addr": ("addr_core", 36, 3, 23, 4),
}
PARAMS = dict(cap=400, min_hits=2, k_name=10, k_addr=10, k_sum=15, chunk=100_000, nn_cap=100,
              tk_cap=100, tk_addr=4, tk_name=2)


def get_signatures(df, split, fields=FIELDS):
    out = {}
    for field, (col, perm, _, seed, _) in fields.items():
        path = os.path.join(CACHE, f"{split}_sig_{field}_{perm}_{seed}.npy")
        if os.path.exists(path):
            out[field] = np.load(path)
        else:
            t = time.time()
            out[field] = lsh.signatures(df[col].to_list(), perm, seed)
            np.save(path, out[field])
            print(f"  {field} signatures {out[field].shape} in {time.time()-t:.0f}s", flush=True)
    return out


class Index:
    """Per-band sorted S2/S3 keys, stored contiguously (bands x n) for fast lookup."""

    def __init__(self, sig, r, country, s_rids):
        self.r, self.country = r, country
        sk = lsh.band_keys(sig[s_rids], r, country[s_rids])
        order = np.argsort(sk, axis=0, kind="stable")
        self.sorted = np.ascontiguousarray(np.take_along_axis(sk, order, axis=0).T)
        self.sid = np.ascontiguousarray(s_rids[order].T)
        del sk, order

    def query(self, sig, q_rids, cap):
        """Unique colliding pairs as int64 codes (q<<32|s) with their band-hit counts."""
        qk = lsh.band_keys(sig[q_rids], self.r, self.country[q_rids])
        qs, ss = [], []
        for b in range(qk.shape[1]):
            q, s = lsh._collide(np.ascontiguousarray(qk[:, b]), q_rids, self.sorted[b], self.sid[b], cap)
            qs.append(q), ss.append(s)
        code = (np.concatenate(qs).astype(np.int64) << 32) | np.concatenate(ss).astype(np.int64)
        return np.unique(code, return_counts=True)


def numname_keys(df):
    """(rid, key) for every name-skeleton token x address number (first 4 numbers)."""
    return (df.select("rid", "country", pl.col("name_skel").str.split(" ").list.unique().alias("tok"),
                      pl.col("addr_n").str.extract_all(r"\d+")
                        .list.eval(pl.element().str.strip_chars_start("0")).list.unique().list.head(4).alias("num"))
              .explode("tok").explode("num")
              .filter((pl.col("tok").str.len_chars() >= 2) & pl.col("num").is_not_null() & (pl.col("num") != ""))
              .select("rid", pl.concat_str("country", "num", "tok", separator="|").hash().alias("k"))
              .unique())


class NumNameIndex:
    """Exact compound-key index: S2/S3 keys whose bucket has <= cap records."""

    def __init__(self, df, cap):
        keys = numname_keys(df)
        src = df.select("rid", "src")
        s = keys.join(src.filter(pl.col("src") != 1).select("rid"), on="rid")
        big = s.group_by("k").len().filter(pl.col("len") > cap).select("k")
        self.s = s.join(big, on="k", how="anti").rename({"rid": "s"})
        self.q = keys.join(src.filter(pl.col("src") == 1).select("rid"), on="rid").rename({"rid": "q"})

    def query(self, q_rids):
        qs = self.q.filter(pl.col("q").is_in(pl.Series(q_rids)))
        p = qs.join(self.s, on="k").group_by("q", "s").len()
        code = (p["q"].to_numpy().astype(np.int64) << 32) | p["s"].to_numpy().astype(np.int64)
        return code, p["len"].to_numpy()


class TokKeyIndex:
    """Exact keys (country, rare name token, rare address token); rarity = per-country
    document frequency computed on the split's own records (no labels)."""

    def __init__(self, df, cap, n_addr, n_name):
        def rarest(col, name, n):
            t = (df.select("rid", "country", pl.col(col).str.split(" ").list.unique().alias(name))
                   .explode(name).filter(pl.col(name).is_not_null() & (pl.col(name).str.len_chars() >= 2)))
            f = t.group_by("country", name).len().rename({"len": "df"})
            return t.join(f, on=["country", name]).sort("rid", "df").group_by("rid").head(n)
        a = rarest("addr_n", "a", n_addr)
        nm = rarest("name_skel", "n", n_name)
        keys = (nm.join(a.select("rid", "a"), on="rid")
                  .select("rid", pl.concat_str("country", "n", "a", separator="|").hash().alias("k")).unique())
        del a, nm
        src = df.select("rid", "src")
        s = keys.join(src.filter(pl.col("src") != 1).select("rid"), on="rid")
        big = s.group_by("k").len().filter(pl.col("len") > cap).select("k")
        self.s = s.join(big, on="k", how="anti").rename({"rid": "s"})
        self.q = keys.join(src.filter(pl.col("src") == 1).select("rid"), on="rid").rename({"rid": "q"})

    query = NumNameIndex.query


def build_indexes(df, sigs, fields=FIELDS):
    country = df["country"].cast(pl.Categorical).to_physical().to_numpy().astype(np.uint32)
    s_rids = np.flatnonzero(df["src"].to_numpy() != 1).astype(np.int32)
    idx = {}
    for f, (_, _, r, _, _) in fields.items():
        t = time.time()
        idx[f] = Index(sigs[f], r, country, s_rids)
        print(f"  index {f}: {time.time()-t:.0f}s", flush=True)
    t = time.time()
    idx["numname"] = NumNameIndex(df, PARAMS["nn_cap"])
    print(f"  index numname: {time.time()-t:.0f}s ({idx['numname'].s.height:,} S2/S3 keys)", flush=True)
    if TOKKEY:
        t = time.time()
        idx["tokkey"] = TokKeyIndex(df, PARAMS["tk_cap"], PARAMS["tk_addr"], PARAMS["tk_name"])
        print(f"  index tokkey: {time.time()-t:.0f}s ({idx['tokkey'].s.height:,} S2/S3 keys)", flush=True)
    return idx


def raw_pairs(idx, sigs, q_rids, P, fields=FIELDS):
    """Union of all LSH collisions for a chunk of S1 rows, scored by MinHash sims.

    Pairs that collide in only `min_hits-1` bands of every LSH pass are dropped before
    scoring (band-hit count is itself a coarse similarity estimate); pairs sharing an
    exact numname key are always kept.
    """
    frames = []
    for f, (_, _, _, _, bit) in fields.items():
        code, cnt = idx[f].query(sigs[f], q_rids, P["cap"])
        frames.append(pl.DataFrame({"code": code, "via": np.full(len(code), bit, np.int8),
                                    f"hits_{f}": cnt.astype(np.int16)}))
    if "numname" in idx:
        code, cnt = idx["numname"].query(q_rids)
        frames.append(pl.DataFrame({"code": code, "via": np.full(len(code), 8, np.int8),
                                    "hits_nn": cnt.astype(np.int16)}))
    if "tokkey" in idx:
        code, cnt = idx["tokkey"].query(q_rids)
        frames.append(pl.DataFrame({"code": code, "via": np.full(len(code), 16, np.int8),
                                    "hits_tk": cnt.astype(np.int16)}))
    pairs = pl.concat(frames, how="diagonal").group_by("code").agg(
        pl.col("via").sum(), *[pl.col(f"hits_{f}").max().fill_null(0) for f in fields],
        pl.col("hits_nn").max().fill_null(0) if "numname" in idx else pl.lit(0, pl.Int16).alias("hits_nn"),
        pl.col("hits_tk").max().fill_null(0) if "tokkey" in idx else pl.lit(0, pl.Int16).alias("hits_tk"))
    hmax = pl.max_horizontal(*[pl.col(f"hits_{f}") for f in fields])
    pairs = pairs.filter((hmax >= P["min_hits"]) | (pl.col("hits_nn") >= 1) | (pl.col("hits_tk") >= 1)).sort("code")
    code = pairs["code"].to_numpy()
    q = (code >> 32).astype(np.int32)
    s = (code & 0xFFFFFFFF).astype(np.int32)
    return pairs.drop("code").with_columns(
        pl.Series("q", q), pl.Series("s", s),
        *[pl.Series(f"mh_{f}", lsh.sig_sim(sigs[f], q, s)) for f in fields])


def select_topk(pairs, P):
    """Keep a pair if it is top-k by name sim, by address sim, or by their sum."""
    p = pairs.with_columns(
        pl.max_horizontal("mh_name", "mh_skel").clip(0).alias("mh_nm"),
    ).with_columns((pl.col("mh_nm") + pl.col("mh_addr").clip(0)).alias("mh_sum"))
    p = p.with_columns(
        pl.col("mh_nm").rank("ordinal", descending=True).over("q").alias("rk_name"),
        pl.col("mh_addr").rank("ordinal", descending=True).over("q").alias("rk_addr"),
        pl.col("mh_sum").rank("ordinal", descending=True).over("q").alias("rk_sum"))
    return p.filter((pl.col("rk_name") <= P["k_name"]) | (pl.col("rk_addr") <= P["k_addr"])
                    | (pl.col("rk_sum") <= P["k_sum"]))


def generate(df, split, q_rids, P=PARAMS, fields=FIELDS, keep_raw=False):
    """Full candidate generation for the given S1 rows, processed in chunks."""
    t = time.time()
    sigs = get_signatures(df, split, fields)
    country = df["country"].cast(pl.Categorical).to_physical().to_numpy().astype(np.uint32)
    idx = build_indexes(df, sigs, fields)
    out = []
    q_rids = np.asarray(q_rids, dtype=np.int32)
    for i in range(0, len(q_rids), P["chunk"]):
        raw = raw_pairs(idx, sigs, q_rids[i:i + P["chunk"]], P, fields)
        out.append(raw if keep_raw else select_topk(raw, P))
        print(f"  chunk {i//P['chunk']}: {raw.height/len(q_rids[i:i+P['chunk']]):.0f} raw/S1 -> "
              f"{out[-1].height:,} kept  [{time.time()-t:.0f}s]", flush=True)
    return pl.concat(out), sigs

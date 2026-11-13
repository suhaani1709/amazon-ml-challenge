"""Stage B pairwise features for (S1, S2/S3) candidate pairs.

String similarities use rapidfuzz's multi-threaded element-wise `cpdist`; token
IDF weights are computed per split from its own records (so unseen countries get
their own statistics).
"""
import math
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .candidates import S1_FEATS
from .data import TAG

STR_COLS = ["name_n", "name_core", "name_skel", "addr_n", "addr_core", "is_domain",
            "name_missing", "addr_missing", "src"]

FEATS = S1_FEATS + [
    "s1", "s_nq", "s_rank", "s_max", "s_gap", "s_other", "q_rank", "q_max", "q_sum", "q_gap",
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "core_ratio", "core_ns_ratio",
    "core_tset", "skel_ratio", "nm_eq", "core_eq", "nm_len_ratio", "q_dom", "s_dom",
    "nm_idf", "nm_idf_shared_max", "nm_idf_q_only", "nm_idf_s_only",
    "ad_ratio", "ad_tsort", "ad_tset", "ad_partial", "adc_tset", "adc_ratio",
    "ad_idf", "ad_idf_shared_max", "ad_idf_q_only", "ad_idf_s_only",
    "num_jacc", "num_shared", "num_q", "num_s", "num_first_eq", "num_q_in_s",
    "ad_len_ratio", "s_name_missing",
]


# v3: features aimed at noise seen in the unseen test country (France), but defined
# generically so they are validated on US/India:
#   acr           - one name is the acronym of the other ("club de secours" / "cs")
#   nm_rare_*     - name similarity after removing tokens that are common *within the
#                   record's country* (generic category words: club, section, services...)
V3_FEATS = ["acr", "nm_rare_tset", "nm_rare_ratio", "nm_rare_eq"]
ACTIVE = FEATS + V3_FEATS if TAG not in ("v1", "v2") else FEATS
RARE_DF = 0.004

# v4: per-country percentile ("quantile-normalized") copies of the continuous features.
# A raw similarity means different things in countries with different formats; its
# percentile among that country's *own* candidate pairs does not. Tables are built from
# the unlabeled candidate pairs of the split being scored (train tables for training,
# test tables for test), so an unseen country (France) gets its own calibration.
# Validated on a leave-one-country-out bench: US->India 0.8905->0.8910,
# India->US 0.9435->0.9532, seen-country scores unchanged.
QN = TAG not in ("v1", "v2", "v3")
BASE_ACTIVE = list(ACTIVE)
QN_GRID = 1001
QN_TABLES = {}          # (country, feature) -> sorted quantile grid
QN_COLS = []            # continuous features that get a percentile copy


def build_qn_tables(frame, countries, cols=None, max_rows=2_000_000, seed=0):
    """frame: feature rows with a `country` column. Sets QN_TABLES / QN_COLS / ACTIVE.
    cols: the percentile feature list (fixed at training time); None = derive it."""
    global QN_COLS, ACTIVE
    QN_COLS = list(cols) if cols is not None else \
        [f for f in BASE_ACTIVE if frame.select(pl.col(f).n_unique()).item() > 10]
    QN_TABLES.clear()
    for c in countries:
        sub = frame.filter(pl.col("country") == c)
        if sub.height > max_rows:
            sub = sub.sample(max_rows, seed=seed)
        for f in QN_COLS:
            v = sub[f].drop_nulls().cast(pl.Float64).to_numpy()
            QN_TABLES[(c, f)] = np.quantile(v, np.linspace(0, 1, QN_GRID)) if len(v) else np.zeros(QN_GRID)
    ACTIVE = BASE_ACTIVE + [f"qn_{f}" for f in QN_COLS]
    return pl.DataFrame([{"country": c, "feature": f, "grid": g.tolist()} for (c, f), g in QN_TABLES.items()])


def add_qn(x):
    """Adds qn_<feature> = mid-rank percentile of the value within its country's table."""
    ctry = x["country"].to_numpy()
    cols = {}
    for f in QN_COLS:
        v = x[f].cast(pl.Float64).to_numpy()
        out = np.full(len(v), np.nan)
        for c in np.unique(ctry):
            g = QN_TABLES[(c, f)]
            m = (ctry == c) & ~np.isnan(v)
            out[m] = (np.searchsorted(g, v[m], "left") + np.searchsorted(g, v[m], "right")) / (2.0 * len(g))
        cols[f"qn_{f}"] = pl.Series(f"qn_{f}", out, dtype=pl.Float32).fill_nan(None)
    return x.with_columns(**cols)


def add_rare_names(df):
    """Adds `name_rare`: name_n without tokens whose per-country document frequency
    exceeds RARE_DF (computed on the split's own records, no labels)."""
    ex = (df.select("rid", "country", pl.col("name_n").str.split(" ").alias("t"))
            .with_columns(pl.int_ranges(pl.col("t").list.len()).alias("pos")).explode("t", "pos")
            .filter(pl.col("t") != ""))
    n = df.group_by("country").len().rename({"len": "n"})
    common = (ex.unique(["rid", "t"]).group_by("country", "t").len().join(n, on="country")
                .filter(pl.col("len") > RARE_DF * pl.col("n")).select("country", "t", pl.lit(True).alias("c")))
    kept = (ex.join(common, on=["country", "t"], how="left").filter(pl.col("c").is_null())
              .sort("rid", "pos").group_by("rid", maintain_order=True).agg(pl.col("t").str.join(" ").alias("name_rare")))
    return df.join(kept, on="rid", how="left").with_columns(pl.col("name_rare").fill_null("")).sort("rid")


def _initials(col, min_len):
    return (col.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= min_len)
                                           .str.slice(0, 1)).list.join(""))


def token_idf(df, col):
    n = df.height
    return (df.select(pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
              .filter(pl.col("t") != "").group_by("t").len()
              .select("t", (pl.lit(n) / pl.col("len")).log().cast(pl.Float32).alias("idf")))


def _idf_overlap(pid, qcol, scol, idf, prefix):
    """IDF-weighted token Jaccard + mass of shared / one-sided tokens."""
    def side(texts):
        return (pl.DataFrame({"pid": pid, "t": texts}).with_columns(pl.col("t").str.split(" ").list.unique())
                  .explode("t").filter(pl.col("t") != "").join(idf, on="t", how="left")
                  .with_columns(pl.col("idf").fill_null(math.log(1e7))))
    a, b = side(qcol), side(scol)
    sh = a.join(b.select("pid", "t"), on=["pid", "t"]).group_by("pid").agg(
        pl.col("idf").sum().alias("sh"), pl.col("idf").max().alias("shmax"))
    out = (pl.DataFrame({"pid": pid})
             .join(a.group_by("pid").agg(pl.col("idf").sum().alias("qa")), on="pid", how="left")
             .join(b.group_by("pid").agg(pl.col("idf").sum().alias("sb")), on="pid", how="left")
             .join(sh, on="pid", how="left").fill_null(0).sort("pid"))
    uni = out["qa"] + out["sb"] - out["sh"]
    return {f"{prefix}_idf": (out["sh"] / uni.clip(1e-6)).to_numpy(),
            f"{prefix}_idf_shared_max": out["shmax"].to_numpy(),
            f"{prefix}_idf_q_only": (out["qa"] - out["sh"]).to_numpy(),
            f"{prefix}_idf_s_only": (out["sb"] - out["sh"]).to_numpy()}


def _sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _numbers(col):
    return (col.str.extract_all(r"\d+")
               .list.eval(pl.element().str.strip_chars_start("0").replace("", "0")).list.unique())


def pair_features(c, df, idf_name, idf_addr, chunk=2_000_000):
    """c: candidate frame with q, s + stage-A features. Returns c with FEATS added."""
    t = time.time()
    rec = df.select("rid", *STR_COLS, *(["name_rare"] if "name_rare" in df.columns else []))
    outs = []
    for i in range(0, c.height, chunk):
        x = c.slice(i, chunk).with_row_index("pid")
        x = (x.join(rec.rename({k: f"q_{k}" for k in rec.columns}), left_on="q", right_on="q_rid")
              .join(rec.rename({k: f"s_{k}" for k in rec.columns}), left_on="s", right_on="s_rid")
              .sort("pid"))
        f = {}
        qn, sn = x["q_name_n"].to_list(), x["s_name_n"].to_list()
        f["nm_ratio"] = _sim(qn, sn, fuzz.ratio)
        f["nm_tsort"] = _sim(qn, sn, fuzz.token_sort_ratio)
        f["nm_tset"] = _sim(qn, sn, fuzz.token_set_ratio)
        f["nm_partial"] = _sim(qn, sn, fuzz.partial_ratio)
        f["nm_jw"] = _sim(qn, sn, JaroWinkler.normalized_similarity)
        qc, sc = x["q_name_core"].to_list(), x["s_name_core"].to_list()
        f["core_ratio"] = _sim(qc, sc, fuzz.ratio)
        f["core_tset"] = _sim(qc, sc, fuzz.token_set_ratio)
        f["core_ns_ratio"] = _sim([s.replace(" ", "") for s in qc], [s.replace(" ", "") for s in sc],
                                  fuzz.ratio)
        f["skel_ratio"] = _sim(x["q_name_skel"].to_list(), x["s_name_skel"].to_list(), fuzz.ratio)
        qa, sa = x["q_addr_n"].to_list(), x["s_addr_n"].to_list()
        f["ad_ratio"] = _sim(qa, sa, fuzz.ratio)
        f["ad_tsort"] = _sim(qa, sa, fuzz.token_sort_ratio)
        f["ad_tset"] = _sim(qa, sa, fuzz.token_set_ratio)
        f["ad_partial"] = _sim(qa, sa, fuzz.partial_ratio)
        qac, sac = x["q_addr_core"].to_list(), x["s_addr_core"].to_list()
        f["adc_tset"] = _sim(qac, sac, fuzz.token_set_ratio)
        f["adc_ratio"] = _sim(qac, sac, fuzz.ratio)
        pid = x["pid"].to_numpy()
        f.update(_idf_overlap(pid, qn, sn, idf_name, "nm"))
        f.update(_idf_overlap(pid, qa, sa, idf_addr, "ad"))
        x = x.with_columns(**{k: pl.Series(v) for k, v in f.items()})
        # missing S2/S3 address -> address similarities are undefined, not "different"
        miss = pl.col("s_addr_missing") | pl.col("q_addr_missing")
        x = x.with_columns([pl.when(miss).then(None).otherwise(pl.col(k)).alias(k)
                            for k in f if k.startswith("ad")])
        x = x.with_columns(_numbers(pl.col("q_addr_n")).alias("nq"), _numbers(pl.col("s_addr_n")).alias("ns"))
        x = x.with_columns(
            pl.col("nq").list.set_intersection("ns").list.len().alias("num_shared"),
            pl.col("nq").list.len().alias("num_q"), pl.col("ns").list.len().alias("num_s"),
            (pl.col("nq").list.first() == pl.col("ns").list.first()).cast(pl.Int8).alias("num_first_eq"),
        ).with_columns(
            (pl.col("num_shared") / (pl.col("num_q") + pl.col("num_s") - pl.col("num_shared")).clip(1))
              .alias("num_jacc"),
            (pl.col("num_shared") / pl.col("num_q").clip(1)).alias("num_q_in_s"),
            (pl.col("q_name_n") == pl.col("s_name_n")).cast(pl.Int8).alias("nm_eq"),
            (pl.col("q_name_core") == pl.col("s_name_core")).cast(pl.Int8).alias("core_eq"),
            (pl.min_horizontal(pl.col("q_name_n").str.len_chars(), pl.col("s_name_n").str.len_chars())
             / pl.max_horizontal(pl.col("q_name_n").str.len_chars(), pl.col("s_name_n").str.len_chars()).clip(1))
              .alias("nm_len_ratio"),
            (pl.min_horizontal(pl.col("q_addr_n").str.len_chars(), pl.col("s_addr_n").str.len_chars())
             / pl.max_horizontal(pl.col("q_addr_n").str.len_chars(), pl.col("s_addr_n").str.len_chars()).clip(1))
              .alias("ad_len_ratio"),
            pl.col("q_is_domain").cast(pl.Int8).alias("q_dom"),
            pl.col("s_is_domain").cast(pl.Int8).alias("s_dom"),
            pl.col("s_name_missing").cast(pl.Int8),
        )
        if "q_name_rare" in x.columns:
            qr, sr = x["q_name_rare"].to_list(), x["s_name_rare"].to_list()
            x = x.with_columns(
                pl.Series("nm_rare_tset", _sim(qr, sr, fuzz.token_set_ratio)),
                pl.Series("nm_rare_ratio", _sim(qr, sr, fuzz.ratio)),
                ((pl.col("q_name_rare") == pl.col("s_name_rare")) & (pl.col("q_name_rare") != ""))
                  .cast(pl.Int8).alias("nm_rare_eq"),
            ).with_columns([pl.when((pl.col("q_name_rare") == "") | (pl.col("s_name_rare") == ""))
                              .then(None).otherwise(pl.col(k)).alias(k) for k in ("nm_rare_tset", "nm_rare_ratio")])
            qns = pl.col("q_name_n").str.replace_all(" ", "")
            sns = pl.col("s_name_n").str.replace_all(" ", "")
            x = x.with_columns(
                (((sns.str.len_chars() >= 2) & ((sns == _initials(pl.col("q_name_n"), 1)) | (sns == _initials(pl.col("q_name_n"), 3))))
                 | ((qns.str.len_chars() >= 2) & ((qns == _initials(pl.col("s_name_n"), 1)) | (qns == _initials(pl.col("s_name_n"), 3)))))
                .cast(pl.Int8).alias("acr"))
        outs.append(x.select(c.columns + [k for k in BASE_ACTIVE if k not in c.columns]))  # qn_* added later by add_qn
        print(f"  features {min(i+chunk, c.height):,}/{c.height:,} [{time.time()-t:.0f}s]", flush=True)
    return pl.concat(outs)


def matrix(x):
    return x.select([pl.col(k).cast(pl.Float32) for k in ACTIVE]).to_numpy()

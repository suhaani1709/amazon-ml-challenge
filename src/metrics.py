"""Official metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import polars as pl


def macro_f05(pred, truth, s1_ids, beta=0.5):
    """pred / truth: DataFrames (s1_id, m_id) of predicted / true pairs.
    s1_ids: every evaluated S1 id (so empty predictions and singletons count).

    Per S1: both empty -> 1.0; exactly one empty -> 0.0; else F-beta of the sets.
    Returns dict with macro F0.5 and pair-level precision / recall / counts.
    """
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids})
    tp = pred.join(truth, on=["s1_id", "m_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pred.group_by("s1_id").len().rename({"len": "np"})
    ntrue = truth.group_by("s1_id").len().rename({"len": "nt"})
    d = (base.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
             .join(ntrue, on="s1_id", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (1 + b2) * p * r / (b2 * p + r)
    d = d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
          .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise(f).alias("f"))
    TP, NP, NT = d["tp"].sum(), d["np"].sum(), d["nt"].sum()
    single = d.filter(pl.col("nt") == 0)
    return {"macro_f05": d["f"].mean(), "pair_precision": TP / max(NP, 1),
            "pair_recall": TP / max(NT, 1), "tp": TP, "fp": NP - TP, "fn": NT - TP,
            "n_s1": d.height, "singleton_acc": single["f"].mean() if single.height else None}

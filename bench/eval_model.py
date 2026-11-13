"""Evaluate a new model on the uncertain-pair validation bench.

Input: a parquet/TSV with column `p_new` aligned row-by-row with
bench/val_uncertain_pairs.parquet (60,949 pairs of 1/3 of the validation S1s whose
XGBoost score is in [0.02, 0.98)). Reports macro F0.5 on the held-out half B (settings
chosen on half A) for: v5 (XGBoost), v6 (XGBoost + Ditto, submitted: LB 0.971),
new model alone in the band, and v6 + new model.

    python bench/eval_model.py scores.parquet
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, ".")
from src.train import decide, evaluate, fold_of, truth_pairs  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(path):
    b = pl.read_parquet("bench/val_uncertain_pairs.parquet")
    new = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path, separator="\t", quote_char=None)
    assert new.height == b.height, f"expected {b.height} scores, got {new.height}"
    b = b.with_columns(new["p_new"].cast(pl.Float64))
    rec = pl.read_parquet("cache/train_records.parquet", columns=["rid", "entity_id", "src"])
    T = truth_pairs(rec)
    vq = rec.filter((pl.col("src") == 1) & (fold_of(pl.col("rid")) == 1) & (pl.col("rid") % 3 == 0))["rid"]
    H = {"A": vq.filter((vq // 6) % 2 == 0).to_numpy(), "B": vq.filter((vq // 6) % 2 == 1).to_numpy()}
    full = pl.read_parquet("cache/v5/val_scored.parquet").filter(pl.col("q").is_in(vq)).select("q", "s", "p")
    band = (b["p_xgb"] >= 0.05) & (b["p_xgb"] < 0.95)
    px, pd, pn = b["p_xgb"].to_numpy(), b["p_ditto"].to_numpy(), b["p_new"].to_numpy()

    def with_band(vals, mask):
        upd = b.select("q", "s").with_columns(pl.Series("pb", vals)).filter(pl.Series(mask))
        return full.join(upd, on=["q", "s"], how="left").with_columns(
            pl.coalesce("pb", "p").alias("p")).drop("pb")

    def f(d, thr, h):
        r = evaluate(decide(d, thr), T.filter(pl.col("q").is_in(H[h])), H[h])
        return r["macro_f05"], r["pair_precision"], r["pair_recall"]

    from sklearn.metrics import roc_auc_score
    m = band.to_numpy()
    print(f"AUC in band [0.05,0.95): XGBoost {roc_auc_score(b['y'].to_numpy()[m], px[m]):.4f} | "
          f"Ditto {roc_auc_score(b['y'].to_numpy()[m], pd[m]):.4f} | NEW {roc_auc_score(b['y'].to_numpy()[m], pn[m]):.4f}")
    v6 = 1 / (1 + np.exp(-(0.5 * logit(pd) + 0.5 * logit(px))))
    cands = {"v5 (XGBoost only)": [(full, t) for t in (0.7,)],
             "v6 (XGB+Ditto) [submitted]": [(with_band(v6, m), t) for t in (0.8,)]}
    unsure = m & (pd >= 0.1) & (pd <= 0.9)   # what can be scored on test tonight
    for name, weights, where in (("NEW alone", None, m), ("v6 + NEW", (0.25, 0.4, 0.5, 0.6), m),
                                 ("v6 + NEW, Ditto-unsure only", (0.25, 0.4, 0.5, 0.6, 1.0), unsure)):
        opts = []
        for w in (weights or (1.0,)):
            z = w * logit(pn) + (1 - w) * logit(v6) if weights else logit(pn)
            pv = np.where(where, 1 / (1 + np.exp(-z)), v6)
            for t in (0.6, 0.7, 0.75, 0.8, 0.85):
                opts.append((with_band(pv, m), t, w))
        cands[name] = opts
    print(f"{'':30} {'half A (tuning)':>16} {'half B (held out)':>18}  P / R on B")
    for name, opts in cands.items():
        best = max(opts, key=lambda o: f(o[0], o[1], "A")[0])
        a, b_ = f(best[0], best[1], "A"), f(best[0], best[1], "B")
        extra = f"  thr {best[1]}" + (f" w {best[2]}" if len(best) > 2 else "")
        print(f"{name:30} {a[0]:16.4f} {b_[0]:18.4f}  {b_[1]:.4f} / {b_[2]:.4f}{extra}")


if __name__ == "__main__":
    main(sys.argv[1])

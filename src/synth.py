"""Synthetic labelled data for a country without labels (France in test).

The test set contains a country (France) that never appears in training. We create
a labelled training split for such a country from its own **S1 text only**:

  * S1 records of the country are split in two halves. Half A keeps its S1 record;
    half B does not (its copies become realistic "orphan" decoys, as seen in test).
  * Every S1 gets 0..6 noisy S2/S3 copies, with the number of copies and the noise
    rates taken from the training ground truth (measured per source: e.g. S2 addresses
    are upper-cased 62% of the time, 4.5% of copies have a blank address).
  * The noise is *generic* (case, accents, typos, leetspeak, token reorder / drop,
    brackets, tags, websites, acronyms, legal-form swaps, address part drop / reorder /
    abbreviation, house-number perturbation, junk tokens). Country-specific vocabulary
    (legal forms, category words, street words) is taken from the target country's own
    frequent tokens, so no hand-written knowledge about the country is used.

No real S2/S3 records and no labels of the target country are used. The generated
TSVs follow the challenge file format, so the normal pipeline can process them.
"""
import os
import random
import re
import sys
from collections import Counter

import polars as pl

from .data import ROOT, read_tsv

SYN_ROOT = os.path.join(ROOT, "data_syn")

# copies per S1 (training ground truth), per source
N2_DIST = [0.1304, 0.3576, 0.2958, 0.1513, 0.0540, 0.0109]
N3_DIST = [0.1207, 0.3246, 0.3029, 0.1688, 0.0658, 0.0160, 0.0012]
P = {  # per-copy noise probabilities, (S2, S3), measured on 40k training pairs
    "name_upper": (0.205, 0.029), "name_lower": (0.066, 0.073), "accent": (0.065, 0.069),
    "domain": (0.050, 0.051), "tag": (0.008, 0.007), "phone": (0.004, 0.003),
    "leet": (0.020, 0.017), "brackets": (0.063, 0.065), "reorder": (0.089, 0.098),
    "double_space": (0.115, 0.111), "addr_missing": (0.045, 0.043), "addr_upper": (0.625, 0.0),
    "hash": (0.026, 0.024), "junk": (0.018, 0.016), "addr_drop": (0.289, 0.153),
}
# not directly measurable with a simple rule; set to plausible values
P_TYPO, P_LEGAL, P_CATEGORY, P_TOKDROP, P_ACRONYM = 0.30, 0.30, 0.08, 0.08, 0.005
P_ADDR_REORDER, P_ADDR_ABBR, P_ADDR_TYPO, P_NUM = 0.15, 0.20, 0.20, 0.08
ACCENTS = {"a": "áàâä", "e": "éèêë", "i": "íìîï", "o": "óòôö", "u": "úùûü", "c": "ç", "n": "ñ"}
LEET = {"o": "0", "l": "1", "e": "3", "s": "5", "i": "l", "a": "4"}


class Vocab:
    """Country vocabulary learned from that country's S1 records only."""

    def __init__(self, names, addrs):
        last = Counter(n.split()[-1] for n in names if n.split())
        allt = Counter(t for n in names for t in set(n.split()))
        # legal forms: frequent tokens that occur (almost) only in last position
        self.legal = [t for t, c in last.most_common(60)
                      if c >= 0.8 * allt[t] and c >= 0.005 * len(names)][:15]
        self.category = [t for t, _ in allt.most_common(60) if t not in self.legal][:30]
        # street types: tokens that directly follow a house number ("55 rue", "36 avenue")
        st = Counter()
        for a in addrs:
            toks = a.replace(",", " ").split()
            st.update(toks[i + 1] for i in range(len(toks) - 1) if toks[i].isdigit() and toks[i + 1].isalpha())
        self.street = {t for t, _ in st.most_common(15)}


def _typo(tok, rng):
    if len(tok) < 4:
        return tok
    i = rng.randrange(1, len(tok) - 1)
    op = rng.random()
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op < 0.3:
        return tok[:i] + tok[i + 1:]
    if op < 0.55:
        return tok[:i] + rng.choice(letters) + tok[i:]
    if op < 0.8:
        return tok[:i] + rng.choice(letters) + tok[i + 1:]
    return tok[:i - 1] + tok[i] + tok[i - 1] + tok[i + 1:]


def _match_case(new, like):
    return new.upper() if like.isupper() else new.capitalize() if like[:1].isupper() else new


def noisy_name(name, v, s, rng):
    toks = name.split()
    if not toks:
        return name
    if rng.random() < P_ACRONYM and len(toks) >= 2:
        return "".join(t[0] for t in toks if t[0].isalnum()).upper()
    if rng.random() < P_LEGAL and v.legal:
        if toks[-1].lower() in v.legal and len(toks) > 1 and rng.random() < 0.5:
            toks = toks[:-1]                                             # drop legal form
        elif toks[-1].lower() in v.legal:
            toks[-1] = _match_case(rng.choice(v.legal), toks[-1])       # swap legal form
        else:
            toks.append(_match_case(rng.choice(v.legal), toks[0]))      # add one
    if rng.random() < P_CATEGORY and v.category:
        toks.append(_match_case(rng.choice(v.category), toks[0]))
    if rng.random() < P_TOKDROP and len(toks) > 2:
        del toks[rng.randrange(1, len(toks))]
    if rng.random() < P["reorder"][s]:
        rng.shuffle(toks)
    if rng.random() < P_TYPO:
        i = rng.randrange(len(toks))
        toks[i] = _typo(toks[i], rng)
    if rng.random() < P["brackets"][s] and len(toks) > 1:
        i = rng.randrange(1, len(toks))
        toks[i] = rng.choice(["(%s)", "[%s]"]) % toks[i]
    out = " ".join(toks)
    if rng.random() < P["domain"][s]:
        core = [t for t in toks if t.lower().strip("()[]") not in v.legal] or toks
        return "".join(re.sub(r"\W", "", t) for t in core).lower() + ".com"
    if rng.random() < P["accent"][s]:
        idx = [i for i, c in enumerate(out) if c.lower() in ACCENTS]
        if idx:
            i = rng.choice(idx)
            c = rng.choice(ACCENTS[out[i].lower()])
            out = out[:i] + (c.upper() if out[i].isupper() else c) + out[i + 1:]
    if rng.random() < P["leet"][s]:
        idx = [i for i, c in enumerate(out) if c.lower() in LEET]
        if idx:
            i = rng.choice(idx)
            out = out[:i] + LEET[out[i].lower()] + out[i + 1:]
    if rng.random() < P["double_space"][s]:
        out = out.replace(" ", "  ", 1)
    if rng.random() < P["tag"][s]:
        out += f" [#{rng.randrange(10000, 99999)}]"
    if rng.random() < P["phone"][s]:
        out += f" - {rng.randrange(6_000_000_000, 9_999_999_999)}"
    r = rng.random()
    if r < P["name_upper"][s]:
        out = out.upper()
    elif r < P["name_upper"][s] + P["name_lower"][s]:
        out = out.lower()
    return out


def noisy_addr(addr, v, s, rng):
    if addr is None or rng.random() < P["addr_missing"][s]:
        return None
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if rng.random() < P["addr_drop"][s] and len(parts) > 1:
        for _ in range(rng.choice([1, 1, 2])):
            if len(parts) > 1:
                del parts[rng.randrange(len(parts))]
    if rng.random() < P_ADDR_REORDER and len(parts) > 1:
        rng.shuffle(parts)
    toks = ", ".join(parts).split(" ")
    if rng.random() < P_ADDR_ABBR:
        idx = [i for i, t in enumerate(toks) if t.strip(",").lower() in v.street]
        if idx:
            i = rng.choice(idx)
            t = toks[i].rstrip(",")
            toks[i] = t[:rng.choice([1, 2, 3])] + "." + ("," if toks[i].endswith(",") else "")
    if rng.random() < P_ADDR_TYPO:
        idx = [i for i, t in enumerate(toks) if len(t) >= 4 and t.isalpha()]
        if idx:
            i = rng.choice(idx)
            toks[i] = _typo(toks[i], rng)
    if rng.random() < P_NUM:
        idx = [i for i, t in enumerate(toks) if re.fullmatch(r"\d+,?", t)]
        if idx:
            i = rng.choice(idx)
            d = toks[i].rstrip(",")
            op = rng.random()
            d = d[1:] if (op < 0.4 and len(d) > 1) else ("0" + d if op < 0.7 else d[:-1] + str(rng.randrange(10)))
            toks[i] = d + ("," if toks[i].endswith(",") else "")
    out = " ".join(toks)
    if rng.random() < P["hash"][s]:
        out = re.sub(r"(^|\s)(\d)", r"\1##\2", out, count=1)
    if rng.random() < P["junk"][s]:
        parts = out.split(", ")
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(["N/A", "None", "null"]))
        out = ", ".join(parts)
    if rng.random() < P["addr_upper"][s]:
        out = out.upper()
    return out


def make_split(split, s1, seed=0):
    """s1: DataFrame(entity_id, business_name, business_address, country) of ONE country.
    Writes data_syn/<split>/<split>_source{1,2,3}.tsv and <split>_ground_truth.tsv."""
    rng = random.Random(seed)
    names, addrs = s1["business_name"].to_list(), s1["business_address"].to_list()
    v = Vocab([n.lower() for n in names], [(a or "").lower() for a in addrs])
    country = s1["country"][0]
    order = list(range(s1.height))
    rng.shuffle(order)
    keep = set(order[: len(order) // 2])  # half A keeps its S1; half B -> orphan decoys
    rows1, rows2, rows3, gt = [], [], [], []
    c2 = c3 = 0
    for i in range(s1.height):
        sid = f"S1-{i}"
        n2 = rng.choices(range(len(N2_DIST)), N2_DIST)[0]
        n3 = rng.choices(range(len(N3_DIST)), N3_DIST)[0]
        ids = []
        for _ in range(n2):
            rows2.append((f"S2-{c2}", noisy_name(names[i], v, 0, rng), noisy_addr(addrs[i], v, 0, rng), country))
            ids.append(f"S2-{c2}")
            c2 += 1
        for _ in range(n3):
            rows3.append((f"S3-{c3}", noisy_name(names[i], v, 1, rng), noisy_addr(addrs[i], v, 1, rng), country))
            ids.append(f"S3-{c3}")
            c3 += 1
        if i in keep:
            rows1.append((sid, names[i], addrs[i], country))
            gt.append((sid, ",".join(ids)))
    d = os.path.join(SYN_ROOT, split)
    os.makedirs(d, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    for k, rows in ((1, rows1), (2, rows2), (3, rows3)):
        pl.DataFrame(rows, schema=cols, orient="row").write_csv(
            os.path.join(d, f"{split}_source{k}.tsv"), separator="\t", quote_style="never", null_value="")
    pl.DataFrame(gt, schema=["source1_entity_id", "matched_entity_ids"], orient="row").write_csv(
        os.path.join(d, f"{split}_ground_truth.tsv"), separator="\t", quote_style="never")
    print(f"[{split}] {len(rows1):,} S1, {len(rows2):,} S2, {len(rows3):,} S3 synthetic records "
          f"({country}); legal forms learned: {v.legal[:12]}")
    return v


if __name__ == "__main__":
    # usage: python -m src.synth <split_name> <train|test> <country> <n_s1>
    split, src_split, country, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    s1 = read_tsv(os.path.join(ROOT, "data", src_split, f"{src_split}_source1.tsv"))
    s1 = s1.filter(pl.col("country") == country)
    make_split(split, s1.sample(min(n, s1.height), seed=0))

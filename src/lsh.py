"""MinHash + banded LSH over character 3-grams, implemented with numba.

Strings are packed into one flat uint32 code-point array (+ offsets), so there are
no per-record Python objects. Grams are taken *within tokens* (a token shorter than
3 chars is one gram), which makes the representation invariant to token order
("Novo Limited Private Insurance" == "Novo Insurance Private Limited").

Candidate retrieval: for every band, S1 and S2/S3 records sharing the band key (in
the same country) collide. Buckets larger than `cap` on the S2/S3 side are skipped
(they are generic keys that carry no identity). Colliding pairs are scored by the
MinHash Jaccard estimate and the top-K per S1 are kept.
"""
import numpy as np
import numba as nb

EMPTY = np.uint32(0xFFFFFFFF)


def pack(strings):
    """list[str] -> (codepoints uint32[], offsets int64[n+1])"""
    lens = np.fromiter((len(s) for s in strings), dtype=np.int64, count=len(strings))
    off = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    cps = np.frombuffer("".join(strings).encode("utf-32-le"), dtype=np.uint32)
    return cps, off


@nb.njit(inline="always")
def _mix(x):
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return x


@nb.njit(parallel=True, cache=True)
def minhash(cps, off, seeds):
    n = off.shape[0] - 1
    P = seeds.shape[0]
    sig = np.full((n, P), 0xFFFFFFFF, dtype=np.uint32)
    for i in nb.prange(n):
        a, b = off[i], off[i + 1]
        j = a
        while j < b:
            if cps[j] == 32:
                j += 1
                continue
            k = j
            while k < b and cps[k] != 32:
                k += 1
            L = k - j
            ng = 1 if L < 3 else L - 2
            for g in range(ng):
                h = np.uint64(1469598103934665603)
                m = L if L < 3 else 3
                for t in range(m):
                    h = (h ^ np.uint64(cps[j + g + t])) * np.uint64(1099511628211)
                for p in range(P):
                    v = np.uint32(_mix(h ^ seeds[p]) >> np.uint64(32))
                    if v < sig[i, p]:
                        sig[i, p] = v
            j = k
    return sig


def signatures(strings, num_perm, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**63 - 1, size=num_perm, dtype=np.int64).astype(np.uint64)
    cps, off = pack(strings)
    return minhash(cps, off, seeds)


@nb.njit(parallel=True, cache=True)
def band_keys(sig, r, country):
    n, P = sig.shape
    nb_ = P // r
    keys = np.empty((n, nb_), dtype=np.uint64)
    for i in nb.prange(n):
        empty = sig[i, 0] == 0xFFFFFFFF
        for b in range(nb_):
            if empty:
                keys[i, b] = 0
                continue
            h = _mix(np.uint64(country[i]) + np.uint64(b) * np.uint64(0x9E3779B97F4A7C15))
            for t in range(r):
                h = _mix(h ^ np.uint64(sig[i, b * r + t]))
            keys[i, b] = h | np.uint64(1)  # 0 reserved for "empty record"
    return keys


@nb.njit(cache=True)
def _collide(qk, qid, sk_sorted, sid_sorted, cap):
    """For each query key find the matching run in the sorted index keys."""
    # pass 1: count
    lo = np.searchsorted(sk_sorted, qk, side="left")
    hi = np.searchsorted(sk_sorted, qk, side="right")
    tot = 0
    for i in range(qk.shape[0]):
        w = hi[i] - lo[i]
        if qk[i] != 0 and 0 < w <= cap:
            tot += w
    out_q = np.empty(tot, dtype=np.int32)
    out_s = np.empty(tot, dtype=np.int32)
    p = 0
    for i in range(qk.shape[0]):
        w = hi[i] - lo[i]
        if qk[i] != 0 and 0 < w <= cap:
            for j in range(lo[i], hi[i]):
                out_q[p] = qid[i]
                out_s[p] = sid_sorted[j]
                p += 1
    return out_q, out_s


def lsh_pairs(q_keys, q_ids, s_keys, s_ids, cap):
    """All (query, index) pairs sharing at least one band key; unique pairs."""
    qs, ss = [], []
    for b in range(q_keys.shape[1]):
        order = np.argsort(s_keys[:, b], kind="stable")
        q, s = _collide(q_keys[:, b], q_ids, s_keys[order, b], s_ids[order], cap)
        qs.append(q)
        ss.append(s)
    code = (np.concatenate(qs).astype(np.int64) << 32) | np.concatenate(ss).astype(np.int64)
    code = np.unique(code)
    return (code >> 32).astype(np.int32), (code & 0xFFFFFFFF).astype(np.int32)


@nb.njit(parallel=True, cache=True)
def sig_sim(sig, a, b):
    """MinHash Jaccard estimate between rows a[k] and b[k]; -1 if either is empty."""
    n = a.shape[0]
    P = sig.shape[1]
    out = np.empty(n, dtype=np.float32)
    for k in nb.prange(n):
        x, y = a[k], b[k]
        if sig[x, 0] == 0xFFFFFFFF or sig[y, 0] == 0xFFFFFFFF:
            out[k] = -1.0
            continue
        m = 0
        for p in range(P):
            if sig[x, p] == sig[y, p]:
                m += 1
        out[k] = m / P
    return out

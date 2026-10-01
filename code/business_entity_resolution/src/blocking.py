"""
Candidate generation (blocking) inside each country.

Every classic blocking key becomes one sparse TF-IDF feature, grouped in three blocks
that are L2-normalised separately so that neither side can swamp the other (Indian /
US business names repeat across hundreds of entities, so a name-only index ranks
same-name businesses in other cities above the true, slightly misspelt record):

  name block     w|<core token>   b|<sorted token pair>   c|<compact name / domain stem>
  address block  h|<house no>|<street word>   hn|<house no>|<name token>  (postcode+name
                 analogue: postcodes are almost absent in this data)
                 hl|<house no>|<locality word>   x|<compound id: D-12 -> d12, 22/235>
                 nb|<consecutive address numbers>  (house keys use the first two numbers)
                 hv|<house no with its first or last digit dropped>|<street word>  (the noise
                 clips house numbers: 2424 -> 424, 17177 -> 1717; crossed so that only an exact
                 number on one side meets a clipped variant on the other)
                 sl|<street word>|<locality word>  (records whose house number was dropped)
                 s|<street word>  l|<locality word>  n|<address number>
  char block     character 3-grams of the core name (typos, scrambles, glued words)

Retrieval is an exact sparse top-k nearest-neighbour search (numba, parallel inverted
index accumulation) from every Source 1 record into all Source 2/3 records of the same
country. One pass over the index per query accumulates the three block cosines and
keeps three top-k lists, which are unioned:
    combined  = wn*cos_name + wa*cos_addr + wc*cos_char      (top k_comb)
    name-only = name + char blocks  (records with missing / different address)
    addr-only = address block       (records whose name was replaced by an alias)
The union is ranked by the combined score (the "cheap similarity") and the best
`max_candidates` per Source 1 record are kept - no hard key-density cut-offs. Only
features with df > max_df are excluded from the index (they carry ~no IDF weight).
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
from numba import njit, prange

NS_CODES = {"w": 0, "b": 1, "c": 2, "h": 3, "hn": 4, "s": 5, "l": 6, "n": 7, "g": 8, "x": 9, "hl": 10, "nb": 11,
            "hv": 12, "sl": 13}
NAME_NS = {"w": 1.0, "b": 1.0, "c": 1.5}
ADDR_NS = {"h": 1.5, "hn": 1.5, "hl": 1.2, "x": 1.5, "nb": 1.2, "s": 1.0, "l": 0.6, "n": 0.6, "hv": 1.2, "sl": 0.8}
BLOCK_WEIGHTS = {"name": 0.40, "addr": 0.45, "char": 0.15}
STREET_TYPE_TOKENS = {
    "st", "rd", "ave", "dr", "ln", "blvd", "ct", "cir", "pl", "ter", "pkwy", "hwy", "way", "trl",
    "sq", "plz", "xing", "pt", "pike", "expy", "fwy", "tpke", "aly", "hts", "mt", "ft", "rte",
    "rue", "allee", "imp", "ch", "quai", "crs", "pass", "cite", "res", "chs", "lot", "marg",
    "ne", "nw", "se", "sw", "prom", "esp", "sent", "cour", "ste",
}
BLOCK_COLUMNS = ["n_core", "n_cmp", "n_full", "a_house", "a_street", "a_loc", "a_nums", "a_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation (polars, vectorised, chunked to bound memory)
# ─────────────────────────────────────────────────────────────────────────────

def _tok_list(col: str, min_len: int = 2) -> pl.Expr:
    return (pl.col(col).str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= min_len))
            .list.unique(maintain_order=True))


def _word_features_chunk(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    base = df.select(
        (pl.int_range(0, pl.len(), dtype=pl.UInt32) + offset).alias("row"),
        _tok_list("n_core").alias("nt"),
        pl.col("n_cmp"),
        pl.col("n_full").str.replace_all(" ", "").alias("n_fc"),
        pl.col("a_house"),
        _tok_list("a_street", 3).list.eval(
            pl.element().filter(~pl.element().is_in(list(STREET_TYPE_TOKENS)))).alias("st"),
        _tok_list("a_loc", 3).alias("lt"),
        _tok_list("a_nums", 1).alias("nums"),
        _tok_list("a_ids", 2).alias("ids"),
    )
    # two house-number candidates: noise often injects a number in front ("H.NO 992 ..., 15, ...")
    h0 = pl.col("a_house")
    h1 = pl.col("nums").list.get(1, null_on_oob=True)
    toks4 = [pl.col("nt").list.get(i, null_on_oob=True) for i in range(4)]
    pairs = []
    for a in range(4):
        for b in range(a + 1, 4):
            ta, tb = toks4[a], toks4[b]
            pairs.append(pl.when(ta < tb).then(ta + "_" + tb).otherwise(tb + "_" + ta))
    nums5 = [pl.col("nums").list.get(i, null_on_oob=True) for i in range(5)]
    num_bigrams = [nums5[i] + "_" + nums5[i + 1] for i in range(4)]
    specs = [
        ("w", pl.col("nt"), None),
        ("b", pl.concat_list(pairs), None),
        ("c", pl.concat_list([
            pl.when(pl.col("n_cmp").str.len_chars() >= 4).then(pl.col("n_cmp")),
            pl.when(pl.col("n_fc").str.len_chars() >= 4).then(pl.col("n_fc"))]).list.unique(), None),
        ("x", pl.col("ids"), None),
        ("nb", pl.concat_list(num_bigrams), None),
        ("s", pl.col("st"), None),
        ("l", pl.col("lt"), None),
        ("n", pl.col("nums"), None),
    ]
    for house in (h0, h1):
        specs += [("h", pl.col("st").list.head(3), house),
                  ("hn", pl.col("nt").list.head(2), house),
                  ("hl", pl.col("lt").list.head(2), house)]
    parts = []
    for ns, expr, house in specs:
        if house is None:
            pre = pl.lit("")
            sel = base.select("row", expr.alias("f"), pre.alias("pre"))
        else:
            sel = base.select("row", expr.alias("f"), pl.concat_str([pl.lit("#"), house, pl.lit("#")]).alias("pre"))
            sel = sel.filter(pl.col("pre").is_not_null() & (pl.col("pre") != "##"))
        part = (sel.explode("f")
                .filter(pl.col("f").is_not_null() & (pl.col("f") != ""))
                .select("row", pl.concat_str([pl.lit(ns + "|"), pl.col("pre"), pl.col("f")])
                        .hash(seed=17).alias("h"),
                        pl.lit(NS_CODES[ns], dtype=pl.UInt8).alias("ns")))
        parts.append(part)
    return pl.concat(parts)


def _retrieval_chunk(df: pl.DataFrame, offset: int, side: int) -> pl.DataFrame:
    """
    Retrieval-only address keys (need to know the side):
      hv  house number with its first or last digit dropped x street word. Source 1 emits its
          exact number as E and its clipped variants as V; Source 2/3 the other way round, so
          exact<->variant pairs match in both directions but two variants never do.
      sl  street word x locality word, for records whose house number was dropped.
    """
    b = df.select((pl.int_range(0, pl.len(), dtype=pl.UInt32) + offset).alias("row"), pl.col("a_house").alias("hs"),
                  _tok_list("a_street", 3).list.eval(
                      pl.element().filter(~pl.element().is_in(list(STREET_TYPE_TOKENS)))).list.head(3).alias("st"),
                  _tok_list("a_loc", 3).list.head(2).alias("lt"))
    digits = b.filter(pl.col("hs").str.contains(r"^\d{2,}$"))
    variants = pl.concat([digits.select("row", pl.col("hs").str.slice(1).alias("v"), "st"),
                          digits.select("row", pl.col("hs").str.head(-1).alias("v"), "st")])
    exact = b.filter(pl.col("hs") != "").select("row", pl.col("hs").alias("v"), "st")
    tag_exact, tag_var = ("E", "V") if side == 1 else ("V", "E")
    parts = []
    for frame, tag in ((exact, tag_exact), (variants, tag_var)):
        p = frame.explode("st").filter(pl.col("st").is_not_null() & (pl.col("v") != ""))
        parts.append(p.select("row", pl.concat_str([pl.lit(f"hv|{tag}#"), pl.col("v"), pl.lit("#"), pl.col("st")])
                              .hash(seed=17).alias("h"), pl.lit(NS_CODES["hv"], dtype=pl.UInt8).alias("ns")))
    sl = (b.select("row", "st", "lt").explode("st").explode("lt")
          .filter(pl.col("st").is_not_null() & pl.col("lt").is_not_null()))
    parts.append(sl.select("row", pl.concat_str([pl.lit("sl|"), pl.col("st"), pl.lit("|"), pl.col("lt")])
                           .hash(seed=17).alias("h"), pl.lit(NS_CODES["sl"], dtype=pl.UInt8).alias("ns")))
    return pl.concat(parts)


def word_features(df: pl.DataFrame, chunk: int = 400_000, side: int = None) -> pl.DataFrame:
    """
    Long frame (row:u32, h:u64, ns:u8) of hashed key features. side=1 (Source 1) or 2
    (Source 2/3) adds the retrieval-only keys hv / sl.
    """
    parts = []
    for s in range(0, len(df), chunk):
        parts.append(_word_features_chunk(df.slice(s, chunk), s))
        if side is not None:
            parts.append(_retrieval_chunk(df.slice(s, chunk), s, side))
    return pl.concat(parts)


def _char_features_chunk(df: pl.DataFrame, offset: int, n: int) -> pl.DataFrame:
    s = df.select((pl.int_range(0, pl.len(), dtype=pl.UInt32) + offset).alias("row"),
                  (pl.lit(" ") + pl.col("n_core") + pl.lit(" ")).alias("s"))
    s = s.filter(pl.col("s").str.len_chars() > n + 1)
    s = s.with_columns(pl.int_ranges(0, pl.col("s").str.len_chars() - n + 1).alias("off")).explode("off")
    s = s.with_columns(pl.col("s").str.slice(pl.col("off"), n).alias("g"))
    s = s.filter(~pl.col("g").str.contains("^ *$"))
    return s.select("row", pl.col("g").hash(seed=29).alias("h"),
                    pl.lit(NS_CODES["g"], dtype=pl.UInt8).alias("ns")).unique(["row", "h"])


def char_features(df: pl.DataFrame, n: int = 3, chunk: int = 400_000) -> pl.DataFrame:
    return pl.concat([_char_features_chunk(df.slice(s, chunk), s, n) for s in range(0, len(df), chunk)])


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(f1: pl.DataFrame, f2: pl.DataFrame, n1: int, n2: int, ns_weights: Dict[str, float],
                max_df: int, min_df: int = 2, normalize: bool = True
                ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    TF-IDF rows for both sides over the features that occur on both sides (only those
    can ever produce a match), dropping features with df > max_df. Rows are
    L2-normalised unless normalize=False (raw idf weights, for containment features).
    """
    if len(f1) == 0 or len(f2) == 0:
        return sp.csr_matrix((n1, 1), dtype=np.float32), sp.csr_matrix((n2, 1), dtype=np.float32)
    h1 = f1["h"].to_numpy()
    r1 = f1["row"].to_numpy()
    ns1 = f1["ns"].to_numpy()
    u1, inv1, c1 = np.unique(h1, return_inverse=True, return_counts=True)
    del h1
    ns_u = np.zeros(len(u1), dtype=np.uint8)
    ns_u[inv1] = ns1
    del ns1
    h2 = f2["h"].to_numpy()
    r2 = f2["row"].to_numpy()
    pos = np.searchsorted(u1, h2)
    pos[pos >= len(u1)] = 0
    hit = u1[pos] == h2
    del h2
    pos, r2 = pos[hit], r2[hit]
    c2 = np.bincount(pos, minlength=len(u1))
    df = c1 + c2
    keep = (c2 > 0) & (df >= min_df) & (df <= max_df)
    code_w = np.ones(16, dtype=np.float32)
    for k, v in ns_weights.items():
        code_w[NS_CODES[k]] = v
    w = (np.log((n1 + n2) / np.maximum(df, 1)) * code_w[ns_u]).astype(np.float32)
    col = np.full(len(u1), -1, dtype=np.int64)
    col[keep] = np.arange(int(keep.sum()))
    V = max(int(keep.sum()), 1)
    out = []
    for rows, vid, n in ((r1, inv1, n1), (r2, pos, n2)):
        m = keep[vid]
        mat = sp.csr_matrix((w[vid[m]], (rows[m], col[vid[m]])), shape=(n, V), dtype=np.float32)
        mat.sum_duplicates()
        mat.sort_indices()
        out.append(l2norm(mat) if normalize else mat)
    return out[0], out[1]


def l2norm(m: sp.csr_matrix) -> sp.csr_matrix:
    lens = np.diff(m.indptr)
    sq = np.sqrt(np.add.reduceat(np.append(m.data ** 2, 0).astype(np.float64), m.indptr[:-1]) * (lens > 0))
    sq[sq == 0] = 1.0
    m.data = (m.data / np.repeat(sq, lens)).astype(np.float32)
    return m


def weighted_hstack(mats: List[sp.csr_matrix], weights: List[float]) -> sp.csr_matrix:
    m = sp.hstack([mm * np.float32(np.sqrt(w)) for mm, w in zip(mats, weights)], format="csr", dtype=np.float32)
    m.sort_indices()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Sparse kernels (numba)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _pair_dot(ip_a, ix_a, v_a, ip_b, ix_b, v_b, I, J):
    out = np.zeros(I.shape[0], dtype=np.float32)
    for k in prange(I.shape[0]):
        i, j = I[k], J[k]
        pa, ea = ip_a[i], ip_a[i + 1]
        pb, eb = ip_b[j], ip_b[j + 1]
        s = 0.0
        while pa < ea and pb < eb:
            ca, cb = ix_a[pa], ix_b[pb]
            if ca == cb:
                s += v_a[pa] * v_b[pb]
                pa += 1
                pb += 1
            elif ca < cb:
                pa += 1
            else:
                pb += 1
        out[k] = s
    return out


def pair_dot(A: sp.csr_matrix, B: sp.csr_matrix, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Row-wise dot products <A[I[k]], B[J[k]]> (both CSR with sorted indices)."""
    return _pair_dot(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data,
                     np.ascontiguousarray(I, dtype=np.int64), np.ascontiguousarray(J, dtype=np.int64))


@njit(cache=True)
def _heap_push(hs, hj, size, k, s, j):
    """Min-heap of the k best (score, id); returns the new size."""
    if size < k:
        hs[size] = s
        hj[size] = j
        pos = size
        size += 1
        while pos > 0:
            par = (pos - 1) // 2
            if hs[pos] < hs[par]:
                hs[pos], hs[par] = hs[par], hs[pos]
                hj[pos], hj[par] = hj[par], hj[pos]
                pos = par
            else:
                break
        return size
    if s <= hs[0]:
        return size
    hs[0] = s
    hj[0] = j
    pos = 0
    while True:
        l = 2 * pos + 1
        if l >= size:
            break
        r = l + 1
        c = l
        if r < size and hs[r] < hs[l]:
            c = r
        if hs[c] < hs[pos]:
            hs[c], hs[pos] = hs[pos], hs[c]
            hj[c], hj[pos] = hj[pos], hj[c]
            pos = c
        else:
            break
    return size


@njit(parallel=True, cache=True)
def _topk3_kernel(q_ptr, q_idx, q_val, t_ptr, t_idx, t_val, blk, n2,
                  wn, wa, wc, k0, k1, k2, n_chunks, out0, out1, out2):
    """
    One pass over the inverted index per query, accumulating the three block cosines
    separately, then three heaps:
      out0 = combined (wn*name + wa*addr + wc*char), out1 = name+char, out2 = address.
    """
    n1 = q_ptr.shape[0] - 1
    step = (n1 + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        start = c * step
        end = min(n1, start + step)
        if start >= end:
            continue
        acc = np.zeros((3, n2), dtype=np.float32)
        seen = np.zeros(n2, dtype=np.uint8)
        touched = np.empty(n2, dtype=np.int32)
        hs0 = np.empty(k0, dtype=np.float32)
        hj0 = np.empty(k0, dtype=np.int32)
        hs1 = np.empty(k1, dtype=np.float32)
        hj1 = np.empty(k1, dtype=np.int32)
        hs2 = np.empty(k2, dtype=np.float32)
        hj2 = np.empty(k2, dtype=np.int32)
        for i in range(start, end):
            nt = 0
            for p in range(q_ptr[i], q_ptr[i + 1]):
                f = q_idx[p]
                w = q_val[p]
                b = blk[f]
                for r in range(t_ptr[f], t_ptr[f + 1]):
                    j = t_idx[r]
                    if seen[j] == 0:
                        seen[j] = 1
                        touched[nt] = j
                        nt += 1
                    acc[b, j] += w * t_val[r]
            s0 = 0
            s1 = 0
            s2 = 0
            for t in range(nt):
                j = touched[t]
                cn = acc[0, j]
                ca = acc[1, j]
                cc = acc[2, j]
                acc[0, j] = 0.0
                acc[1, j] = 0.0
                acc[2, j] = 0.0
                seen[j] = 0
                s0 = _heap_push(hs0, hj0, s0, k0, wn * cn + wa * ca + wc * cc, j)
                if k1 > 0 and (cn > 0.0 or cc > 0.0):
                    s1 = _heap_push(hs1, hj1, s1, k1, 0.7 * cn + 0.3 * cc, j)
                if k2 > 0 and ca > 0.0:
                    s2 = _heap_push(hs2, hj2, s2, k2, ca, j)
            for t in range(s0):
                out0[i, t] = hj0[t]
            for t in range(s1):
                out1[i, t] = hj1[t]
            for t in range(s2):
                out2[i, t] = hj2[t]


def topk3_search(Q: sp.csr_matrix, D: sp.csr_matrix, blk: np.ndarray, weights: Tuple[float, float, float],
                 ks: Tuple[int, int, int], n_chunks: int = 512) -> np.ndarray:
    """Union of the three top-k lists as flat keys i*n2 + j."""
    n1, n2 = Q.shape[0], D.shape[0]
    T = D.T.tocsr()
    outs = [np.full((n1, max(k, 1)), -1, dtype=np.int32) for k in ks]
    _topk3_kernel(Q.indptr.astype(np.int64), Q.indices.astype(np.int32), Q.data.astype(np.float32),
                  T.indptr.astype(np.int64), T.indices.astype(np.int32), T.data.astype(np.float32),
                  blk.astype(np.int8), n2, np.float32(weights[0]), np.float32(weights[1]),
                  np.float32(weights[2]), ks[0], ks[1], ks[2], n_chunks, outs[0], outs[1], outs[2])
    del T
    keys = []
    for out in outs:
        I = np.repeat(np.arange(n1, dtype=np.int64), out.shape[1])
        J = out.ravel().astype(np.int64)
        m = J >= 0
        keys.append(I[m] * n2 + J[m])
    return np.unique(np.concatenate(keys))


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation for one country
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(s1: pl.DataFrame, s23: pl.DataFrame, k_comb: int = 25, k_name: int = 10,
                        k_addr: int = 15, max_candidates: int = 45, max_df_word: int = 5000,
                        max_df_char: int = 2000, block_weights: Dict[str, float] = None,
                        verbose: bool = True) -> pl.DataFrame:
    """
    Returns a polars frame (i1:u32 row in s1, j:u32 row in s23, cos_name, cos_addr,
    cos_char, cheap, rank) with at most `max_candidates` rows per Source 1 record.
    """
    bw = dict(BLOCK_WEIGHTS, **(block_weights or {}))
    t0 = time.time()
    n1, n2 = len(s1), len(s23)

    def log(msg):
        if verbose:
            print(f"    [{time.time()-t0:5.0f}s] {msg}", flush=True)

    fw1, fw2 = word_features(s1, side=1), word_features(s23, side=2)
    name_ns = [NS_CODES[k] for k in NAME_NS]
    addr_ns = [NS_CODES[k] for k in ADDR_NS]
    N1, N2 = build_tfidf(fw1.filter(pl.col("ns").is_in(name_ns)), fw2.filter(pl.col("ns").is_in(name_ns)),
                         n1, n2, NAME_NS, max_df_word)
    A1, A2 = build_tfidf(fw1.filter(pl.col("ns").is_in(addr_ns)), fw2.filter(pl.col("ns").is_in(addr_ns)),
                         n1, n2, ADDR_NS, max_df_word)
    del fw1, fw2
    C1, C2 = build_tfidf(char_features(s1), char_features(s23), n1, n2, {"g": 1.0}, max_df_char)
    log(f"tf-idf: name {N1.shape[1]:,} addr {A1.shape[1]:,} char {C1.shape[1]:,} features; "
        f"S23 nnz {N2.nnz + A2.nnz + C2.nnz:,}")
    Q = sp.hstack([N1, A1, C1], format="csr", dtype=np.float32)
    D = sp.hstack([N2, A2, C2], format="csr", dtype=np.float32)
    blk = np.concatenate([np.zeros(N1.shape[1]), np.ones(A1.shape[1]), np.full(C1.shape[1], 2)])
    key = topk3_search(Q, D, blk, (bw["name"], bw["addr"], bw["char"]), (k_comb, k_name, k_addr))
    del Q, D
    log(f"top-k union: {len(key):,} pairs ({len(key)/max(n1,1):.1f}/S1)")
    I = (key // n2).astype(np.int64)
    J = (key % n2).astype(np.int64)
    del key
    cn, ca, cc = pair_dot(N1, N2, I, J), pair_dot(A1, A2, I, J), pair_dot(C1, C2, I, J)
    cands = pl.DataFrame({
        "i1": I.astype(np.uint32), "j": J.astype(np.uint32), "cos_name": cn, "cos_addr": ca,
        "cos_char": cc, "cheap": bw["name"] * cn + bw["addr"] * ca + bw["char"] * cc})
    cands = cands.with_columns(
        pl.col("cheap").rank("ordinal", descending=True).over("i1").cast(pl.UInt16).alias("rank"))
    cands = cands.filter(pl.col("rank") <= max_candidates).sort(["i1", "rank"])
    log(f"kept {len(cands):,} pairs ({len(cands)/max(n1,1):.1f}/S1)")
    return cands


@njit(parallel=True, cache=True)
def _pair_stats(ip_a, ix_a, v_a, ip_b, ix_b, v_b, I, J, out):
    """out[k] = (dot, sum_a over shared, sum_b over shared, |a|^2, |b|^2, sum_a, sum_b)."""
    for k in prange(I.shape[0]):
        i, j = I[k], J[k]
        pa, ea = ip_a[i], ip_a[i + 1]
        pb, eb = ip_b[j], ip_b[j + 1]
        dot = 0.0
        sa = 0.0
        sb = 0.0
        na = 0.0
        nb = 0.0
        ta = 0.0
        tb = 0.0
        for q in range(pa, ea):
            na += v_a[q] * v_a[q]
            ta += v_a[q]
        for q in range(pb, eb):
            nb += v_b[q] * v_b[q]
            tb += v_b[q]
        while pa < ea and pb < eb:
            ca, cb = ix_a[pa], ix_b[pb]
            if ca == cb:
                dot += v_a[pa] * v_b[pb]
                sa += v_a[pa]
                sb += v_b[pb]
                pa += 1
                pb += 1
            elif ca < cb:
                pa += 1
            else:
                pb += 1
        out[k, 0] = dot / np.sqrt(max(na * nb, 1e-12))
        out[k, 1] = sa / max(ta, 1e-12)
        out[k, 2] = sb / max(tb, 1e-12)
        out[k, 3] = ea - ip_a[i]
        out[k, 4] = eb - ip_b[j]


def pair_stats(A: sp.csr_matrix, B: sp.csr_matrix, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    For raw (un-normalised) idf rows: cosine, idf-weighted containment of A's features in
    B and of B's in A, and the feature counts of both rows. Returns float32 (n, 5).
    """
    out = np.zeros((len(I), 5), dtype=np.float32)
    _pair_stats(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data,
                np.ascontiguousarray(I, dtype=np.int64), np.ascontiguousarray(J, dtype=np.int64), out)
    return out

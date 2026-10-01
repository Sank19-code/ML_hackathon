"""
Second-stage context features and the final decisions.

Group features (computed from first-stage probabilities p1, per country):
  * inside a Source 1 entity: rank, gap to the best candidate, number of strong
    candidates, probability mass, best score / rank inside the same source
  * across entities competing for the same Source 2/3 record (one-to-one structure:
    in the ground truth every S2/S3 record belongs to at most one Source 1 entity):
    how many entities retrieved it, this entity's rank, margin over the best rival
  * the same margins for the blocking address / name cosines, and how many entities
    have a strong address match with the record (multi-tenant addresses)
  * sibling clusters: how many other candidates carry the same house number at a sibling
    offset from the entity's number, and how many confident candidates confirm the entity's
    own number
  * S2 <-> S3 cluster consistency: similarity of the candidate to the entity's other
    strong candidates, weighted by their p1 (records that agree with the rest of the
    cluster are supported, stray ones are not)

Decisions (after one_to_one: each S2/S3 record goes only to the entity where it scores highest):
  apply_thresholds / tune_thresholds   per-source thresholds, coordinate ascent on macro F0.5
  expected_f_select / tune_expected_f  per-entity choice of the top-k candidates (or none)
                                       that maximises the expected F0.5 of the entity
"""
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl

try:
    from .metrics import entity_scores
except ImportError:
    from metrics import entity_scores

GROUP_FEATURES = [
    "p1", "p1_rank_s1", "p1_max_s1", "p1_gap_s1", "p1_n50_s1", "p1_n80_s1", "p1_sum_s1",
    "p1_max_src_s1", "p1_rank_src_s1", "n_s1_j", "p1_rank_j", "p1_max_other_j", "p1_margin_j",
    "support_max", "support_mean", "addr_margin_j", "name_margin_j", "n_addr_hi_j",
    "sib_cluster_n", "n_conf_ha_hi",
    # v6: exact-twin / content-core context and shift-stable counts
    "n_conf_ha_src", "n_conf_cc_eq", "n_conf_cc_eq_src", "p1_nunc_s1", "p1_rank_nonsib_s1",
]


def j_aggregates(pairs: pl.DataFrame) -> pl.DataFrame:
    """
    Per Source 2/3 record: number of competing entities, the two best p1 and the two best
    blocking address / name cosines among the entities that retrieved it.
    """
    return pairs.group_by("j").agg(
        pl.len().cast(pl.Float32).alias("n_s1_j"),
        pl.col("p1").max().alias("first_j"),
        pl.col("p1").top_k(2).min().alias("second_j"),
        pl.col("cos_addr").max().alias("a1_j"),
        pl.col("cos_addr").top_k(2).min().alias("a2_j"),
        pl.col("cos_name").max().alias("n1_j"),
        pl.col("cos_name").top_k(2).min().alias("n2_j"),
        (pl.col("cos_addr") >= 0.6).sum().cast(pl.Float32).alias("n_addr_hi_j"),
        pl.len().alias("_n"),
    ).with_columns([pl.when(pl.col("_n") < 2).then(0.0).otherwise(pl.col(c)).alias(c)
                    for c in ("second_j", "a2_j", "n2_j")]).drop("_n")


def group_features_chunk(chunk: pl.DataFrame, jagg: pl.DataFrame, sim_fn=None, top: int = 4) -> np.ndarray:
    """
    chunk: rows (i1, j, p1, is_s2, cos_addr, cos_name, hb, sib, has_ha, cc) containing complete
    Source 1 groups (hb = candidate house number, sib = sibling-offset flag, has_ha = candidate
    numbers contain the Source 1 house number, cc = content-core Jaccard, -1 unknown).
    sim_fn(J, K): similarity between Source 2/3 records J[i] and K[i] (cluster support).
    Returns float32 (n, len(GROUP_FEATURES)) in chunk row order.
    """
    if "cc" not in chunk.columns:
        chunk = chunk.with_columns(pl.lit(-1.0).alias("cc"))
    df = chunk.select("i1", "j", "p1", "is_s2", "cos_addr", "cos_name", "hb", "sib", "has_ha", "cc").with_columns(
        pl.int_range(0, pl.len()).alias("_pos"))
    df = df.join(jagg, on="j", how="left").with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("i1").cast(pl.Float32).alias("p1_rank_s1"),
        pl.col("p1").max().over("i1").alias("p1_max_s1"),
        (pl.col("p1") > 0.5).sum().over("i1").cast(pl.Float32).alias("p1_n50_s1"),
        (pl.col("p1") > 0.8).sum().over("i1").cast(pl.Float32).alias("p1_n80_s1"),
        pl.col("p1").sum().over("i1").alias("p1_sum_s1"),
        pl.col("p1").max().over(["i1", "is_s2"]).alias("p1_max_src_s1"),
        pl.col("p1").rank("ordinal", descending=True).over(["i1", "is_s2"]).cast(pl.Float32).alias("p1_rank_src_s1"),
    ).with_columns(
        (pl.col("p1_max_s1") - pl.col("p1")).alias("p1_gap_s1"),
        pl.when(pl.col("p1") >= pl.col("first_j")).then(1.0)
        .when(pl.col("p1") >= pl.col("second_j")).then(2.0).otherwise(3.0).alias("p1_rank_j"),
        pl.when(pl.col("p1") >= pl.col("first_j")).then(pl.col("second_j"))
        .otherwise(pl.col("first_j")).alias("p1_max_other_j"),
    ).with_columns(
        (pl.col("p1") - pl.col("p1_max_other_j")).alias("p1_margin_j"),
        # sibling clusters: other candidates carrying the same sibling-offset house number, and
        # confident candidates confirming the Source 1 house number
        (pl.col("sib").sum().over(["i1", "hb"]) - pl.col("sib")).cast(pl.Float32).alias("sib_cluster_n"),
        ((pl.col("has_ha") & (pl.col("p1") > 0.5)).cast(pl.Int32).sum().over("i1")
         - (pl.col("has_ha") & (pl.col("p1") > 0.5)).cast(pl.Int32)).cast(pl.Float32).alias("n_conf_ha_hi"),
        (pl.col("cos_addr") - pl.when(pl.col("cos_addr") >= pl.col("a1_j")).then(pl.col("a2_j"))
         .otherwise(pl.col("a1_j"))).alias("addr_margin_j"),
        (pl.col("cos_name") - pl.when(pl.col("cos_name") >= pl.col("n1_j")).then(pl.col("n2_j"))
         .otherwise(pl.col("n1_j"))).alias("name_margin_j"),
        # a confident copy from the SAME source confirming the entity's house number / content core
        # (a second copy with a typo vs a neighbouring business)
        ((pl.col("has_ha") & (pl.col("p1") > 0.5)).cast(pl.Int32).sum().over(["i1", "is_s2"])
         - (pl.col("has_ha") & (pl.col("p1") > 0.5)).cast(pl.Int32)).cast(pl.Float32).alias("n_conf_ha_src"),
        (((pl.col("cc") >= 1.0) & (pl.col("p1") > 0.5)).cast(pl.Int32).sum().over("i1")
         - ((pl.col("cc") >= 1.0) & (pl.col("p1") > 0.5)).cast(pl.Int32)).cast(pl.Float32).alias("n_conf_cc_eq"),
        (((pl.col("cc") >= 1.0) & (pl.col("p1") > 0.5)).cast(pl.Int32).sum().over(["i1", "is_s2"])
         - ((pl.col("cc") >= 1.0) & (pl.col("p1") > 0.5)).cast(pl.Int32)).cast(pl.Float32).alias("n_conf_cc_eq_src"),
        ((pl.col("p1") >= 0.1) & (pl.col("p1") < 0.9)).sum().over("i1").cast(pl.Float32).alias("p1_nunc_s1"),
        pl.col("p1").rank("ordinal", descending=True).over(["i1", "sib"]).cast(pl.Float32).alias("p1_rank_nonsib_s1"),
    )

    if sim_fn is not None:
        tops = (df.filter(pl.col("p1_rank_s1") <= top)
                .select("i1", pl.col("j").alias("k"), pl.col("p1").alias("pk")))
        pr = df.select("i1", "j").join(tops, on="i1", how="inner").filter(pl.col("j") != pl.col("k"))
        if len(pr):
            sim = sim_fn(pr["j"].to_numpy(), pr["k"].to_numpy())
            pr = pr.with_columns((pl.Series(sim) * pl.col("pk")).alias("s"))
            agg = pr.group_by(["i1", "j"]).agg(pl.col("s").max().alias("support_max"),
                                              pl.col("s").mean().alias("support_mean"))
            df = df.join(agg, on=["i1", "j"], how="left")
    for c in ("support_max", "support_mean"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns(pl.col("support_max").fill_null(0.0), pl.col("support_mean").fill_null(0.0))
    df = df.sort("_pos")
    return df.select([pl.col(c).cast(pl.Float32) for c in GROUP_FEATURES]).to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Decisions
# ─────────────────────────────────────────────────────────────────────────────

def one_to_one(df: pl.DataFrame, prob: str = "p", key: str = "eid") -> pl.DataFrame:
    """Keep, for every Source 2/3 record, only its highest-scoring entity."""
    return df.sort(prob, descending=True).unique(key, keep="first", maintain_order=False)


def apply_thresholds(best: pl.DataFrame, thresholds: Dict[str, float], default: float,
                     prob: str = "p", group: str = "src") -> pl.DataFrame:
    tau = pl.col(group).replace_strict(list(thresholds.keys()), list(thresholds.values()),
                                       default=default, return_dtype=pl.Float64)
    return best.filter(pl.col(prob) >= tau)


def tune_thresholds(best: pl.DataFrame, entities: pl.DataFrame, gt: pl.DataFrame, prob: str = "p",
                    group: str = "src", grid: Optional[np.ndarray] = None, rounds: int = 2,
                    init: float = 0.5) -> Tuple[Dict[str, float], float]:
    """
    best: one-to-one filtered pairs (s1, eid, <group>, <prob>) for one country.
    Coordinate ascent of one threshold per group value, maximising macro F0.5.
    """
    if grid is None:
        grid = np.round(np.arange(0.10, 0.991, 0.02), 3)
    groups = sorted(best[group].unique().to_list())
    th = {g: init for g in groups}
    best = best.select("s1", "eid", group, prob)

    def score(t):
        pred = apply_thresholds(best, t, init, prob, group)
        return entity_scores(entities, gt, pred)["f05"].mean()

    cur = score(th)
    for _ in range(rounds):
        for g in groups:
            for v in grid:
                trial = dict(th)
                trial[g] = float(v)
                s = score(trial)
                if s > cur + 1e-7:
                    cur, th = s, trial
    return th, cur


def expected_f_select(best: pl.DataFrame, floor: float = 0.05, missed: float = 0.0, gamma: float = 1.0,
                      prob: str = "p") -> pl.DataFrame:
    """
    Per-entity decision that maximises the expected F0.5 of the entity.

    With calibrated, independent match probabilities p_i (after one-to-one assignment),
    F0.5 = 1.25 TP / (0.25 |true| + |pred|). For the top-k candidates the expectation is
    approximated by 1.25 * sum_topk(p) / (0.25 * (sum(p) + missed) + k); predicting nothing
    scores 1 only when the entity is a singleton, probability prod(1 - p_i). The best k
    (possibly 0) is chosen per entity. `missed` accounts for true matches outside the
    candidate set, `gamma` sharpens / flattens the probabilities, `floor` drops negligible
    candidates.
    """
    df = best.filter(pl.col(prob) >= floor).with_columns((pl.col(prob) ** gamma).clip(0.0, 0.999999).alias("_q"))
    df = df.sort(["s1", "_q"], descending=[False, True]).with_columns(
        pl.col("_q").cum_sum().over("s1").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
        pl.col("_q").sum().over("s1").alias("_tot"),
        (1.0 - pl.col("_q")).log().sum().over("s1").exp().alias("_p0"),
    ).with_columns(
        (1.25 * pl.col("_cs") / (0.25 * (pl.col("_tot") + missed) + pl.col("_k"))).alias("_ef"))
    df = df.with_columns(pl.col("_ef").max().over("s1").alias("_efmax"))
    kbest = (df.filter(pl.col("_ef") == pl.col("_efmax")).group_by("s1")
             .agg(pl.col("_k").min().alias("_kb"), pl.col("_efmax").first(), pl.col("_p0").first()))
    kbest = kbest.filter(pl.col("_efmax") > pl.col("_p0"))
    out = df.join(kbest.select("s1", "_kb"), on="s1", how="inner").filter(pl.col("_k") <= pl.col("_kb"))
    return out.drop([c for c in out.columns if c.startswith("_")])


def tune_expected_f(best: pl.DataFrame, entities: pl.DataFrame, gt: pl.DataFrame, prob: str = "p"):
    """Small grid over (gamma, missed, floor) for the expected-F rule."""
    res = []
    for gamma in (0.8, 1.0, 1.25, 1.5, 2.0):
        for missed in (0.0, 0.1, 0.25):
            for floor in (0.05, 0.2, 0.35):
                pred = expected_f_select(best, floor, missed, gamma, prob)
                f = entity_scores(entities, gt, pred.select("s1", "eid"))["f05"].mean()
                res.append((f, gamma, missed, floor))
    res.sort(reverse=True)
    return {"gamma": res[0][1], "missed": res[0][2], "floor": res[0][3]}, res[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration + exact expected-F0.5 decision (v6)
# ─────────────────────────────────────────────────────────────────────────────

def fit_isotonic(p: np.ndarray, y: np.ndarray, n_knots: int = 400) -> Dict[str, list]:
    """
    Isotonic regression of y on p (pool-adjacent-violators on quantile bins), returned as a
    monotone piecewise-linear map {"x": knots, "y": values} that np.interp can apply.
    """
    order = np.argsort(p, kind="stable")
    p, y = p[order].astype(np.float64), y[order].astype(np.float64)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_knots + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    sy = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    sx = np.bincount(idx, weights=p, minlength=len(edges) - 1)
    keep = cnt > 0
    cnt, sy, sx = cnt[keep], sy[keep], sx[keep]
    # pool adjacent violators on the bin means (weights = counts)
    vals, wts, xs = [], [], []
    for c, a, b in zip(cnt, sy / cnt, sx / cnt):
        vals.append(a); wts.append(c); xs.append(b * c)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            x = xs[-2] + xs[-1]
            vals[-2:], wts[-2:], xs[-2:] = [v], [w], [x]
    kx = [x / w for x, w in zip(xs, wts)]
    ky = list(np.clip(vals, 1e-6, 1 - 1e-6))
    return {"x": [0.0] + kx + [1.0], "y": [min(ky[0], 1e-6)] + ky + [max(ky[-1], 1 - 1e-6)]}


def apply_calibration(p: np.ndarray, cal: Optional[Dict[str, list]], temperature: float = 1.0) -> np.ndarray:
    """Calibrated probability; temperature > 1 softens it in logit space (unlabelled countries)."""
    q = np.asarray(p, dtype=np.float64) if cal is None else np.interp(p, cal["x"], cal["y"])
    if temperature != 1.0:
        q = np.clip(q, 1e-6, 1 - 1e-6)
        q = 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)) / temperature))
    return q


try:
    from numba import njit, prange

    @njit(cache=True)
    def _conv_bern(pmf, q):
        T = len(pmf) - 1
        out = np.empty_like(pmf)
        out[0] = pmf[0] * (1.0 - q)
        for x in range(1, T + 1):
            out[x] = pmf[x] * (1.0 - q) + pmf[x - 1] * q
        out[T] += pmf[T] * q
        return out

    @njit(parallel=True, cache=True)
    def _exact_ef_kernel(ptr, p, lam, kmax, T, out_k, out_ef, out_e0):
        n_ent = len(ptr) - 1
        for e in prange(n_ent):
            s, t = ptr[e], ptr[e + 1]
            n = t - s
            K = min(n, kmax)
            # Poisson(lam) pmf for true matches outside the scored list, tail folded into T
            pois = np.zeros(T + 1)
            pois[0] = np.exp(-lam[e])
            acc = pois[0]
            for x in range(1, T):
                pois[x] = pois[x - 1] * lam[e] / x
                acc += pois[x]
            pois[T] = max(0.0, 1.0 - acc)
            cur = pois
            for i in range(n - 1, K - 1, -1):
                cur = _conv_bern(cur, p[s + i])
            suf = np.empty((K + 1, T + 1))
            suf[K] = cur
            for k in range(K - 1, -1, -1):
                cur = _conv_bern(cur, p[s + k])
                suf[k] = cur
            best_k, best_v = 0, suf[0][0]          # predict nothing: F = 1 only if no true match at all
            out_e0[e] = suf[0][0]
            pre = np.zeros(T + 1)
            pre[0] = 1.0
            for k in range(1, K + 1):
                pre = _conv_bern(pre, p[s + k - 1])
                v = 0.0
                for a in range(1, min(k, T) + 1):
                    if pre[a] == 0.0:
                        continue
                    inner = 0.0
                    for c in range(0, T + 1):
                        inner += suf[k][c] / (k + 0.25 * (a + c))
                    v += pre[a] * 1.25 * a * inner
                if v > best_v + 1e-12:
                    best_k, best_v = k, v
            out_k[e] = best_k
            out_ef[e] = best_v
except ImportError:  # pragma: no cover
    _exact_ef_kernel = None


def exact_f_select(best: pl.DataFrame, lam=None, kmax: int = 15, tmax: int = 30,
                   prob: str = "q", min_p: float = 1e-3, return_ef: bool = False):
    """
    Exact expected-F0.5 top-k decision per Source 1 entity (Ye et al. 2012; Poisson-binomial DP).

    best : one-to-one pairs (s1, eid, <prob>, ...) with CALIBRATED probabilities in `prob`
    lam  : expected number of true matches outside the candidate list: a scalar (same for every
           entity) or a frame (s1, lam)
    Candidates below min_p are folded into the Poisson term. Predicting nothing is chosen when
    P(no true match anywhere) beats every top-k set.
    """
    df = best.with_columns(pl.col(prob).cast(pl.Float64).clip(0.0, 1.0 - 1e-9).alias("_q"))
    tail = df.filter(pl.col("_q") < min_p).group_by("s1").agg(pl.col("_q").sum().alias("_tail"))
    df = df.filter(pl.col("_q") >= min_p).sort(["s1", "_q"], descending=[False, True])
    ents = df.group_by("s1", maintain_order=True).agg(pl.len().alias("_n"))
    ents = ents.join(tail, on="s1", how="left")
    if isinstance(lam, pl.DataFrame):
        ents = ents.join(lam.select("s1", pl.col("lam").alias("_lam")), on="s1", how="left")
    else:
        ents = ents.with_columns(pl.lit(float(lam or 0.0)).alias("_lam"))
    ents = ents.with_columns((pl.col("_lam").fill_null(0.0) + pl.col("_tail").fill_null(0.0)).alias("_lamt"))
    ptr = np.zeros(len(ents) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(ents["_n"].to_numpy())
    k = np.zeros(len(ents), dtype=np.int64)
    ef = np.zeros(len(ents), dtype=np.float64)
    e0 = np.zeros(len(ents), dtype=np.float64)
    _exact_ef_kernel(ptr, df["_q"].to_numpy(), ents["_lamt"].to_numpy().astype(np.float64), kmax, tmax, k, ef, e0)
    ents = ents.select("s1").with_columns(pl.Series("_kb", k), pl.Series("_ef", ef), pl.Series("_e0", e0))
    out = (df.with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("_k"))
           .join(ents.select("s1", "_kb"), on="s1", how="inner").filter(pl.col("_k") <= pl.col("_kb")))
    out = out.drop([c for c in out.columns if c.startswith("_")])
    return (out, ents) if return_ef else out


def reoffer_rejected(sc: pl.DataFrame, best: pl.DataFrame, chosen: pl.DataFrame, select_fn,
                     prob: str = "q") -> pl.DataFrame:
    """
    Exclusivity-aware second chance: a Source 2/3 record that its best entity did not accept is
    offered to its second-best entity; the entity keeps it only if its own exact expected-F
    decision (re-run with the record added) selects it. Records stay unassigned otherwise.

    sc     : all scored pairs (s1, eid, <prob>)
    best   : one-to-one pairs used for `chosen`
    chosen : current selection (s1, eid, ...)
    select_fn(frame with s1, eid, <prob>) -> selection frame (same decision rule)
    Returns the new selection as (s1, eid, <prob>).
    """
    cols = ["s1", "eid", prob]
    sc, best, chosen = sc.select(cols), best.select(cols), chosen.select(cols)
    rejected = best.join(chosen.select("eid"), on="eid", how="anti").select("eid")
    second = (sc.join(rejected, on="eid", how="semi")
              .join(best.select("s1", "eid"), on=["s1", "eid"], how="anti")
              .sort(prob, descending=True).unique("eid", keep="first"))
    if len(second) == 0:
        return chosen
    touched = second.select("s1").unique()
    trial = pl.concat([best.join(touched, on="s1", how="semi"), second], how="vertical_relaxed")
    new = select_fn(trial).select(cols)
    if len(new.join(second.select("s1", "eid"), on=["s1", "eid"], how="semi")) == 0:
        return chosen
    return pl.concat([chosen.join(touched, on="s1", how="anti"), new], how="vertical_relaxed")


def shape_buckets(ha: pl.Series, hb: pl.Series, core_a: pl.Series, core_b: pl.Series,
                  offsets) -> pl.Series:
    """
    Pair shapes for optional per-shape logit offsets of unlabelled countries (unused: the submitted
    model has no offsets):
      "sibling"         candidate house number = Source 1 number + a sibling offset
      "samehouse_swap"  same house number, the core names differ by exactly one dissimilar word
      ""                anything else
    """
    from rapidfuzz import fuzz
    d = hb.cast(pl.Int64, strict=False) - ha.cast(pl.Int64, strict=False)
    sib = d.is_in(offsets).fill_null(False).to_numpy()
    same = ((ha == hb) & (ha != "")).to_numpy()
    df = pl.DataFrame({"a": core_a.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(),
                       "b": core_b.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()})
    df = df.select(pl.col("a").list.set_difference("b").alias("da"), pl.col("b").list.set_difference("a").alias("db"))
    one = ((df["da"].list.len() == 1) & (df["db"].list.len() == 1)).to_numpy()
    swap = np.zeros(len(df), dtype=bool)
    idx = np.where(one & same)[0]
    if len(idx):
        from rapidfuzz import process
        wa = df["da"].list.first().gather(idx).to_list()
        wb = df["db"].list.first().gather(idx).to_list()
        swap[idx] = process.cpdist(wa, wb, scorer=fuzz.ratio, workers=-1) < 70
    out = np.where(sib, "sibling", np.where(swap, "samehouse_swap", ""))
    return pl.Series("bucket", out)


def apply_bucket_offsets(q: np.ndarray, buckets: pl.Series, offsets: Dict[str, float]) -> np.ndarray:
    """Shift the calibrated probability of each pre-registered shape by a logit offset."""
    if not offsets:
        return q
    b = np.array([float(offsets.get(x, 0.0)) for x in buckets.to_list()]) if len(buckets) else np.zeros(0)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + b)))

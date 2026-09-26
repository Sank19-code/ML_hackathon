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
    chunk: rows (i1, j, p1, is_s2, cos_addr, cos_name, hb, sib, has_ha) containing complete
    Source 1 groups (hb = candidate house number, sib = sibling-offset flag, has_ha = candidate
    numbers contain the Source 1 house number).
    sim_fn(J, K): similarity between Source 2/3 records J[i] and K[i] (cluster support).
    Returns float32 (n, len(GROUP_FEATURES)) in chunk row order.
    """
    df = chunk.select("i1", "j", "p1", "is_s2", "cos_addr", "cos_name", "hb", "sib", "has_ha").with_columns(
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

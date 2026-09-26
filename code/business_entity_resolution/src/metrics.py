"""
Official metric: per Source 1 entity F_0.5, macro-averaged over all Source 1 entities
(singletons included: 1.0 for an empty prediction, 0.0 otherwise).

compute_entity_f05 / compute_macro_f05 are the reference (dict based) implementations;
macro_f05_frame is the vectorised polars version used for large evaluations and for the
breakdowns tracked after every change (overall, by country, singletons, blocking recall).
"""
from typing import Dict, Optional, Set

import polars as pl


def compute_entity_f05(gt_set: Set[str], pred_set: Set[str]) -> float:
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(gt_set)
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    if not ground_truth:
        return 0.0
    return sum(compute_entity_f05(g, predictions.get(s, set())) for s, g in ground_truth.items()) / len(ground_truth)


def entity_scores(entities: pl.DataFrame, gt_pairs: pl.DataFrame, pred_pairs: pl.DataFrame) -> pl.DataFrame:
    """
    entities   : frame with column s1 (+ any grouping columns, e.g. country)
    gt_pairs   : (s1, eid) true matches
    pred_pairs : (s1, eid) predicted matches
    Returns entities with n_true, n_pred, tp, f05 columns.
    """
    gt_pairs = gt_pairs.select("s1", "eid").unique()
    pred_pairs = pred_pairs.select("s1", "eid").unique()
    n_true = gt_pairs.group_by("s1").agg(pl.len().alias("n_true"))
    n_pred = pred_pairs.group_by("s1").agg(pl.len().alias("n_pred"))
    tp = pred_pairs.join(gt_pairs, on=["s1", "eid"], how="inner").group_by("s1").agg(pl.len().alias("tp"))
    e = (entities.join(n_true, on="s1", how="left").join(n_pred, on="s1", how="left")
         .join(tp, on="s1", how="left")
         .with_columns(pl.col("n_true").fill_null(0), pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0)))
    prec = pl.col("tp") / pl.col("n_pred")
    rec = pl.col("tp") / pl.col("n_true")
    f = (1.25 * prec * rec) / (0.25 * prec + rec)
    e = e.with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when((pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
        .otherwise(f).alias("f05"))
    return e


def report(entities: pl.DataFrame, gt_pairs: pl.DataFrame, pred_pairs: pl.DataFrame,
           cand_pairs: Optional[pl.DataFrame] = None, title: str = "") -> Dict[str, float]:
    """Print and return macro F0.5 overall / by country / singletons (+ blocking recall)."""
    e = entity_scores(entities, gt_pairs, pred_pairs)
    out = {"f05": e["f05"].mean(), "n": len(e)}
    tp = e["tp"].sum()
    out["pair_precision"] = tp / max(e["n_pred"].sum(), 1)
    out["pair_recall"] = tp / max(e["n_true"].sum(), 1)
    sing = e.filter(pl.col("n_true") == 0)
    out["f05_singletons"] = sing["f05"].mean() if len(sing) else float("nan")
    out["f05_nonsingletons"] = e.filter(pl.col("n_true") > 0)["f05"].mean()
    if "country" in e.columns:
        for c, v in e.group_by("country").agg(pl.col("f05").mean()).iter_rows():
            out[f"f05_{c}"] = v
    if cand_pairs is not None:
        g = gt_pairs.select("s1", "eid").unique()
        hit = g.join(cand_pairs.select("s1", "eid").unique(), on=["s1", "eid"], how="semi")
        out["block_pair_recall"] = len(hit) / max(len(g), 1)
        per = (g.with_columns(pl.lit(1).alias("t"))
               .join(hit.with_columns(pl.lit(1).alias("h")), on=["s1", "eid"], how="left")
               .group_by("s1").agg(pl.col("h").fill_null(0).mean()))
        out["block_entity_recall"] = per["h"].mean()
    if title:
        print(f"  [{title}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in out.items()))
    return out

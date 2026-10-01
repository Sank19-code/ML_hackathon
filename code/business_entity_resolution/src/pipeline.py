"""
End-to-end pipeline: data -> normalisation -> blocking -> features -> models -> decisions -> output.

Stages (cached in the work directory; each can be re-run on its own):
  translit   learn the native-script -> Latin word dictionary from the training labels
  normalize  canonical name / address fields for every source file
  block      per-country TF-IDF top-k candidate generation (train and test universes)
  train      stage 1: LightGBM on pair features (+ ambiguity context), cross-fitted by
                      Source 1 entity -> out-of-fold p1 for every training pair
             stage 2: LightGBM on pair features + group / one-to-one / cluster-support
                      features built from p1 -> out-of-fold p
             decisions: one-to-one assignment + per (country, source) thresholds or the
                      expected-F0.5 rule, tuned on one half of the entities and evaluated
                      on the other half
  predict    score the test universe with the fold ensembles, assign, write the TSVs

Pair features are computed on the fly in chunks (about 11 microseconds per pair) rather
than stored: the full feature matrices would need ~16 GB of disk. Everything runs country
by country (matches never cross countries in the ground truth) with integer row keys, so
memory stays bounded on a 16 GB machine. Countries are an open set: a country without
training labels (France in the test set) uses the same country-agnostic model with the pooled
calibration softened by a temperature (1.5) and no missed-match term, settings chosen on
leave-one-country-out rehearsals.
"""
import gc
import json
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    from . import blocking, decide, features, metrics, model
    from .data import prepare_source, read_tsv
    from .normalize import load_translit_dict
    from .translit import learn_translit_dict, save_translit_dict
except ImportError:
    import blocking, decide, features, metrics, model
    from data import prepare_source, read_tsv
    from normalize import load_translit_dict
    from translit import learn_translit_dict, save_translit_dict

DEFAULT_CONFIG = {
    "n_jobs": 14,
    "block": {"k_comb": 25, "k_name": 10, "k_addr": 20, "max_candidates": 50,
              "max_df_word": 5000, "max_df_char": 2000},
    # countries whose Source 2/3 records are partly in a non-Latin script (transliterated names overlap
    # less lexically, renamed businesses rank just below the top 25) get a deeper combined list; decided
    # from the data of each country, never from its name (+0.26 recall points where it applies)
    "deep_block": {"min_native_share": 0.05, "k_comb": 40, "max_candidates": 60},
    "stage1_sample": 0.20,          # fraction of training Source 1 entities used to fit stage 1 (shipped model)
    "stage2_sample": 0.20,          # fraction used to fit stage 2 (shipped model)
    "stage1_rounds": 700,
    "stage2_rounds": 500,
    "unseen_country_margin": 0.05,  # stricter default decision for countries without labels
    # countries without labels: pooled calibration softened by a temperature and the
    # blocking-miss term chosen on leave-one-country-out runs; optional logit offsets per pre-registered
    # pair shape (an unused hook: empty in the submitted model; no leaderboard feedback is used)
    # temperature 1.5 / lam 0 is the only setting that improved BOTH leave-one-country-out directions
    # (US model on India 0.96774 -> 0.96807, India model on US 0.97766 -> 0.97769)
    "unlabelled": {"temperature": 1.5, "lam": 0.0, "bucket_offsets": {}},
}
EXACT_LAMS = (0.0, 0.05, 0.1, 0.2, 0.4)
CHUNK = 1_500_000
CONTEXT_FEATURES = ["s1_name_dup", "s1_name_in_s23", "s1_addr_dup", "b_name_dup", "b_name_in_s1", "b_addr_dup"]
STAGE1_NAMES = features.FEATURE_NAMES + CONTEXT_FEATURES
STAGE2_NAMES = STAGE1_NAMES + decide.GROUP_FEATURES
CC_COL = STAGE1_NAMES.index("cc_jacc")


class Workspace:
    def __init__(self, work_dir: str, model_dir: str):
        self.work = work_dir
        self.model = model_dir
        for d in ("norm", "cands", "scores"):
            os.makedirs(os.path.join(work_dir, d), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def norm(self, split, k):
        return os.path.join(self.work, "norm", f"{split}_s{k}.parquet")

    def cands(self, split, country):
        return os.path.join(self.work, "cands", f"{split}_{_safe(country)}.parquet")

    def ids(self, split, country, side):
        return os.path.join(self.work, "cands", f"{split}_{_safe(country)}_{side}ids.parquet")

    def scores(self, split, country):
        return os.path.join(self.work, "scores", f"{split}_{_safe(country)}.parquet")

    def tmp(self, name):
        os.makedirs(os.path.join(self.work, "tmp"), exist_ok=True)
        return os.path.join(self.work, "tmp", name)

    @property
    def translit(self):
        return os.path.join(self.model, "translit.json")


def _remove(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _safe(country: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (country or "NA"))


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gt_long(path: str) -> pl.DataFrame:
    gt = read_tsv(path)
    return (gt.select(pl.col("source1_entity_id").alias("s1"),
                      pl.col("matched_entity_ids").str.split(",").alias("eid"))
            .explode("eid").filter(pl.col("eid").is_not_null() & (pl.col("eid") != "")))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: transliteration dictionary + normalisation
# ─────────────────────────────────────────────────────────────────────────────

def stage_translit(ws: Workspace, train_dir: str):
    if os.path.isfile(ws.translit):
        return
    _log("learning transliteration dictionary from training pairs")
    s1 = read_tsv(os.path.join(train_dir, "train_source1.tsv")).select("entity_id", "business_name")
    s23 = pl.concat([read_tsv(os.path.join(train_dir, f"train_source{k}.tsv")).select("entity_id", "business_name")
                     for k in (2, 3)])
    s23 = s23.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
    gt = gt_long(os.path.join(train_dir, "train_ground_truth.tsv"))
    j = (gt.join(s23, left_on="eid", right_on="entity_id")
         .join(s1.rename({"business_name": "n1"}), left_on="s1", right_on="entity_id"))
    mapping = learn_translit_dict(zip(j["n1"].to_list(), j["business_name"].to_list()))
    save_translit_dict(mapping, ws.translit)
    _log(f"  {len(mapping):,} native words learned")


def stage_normalize(ws: Workspace, split: str, data_dir: str, n_jobs: int):
    load_translit_dict(ws.translit)
    for k in (1, 2, 3):
        prepare_source(os.path.join(data_dir, f"{split}_source{k}.tsv"), ws.norm(split, k), ws.translit, n_jobs)


# ─────────────────────────────────────────────────────────────────────────────
# Stage: blocking (per country)
# ─────────────────────────────────────────────────────────────────────────────

def countries(ws: Workspace, split: str) -> List[str]:
    c = pl.read_parquet(ws.norm(split, 1), columns=["country"])["country"]
    return sorted(c.unique().to_list())


def load_country(ws: Workspace, split: str, country: str, columns=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.read_parquet(ws.norm(split, 1), columns=columns).filter(pl.col("country") == country)
    s23 = pl.concat([pl.read_parquet(ws.norm(split, k), columns=columns).filter(pl.col("country") == country)
                     for k in (2, 3)])
    return s1, s23


def block_params(cfg: Dict, native_share: float) -> Dict:
    """Blocking depth from a statistic of the country's own records (open set: no country names)."""
    params = dict(cfg["block"])
    deep = cfg.get("deep_block")
    if deep and native_share >= deep["min_native_share"]:
        params.update({k: v for k, v in deep.items() if k != "min_native_share"})
    return params


def stage_block(ws: Workspace, split: str, cfg: Dict, force: bool = False):
    only = cfg.get("block_countries")
    for country in countries(ws, split):
        if only and country not in only:
            continue
        path = ws.cands(split, country)
        if os.path.isfile(path) and not force:
            continue
        _log(f"blocking {split}/{country}")
        cols = ["entity_id", "country", "is_native"] + blocking.BLOCK_COLUMNS
        s1, s23 = load_country(ws, split, country, cols)
        s1.select("entity_id").write_parquet(ws.ids(split, country, "s1"))
        s23.select("entity_id").write_parquet(ws.ids(split, country, "s23"))
        if len(s23) == 0 or len(s1) == 0:
            cands = pl.DataFrame(schema={"i1": pl.UInt32, "j": pl.UInt32, "cos_name": pl.Float32,
                                         "cos_addr": pl.Float32, "cos_char": pl.Float32,
                                         "cheap": pl.Float32, "rank": pl.UInt16})
        else:
            params = block_params(cfg, float(s23["is_native"].mean() or 0.0))
            _log(f"  blocking parameters {params}")
            cands = blocking.generate_candidates(s1, s23.drop("is_native"), **params)
        cands = cands.with_columns(pl.len().over("i1").cast(pl.UInt16).alias("n_cands"))
        cands.write_parquet(path)
        del s1, s23, cands
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Per-country data: candidate table + on-the-fly features
# ─────────────────────────────────────────────────────────────────────────────

def context_arrays(s1: pl.DataFrame, s23: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ambiguity context per record: how many records of the country share its compact name
    (on its own side and on the other side) and its full normalised address. Names repeat
    across hundreds of entities in this data, so a common name must be confirmed by the
    address.
    """
    nm1 = s1.group_by("n_cmp").agg(pl.len().alias("c1"))
    nm2 = s23.group_by("n_cmp").agg(pl.len().alias("c2"))
    ad1 = s1.group_by("a_norm").agg(pl.len().alias("ad"))
    ad2 = s23.group_by("a_norm").agg(pl.len().alias("ad"))

    def side(df, own_nm, other_nm, own_ad, own_c, other_c):
        return (df.select("n_cmp", "a_norm")
                .join(own_nm, on="n_cmp", how="left", maintain_order="left")
                .join(other_nm, on="n_cmp", how="left", maintain_order="left")
                .join(own_ad, on="a_norm", how="left", maintain_order="left")
                .select(pl.col(own_c).fill_null(0), pl.col(other_c).fill_null(0),
                        pl.when(pl.col("a_norm") == "").then(0).otherwise(pl.col("ad")).fill_null(0))
                .to_numpy().astype(np.float32))

    return side(s1, nm1, nm2, ad1, "c1", "c2"), side(s23, nm2, nm1, ad2, "c2", "c1")


class CountryData:
    """Candidate pairs of one country plus everything needed to featurise them."""

    def __init__(self, ws: Workspace, split: str, country: str, gt: Optional[pl.DataFrame] = None,
                 with_featurizer: bool = True):
        self.split, self.country = split, country
        self.cands = pl.read_parquet(ws.cands(split, country))
        self.s1ids = pl.read_parquet(ws.ids(split, country, "s1"))["entity_id"]
        self.s23ids = pl.read_parquet(ws.ids(split, country, "s23"))["entity_id"]
        I = self.cands["i1"].to_numpy()
        J = self.cands["j"].to_numpy()
        h1 = self.s1ids.hash(seed=99).to_numpy()
        self.is_s2 = self.s23ids.str.starts_with("S2-").to_numpy()[J].astype(np.int8)
        self.fold = (h1 % 2).astype(np.int8)[I]
        self.h = (h1 // 2 % 10_000).astype(np.int16)[I]
        self.y = None
        if gt is not None:
            g = (gt.join(pl.DataFrame({"s1": self.s1ids, "i1": np.arange(len(self.s1ids), dtype=np.uint32)}), on="s1")
                 .join(pl.DataFrame({"eid": self.s23ids, "j": np.arange(len(self.s23ids), dtype=np.uint32)}), on="eid")
                 .select("i1", "j", pl.lit(1, dtype=pl.Int8).alias("y")))
            self.y = (self.cands.select("i1", "j").join(g, on=["i1", "j"], how="left", maintain_order="left")
                      ["y"].fill_null(0).to_numpy().astype(np.int8))
        self.fz = None
        if with_featurizer and len(self.cands):
            cols = sorted(set(features.NEEDED_COLUMNS) | set(blocking.BLOCK_COLUMNS) | {"country", "a_norm"})
            s1, s23 = load_country(ws, split, country, cols)
            self.ctx_a, self.ctx_b = context_arrays(s1, s23)
            self.fz = features.PairFeaturizer(s1, s23)
            del s1, s23
            gc.collect()

    def __len__(self):
        return len(self.cands)

    def base(self, rows: np.ndarray) -> np.ndarray:
        """Stage-1 inputs (pair features + context) for the given candidate rows."""
        part = self.cands[rows]
        X = self.fz.featurize(part)
        ctx = np.hstack([self.ctx_a[part["i1"].to_numpy()], self.ctx_b[part["j"].to_numpy()]])
        return np.hstack([X, ctx]).astype(np.float32)

    def chunks(self, chunk: int = CHUNK) -> Iterator[Tuple[int, int]]:
        """Row ranges of ~chunk rows that never split a Source 1 entity (rows sorted by i1)."""
        i1 = self.cands["i1"].to_numpy()
        n, s = len(i1), 0
        while s < n:
            e = min(n, s + chunk)
            if e < n:
                e = int(np.searchsorted(i1, i1[e - 1], side="right"))
            yield s, e
            s = e

    def support_sim(self, J: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Record-record similarity of Source 2/3 records: mean of name and address IDF cosines."""
        fz = self.fz
        return 0.5 * (blocking.pair_stats(fz.Bn, fz.Bn, J, K)[:, 0] + blocking.pair_stats(fz.Ba, fz.Ba, J, K)[:, 0])

    def group(self, s: int, e: int, p1: np.ndarray, jagg: pl.DataFrame, cc: Optional[np.ndarray] = None) -> np.ndarray:
        I, J = self.cands["i1"][s:e], self.cands["j"][s:e]
        if cc is None:
            cc = self.fz.content_core(I.to_numpy(), J.to_numpy())["cc_jacc"]
        ha, hb = self.fz.s1["a_house"].gather(I), self.fz.s23["a_house"].gather(J)
        nb = self.fz.s23["a_nums"].gather(J).str.split(" ")
        delta = hb.cast(pl.Int64, strict=False) - ha.cast(pl.Int64, strict=False)
        chunk = pl.DataFrame({"i1": I, "j": J, "p1": p1[s:e], "is_s2": self.is_s2[s:e],
                              "cos_addr": self.cands["cos_addr"][s:e], "cos_name": self.cands["cos_name"][s:e],
                              "hb": hb, "sib": delta.is_in(features.SIBLING_OFFSETS).fill_null(False).cast(pl.Int32),
                              "has_ha": pl.DataFrame({"nb": nb, "ha": ha}).select(
                                  pl.col("nb").list.contains(pl.col("ha")) & (pl.col("ha") != ""))["nb"],
                              "cc": cc})
        return decide.group_features_chunk(chunk, jagg, self.support_sim)


def _columns_for(ens: model.FoldEnsemble, names: List[str]) -> np.ndarray:
    """Positions of the model's features inside a matrix whose columns are `names`."""
    pos = {n: i for i, n in enumerate(names)}
    missing = [n for n in ens.feature_names if n not in pos]
    if missing:
        raise ValueError(f"model expects features not produced by this code: {missing}")
    return np.array([pos[n] for n in ens.feature_names])


def _predict(ens: model.FoldEnsemble, X: np.ndarray, fold: Optional[np.ndarray],
             names: Optional[List[str]] = None) -> np.ndarray:
    """
    fold=None averages the fold models (test); else each row uses the model of its fold.
    `names` are the column names of X; columns are selected by the model's feature names so
    a model trained on an older feature list keeps working when features are added.
    """
    if names is not None and list(names) != list(ens.feature_names):
        X = X[:, _columns_for(ens, names)]
    if fold is None:
        return ens.predict(X).astype(np.float32)
    p = np.empty(len(X), dtype=np.float32)
    for k in (0, 1):
        m = fold == k
        if m.any():
            p[m] = ens.predict(X[m], fold=k)
    return p


def score_stage1(cd: CountryData, ens1: model.FoldEnsemble, oof: bool) -> np.ndarray:
    p1 = np.empty(len(cd), dtype=np.float32)
    t0 = time.time()
    for s, e in cd.chunks():
        p1[s:e] = _predict(ens1, cd.base(np.arange(s, e)), cd.fold[s:e] if oof else None, STAGE1_NAMES)
    _log(f"  stage-1 scored {cd.split}/{cd.country}: {len(cd):,} pairs in {time.time()-t0:.0f}s")
    return p1


def score_stage2(cd: CountryData, ens2: model.FoldEnsemble, p1: np.ndarray, oof: bool,
                 sample_mask: Optional[np.ndarray] = None, sink=None):
    """
    Stage-2 probabilities for all pairs; optionally also the stage-2 inputs of sampled rows, either
    returned as one matrix or handed chunk by chunk to `sink` (e.g. a writer into a disk memory map).
    """
    jagg = decide.j_aggregates(pl.DataFrame({"j": cd.cands["j"], "p1": p1, "cos_addr": cd.cands["cos_addr"],
                                             "cos_name": cd.cands["cos_name"]}))
    p = np.empty(len(cd), dtype=np.float32) if ens2 is not None else None
    samples = []
    t0 = time.time()
    for s, e in cd.chunks():
        need_all = ens2 is not None
        rows = np.arange(s, e)
        if not need_all:
            rows = rows[sample_mask[s:e]]
            if len(rows) == 0:
                continue
        if need_all:   # all rows featurised anyway: reuse the content-core column for the group features
            Xb = cd.base(rows)
            G = cd.group(s, e, p1, jagg, cc=Xb[:, CC_COL])
            X = np.hstack([Xb, G])
        else:
            G = cd.group(s, e, p1, jagg)
            X = np.hstack([cd.base(rows), G[rows - s]])
        if sample_mask is not None:
            keep = sample_mask[rows]
            block = X[keep] if need_all else X
            if sink is not None:
                sink(block)
            else:
                samples.append(block)
        if need_all:
            p[s:e] = _predict(ens2, X, cd.fold[s:e] if oof else None, STAGE2_NAMES)
    _log(f"  stage-2 pass {cd.split}/{cd.country}: {time.time()-t0:.0f}s")
    return p, (np.vstack(samples) if samples else None)


# ─────────────────────────────────────────────────────────────────────────────
# Stage: training
# ─────────────────────────────────────────────────────────────────────────────

def fit_crossfit(X, y: np.ndarray, fold: np.ndarray, names: List[str], rounds: int,
                 params=None) -> model.FoldEnsemble:
    """
    Model k is trained on rows whose fold != k, so it scores fold k out of sample. X is a matrix or
    a list of row blocks (memory maps); it is binned once and each fold model trains on a subset.
    """
    ens = model.FoldEnsemble(names, params, rounds)
    full = ens.dataset(X, y)
    del X
    gc.collect()
    for k in (0, 1):
        tr = np.where(fold != k)[0]
        _log(f"  fitting fold model {k}: {len(tr):,} rows, {int(y[tr].sum()):,} positives")
        ens.fit_subset(full, tr)
    return ens


def stage_train(ws: Workspace, train_dir: str, cfg: Dict, reuse_stage1: bool = False) -> Dict:
    gt = gt_long(os.path.join(train_dir, "train_ground_truth.tsv"))
    cs = cfg.get("train_countries") or countries(ws, "train")
    frac1 = int(cfg["stage1_sample"] * 10_000)
    frac2 = int(cfg["stage2_sample"] * 10_000)

    # ── stage 1: sample, fit, out-of-fold scores ──
    # sampled rows are featurised chunk by chunk into per-country float32 .npy memory maps on disk and
    # binned by LightGBM straight from those blocks (no in-RAM stacking: 2-3x larger samples fit in 16 GB)
    p1_path = os.path.join(ws.model, "stage1.pkl")
    if not (reuse_stage1 and os.path.isfile(p1_path)):
        Xs, ys, fs, paths = [], [], [], []
        for c in cs:
            cd = CountryData(ws, "train", c, gt)
            rows = np.where(cd.h < frac1)[0]
            _log(f"train/{c}: {len(cd):,} pairs, {int(cd.y.sum()):,} positive; stage-1 sample {len(rows):,}")
            path = ws.tmp(f"stage1_X_{_safe(c)}.npy")
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), len(STAGE1_NAMES)))
            for a in range(0, len(rows), 1_000_000):
                mm[a:a + 1_000_000] = cd.base(rows[a:a + 1_000_000])
            mm.flush()
            Xs.append(mm)
            paths.append(path)
            ys.append(cd.y[rows])
            fs.append(cd.fold[rows])
            del cd
            gc.collect()
        ens1 = fit_crossfit(Xs, np.concatenate(ys), np.concatenate(fs), STAGE1_NAMES, cfg["stage1_rounds"])
        del Xs, mm
        gc.collect()
        _remove(paths)
        ens1.save(p1_path)
        _log("stage 1 importance: " + ", ".join(f"{n}={v:.3f}" for n, v in ens1.importance(15)))
    ens1 = model.FoldEnsemble.load(p1_path)

    # ── stage 1 scores + stage 2 sample (one featurisation pass per country) ──
    p1s, X2, y2, f2, paths2 = {}, [], [], [], []
    for c in cs:
        cd = CountryData(ws, "train", c, gt)
        p1 = None
        if reuse_stage1 and os.path.isfile(ws.scores("train", c + "_p1")):
            p1 = pl.read_parquet(ws.scores("train", c + "_p1"))["p1"].to_numpy()
            if len(p1) != len(cd):
                p1 = None
        if p1 is None:
            p1 = score_stage1(cd, ens1, oof=True)
            pl.DataFrame({"p1": p1}).write_parquet(ws.scores("train", c + "_p1"))
        p1s[c] = p1
        mask = cd.h < frac2
        path = ws.tmp(f"stage2_X_{_safe(c)}.npy")
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(int(mask.sum()), len(STAGE2_NAMES)))
        pos = [0]

        def sink(block, mm=mm, pos=pos):
            mm[pos[0]:pos[0] + len(block)] = block
            pos[0] += len(block)
        score_stage2(cd, None, p1, oof=True, sample_mask=mask, sink=sink)
        assert pos[0] == len(mm)
        mm.flush()
        X2.append(mm)
        paths2.append(path)
        y2.append(cd.y[mask])
        f2.append(cd.fold[mask])
        del cd, mm, sink
        gc.collect()
    y2, f2 = np.concatenate(y2), np.concatenate(f2)
    ens2 = fit_crossfit(X2, y2, f2, STAGE2_NAMES, cfg["stage2_rounds"])
    del X2
    gc.collect()
    _remove(paths2)
    ens2.save(os.path.join(ws.model, "stage2.pkl"))
    _log("stage 2 importance: " + ", ".join(f"{n}={v:.3f}" for n, v in ens2.importance(15)))

    # ── stage 2 out-of-fold scores ──
    tables = {}
    for c in cs:
        cd = CountryData(ws, "train", c, gt)
        p, _ = score_stage2(cd, ens2, p1s[c], oof=True)
        tables[c] = pl.DataFrame({"i1": cd.cands["i1"], "j": cd.cands["j"], "is_s2": cd.is_s2, "y": cd.y,
                                  "fold": cd.fold, "p1": p1s[c], "p": p})
        tables[c].write_parquet(ws.scores("train", c))
        del cd
        gc.collect()
    return evaluate_and_tune(ws, tables, gt, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Held-out evaluation and decision tuning
# ─────────────────────────────────────────────────────────────────────────────

def _country_eval_frames(ws: Workspace, country: str, t: pl.DataFrame, gt: pl.DataFrame, prob: str = "p"):
    s1ids = pl.read_parquet(ws.ids("train", country, "s1"))["entity_id"]
    s23ids = pl.read_parquet(ws.ids("train", country, "s23"))["entity_id"]
    ent = pl.DataFrame({"s1": np.arange(len(s1ids), dtype=np.uint32),
                        "fold": (s1ids.hash(seed=99).to_numpy() % 2).astype(np.int8)})
    g = (gt.join(pl.DataFrame({"s1": s1ids, "i1": np.arange(len(s1ids), dtype=np.uint32)}), on="s1")
         .join(pl.DataFrame({"eid": s23ids, "j": np.arange(len(s23ids), dtype=np.uint32)}), on="eid", how="left"))
    # a true match outside this country's S2/S3 pool (never happens in train) keeps a dummy id
    g = g.select(pl.col("i1").alias("s1"), pl.col("j").fill_null(2 ** 32 - 1).alias("eid"))
    sc = t.select(pl.col("i1").alias("s1"), pl.col("j").alias("eid"),
                  pl.when(pl.col("is_s2") == 1).then(pl.lit("S2")).otherwise(pl.lit("S3")).alias("src"),
                  pl.col(prob).alias("p"))
    return ent, g, sc


def _with_y(best: pl.DataFrame, g: pl.DataFrame) -> pl.DataFrame:
    return best.join(g.select("s1", "eid").unique().with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                     on=["s1", "eid"], how="left").with_columns(pl.col("y").fill_null(0))


def _tune_exact(b: pl.DataFrame, ents: pl.DataFrame, g: pl.DataFrame, cal: Dict) -> Tuple[float, float]:
    """Blocking-miss term for the exact expected-F rule on calibrated probabilities."""
    bq = b.with_columns(pl.Series("q", decide.apply_calibration(b["p"].to_numpy(), cal)))
    res = [(metrics.entity_scores(ents, g, decide.exact_f_select(bq, lam).select("s1", "eid"))["f05"].mean(), lam)
           for lam in EXACT_LAMS]
    f, lam = max(res)
    return lam, f


def evaluate_and_tune(ws: Workspace, tables: Dict[str, pl.DataFrame], gt: pl.DataFrame, cfg: Dict,
                      prob: str = "p") -> Dict:
    final_th, final_ef, final_cal, final_lam, summary = {}, {}, {}, {}, {}
    per_method = {"thr": {0: [], 1: []}, "ef": {0: [], 1: []}, "exact": {0: [], 1: []}}
    pooled = []
    for c, t in tables.items():
        ent, g, sc = _country_eval_frames(ws, c, t, gt, prob)
        best = _with_y(decide.one_to_one(sc.filter(pl.col("p") >= 1e-3)), g)
        for k in (0, 1):
            tune_e = ent.filter(pl.col("fold") == k).select("s1")
            eval_e = ent.filter(pl.col("fold") != k).select("s1")
            b_t, g_t = best.join(tune_e, on="s1", how="semi"), g.join(tune_e, on="s1", how="semi")
            b_e, g_e = best.join(eval_e, on="s1", how="semi"), g.join(eval_e, on="s1", how="semi")
            sc_e = sc.join(eval_e, on="s1", how="semi")
            th, _ = decide.tune_thresholds(b_t, tune_e, g_t)
            r = metrics.report(eval_e, g_e, decide.apply_thresholds(b_e, th, 0.5), sc_e,
                               title=f"{c} thresholds tuned on fold {k} -> fold {1-k} {th}")
            per_method["thr"][k].append((len(eval_e), r))
            ef, _ = decide.tune_expected_f(b_t, tune_e, g_t)
            r = metrics.report(eval_e, g_e, decide.expected_f_select(b_e, **ef), sc_e,
                               title=f"{c} expected-F tuned on fold {k} -> fold {1-k} {ef}")
            per_method["ef"][k].append((len(eval_e), r))
            cal = decide.fit_isotonic(b_t["p"].to_numpy(), b_t["y"].to_numpy())
            lam, _ = _tune_exact(b_t, tune_e, g_t, cal)
            be = b_e.with_columns(pl.Series("q", decide.apply_calibration(b_e["p"].to_numpy(), cal)))
            r = metrics.report(eval_e, g_e, decide.exact_f_select(be, lam), sc_e,
                               title=f"{c} calibrated exact expected-F tuned on fold {k} -> fold {1-k} lam={lam}")
            per_method["exact"][k].append((len(eval_e), r))
        final_th[c], _ = decide.tune_thresholds(best, ent.select("s1"), g)
        final_ef[c], _ = decide.tune_expected_f(best, ent.select("s1"), g)
        final_cal[c] = decide.fit_isotonic(best["p"].to_numpy(), best["y"].to_numpy())
        final_lam[c], _ = _tune_exact(best, ent.select("s1"), g, final_cal[c])
        pooled.append(best.select("p", "y"))
    heldout = {}
    for meth, folds in per_method.items():
        vals = []
        for k in (0, 1):
            n = sum(x[0] for x in folds[k])

            def wavg(key):
                return sum(x[0] * x[1][key] for x in folds[k]) / max(n, 1)
            vals.append(wavg("f05"))
            _log(f"HELD-OUT [{meth}] fold {1-k}: macro F0.5 = {wavg('f05'):.4f}  singletons "
                 f"{wavg('f05_singletons'):.4f}  non-singletons {wavg('f05_nonsingletons'):.4f}  "
                 f"pair P {wavg('pair_precision'):.4f} R {wavg('pair_recall'):.4f}  "
                 f"blocking entity recall {wavg('block_entity_recall'):.4f}")
        heldout[meth] = float(np.mean(vals))
        summary[f"heldout_{meth}"] = heldout[meth]
    method = max(heldout, key=heldout.get)
    margin = cfg["unseen_country_margin"]
    default_th = {src: min(0.99, max(th[src] for th in final_th.values() if src in th) + margin)
                  for src in ("S2", "S3")}
    default_ef = {"gamma": max(e["gamma"] for e in final_ef.values()),
                  "missed": max(e["missed"] for e in final_ef.values()),
                  "floor": min(0.95, max(e["floor"] for e in final_ef.values()) + margin)}
    pooled = pl.concat(pooled)
    final_cal["__pooled__"] = decide.fit_isotonic(pooled["p"].to_numpy(), pooled["y"].to_numpy())
    un = dict(cfg.get("unlabelled") or {})
    unlabelled = {"temperature": float(un.get("temperature") or 1.0),
                  "lam": float(un["lam"]) if un.get("lam") is not None else max(final_lam.values()),
                  "bucket_offsets": dict(un.get("bucket_offsets") or {})}
    rules = {"method": method, "thresholds": final_th, "default": default_th,
             "expected_f": final_ef, "default_expected_f": default_ef,
             "calibration": final_cal, "exact_lam": final_lam, "unlabelled": unlabelled}
    with open(os.path.join(ws.model, "thresholds.json"), "w") as f:
        json.dump(rules, f, indent=1)
    _log(f"decision method: {method} (held-out {heldout}); thresholds {final_th}; expected-F {final_ef}")
    summary["method"] = method
    return summary


def decide_country(best: pl.DataFrame, country: str, rules: Dict,
                   buckets: Optional[pl.Series] = None) -> pl.DataFrame:
    """
    Apply the saved decision rule of a country. A country without training labels (an open set:
    decided by the absence of its string among the labelled ones) uses the pooled calibration, the
    unlabelled temperature / blocking-miss term and, if configured, per-shape logit offsets.
    """
    if rules["method"] == "exact":
        labelled = country in rules["exact_lam"]
        un = rules.get("unlabelled", {})
        cal = rules["calibration"].get(country) if labelled else rules["calibration"]["__pooled__"]
        q = decide.apply_calibration(best["p"].to_numpy(), cal, 1.0 if labelled else un.get("temperature", 1.0))
        if not labelled and buckets is not None and un.get("bucket_offsets"):
            q = decide.apply_bucket_offsets(q, buckets, un["bucket_offsets"])
        lam = rules["exact_lam"][country] if labelled else un.get("lam", 0.0)
        return decide.exact_f_select(best.with_columns(pl.Series("q", q)), lam).drop("q")
    if rules["method"] == "ef":
        params = rules["expected_f"].get(country, rules["default_expected_f"])
        return decide.expected_f_select(best, **params)
    th = rules["thresholds"].get(country, rules["default"])
    return decide.apply_thresholds(best, th, max(th.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: predict + write outputs
# ─────────────────────────────────────────────────────────────────────────────

def stage_predict(ws: Workspace, output_dir: str, cfg: Dict):
    ens1 = model.FoldEnsemble.load(os.path.join(ws.model, "stage1.pkl"))
    ens2 = model.FoldEnsemble.load(os.path.join(ws.model, "stage2.pkl"))
    with open(os.path.join(ws.model, "thresholds.json")) as f:
        rules = json.load(f)
    cand_parts, pred_parts = [], []
    for c in cfg.get("predict_countries") or countries(ws, "test"):
        cd = CountryData(ws, "test", c)
        _log(f"test/{c}: {len(cd):,} pairs")
        p1 = _saved_test_p1(ws, cd) if cfg.get("reuse_stage1") else None
        if p1 is not None:
            _log(f"  reusing saved stage-1 scores for test/{c}")
            p, _ = score_stage2(cd, ens2, p1, oof=False)
        elif len(cd):
            p1 = score_stage1(cd, ens1, oof=False)
            p, _ = score_stage2(cd, ens2, p1, oof=False)
        else:
            p1 = p = np.zeros(0, dtype=np.float32)
        t = pl.DataFrame({"i1": cd.cands["i1"], "j": cd.cands["j"], "rank": cd.cands["rank"],
                          "is_s2": cd.is_s2, "p1": p1, "p": p})
        t.write_parquet(ws.scores("test", c))
        t = t.with_columns(cd.s1ids.gather(t["i1"]).alias("s1"), cd.s23ids.gather(t["j"]).alias("eid"),
                           pl.when(pl.col("is_s2") == 1).then(pl.lit("S2")).otherwise(pl.lit("S3")).alias("src"))
        labelled = c in rules["thresholds"]
        if not labelled:
            _log(f"  {c}: no training labels -> unlabelled-country decision path")
        # same candidate set as the decision tuning in evaluate_and_tune (pairs with p >= 1e-3)
        best = decide.one_to_one(t.filter(pl.col("p") >= 1e-3).select("s1", "eid", "src", "p", "i1", "j"))
        buckets = None
        if not labelled and rules.get("unlabelled", {}).get("bucket_offsets") and len(best):
            buckets = decide.shape_buckets(cd.fz.s1["a_house"].gather(best["i1"]), cd.fz.s23["a_house"].gather(best["j"]),
                                           cd.fz.s1["n_core"].gather(best["i1"]), cd.fz.s23["n_core"].gather(best["j"]),
                                           features.SIBLING_OFFSETS)
            _log(f"  {c}: shape buckets {dict(zip(*np.unique(buckets.to_numpy(), return_counts=True)))}")
        pred = decide_country(best, c, rules, buckets).select("s1", "eid", "p")
        _log(f"  {c}: {len(pred):,} matches for {pred['s1'].n_unique():,} of {len(cd.s1ids):,} entities")
        pred_parts.append(pred)
        cand_parts.append(t.select("s1", "eid", "rank"))
        del cd
        gc.collect()
    write_outputs(ws, output_dir, pl.concat(cand_parts), pl.concat(pred_parts))


def _saved_test_p1(ws: Workspace, cd: CountryData) -> Optional[np.ndarray]:
    """Stage-1 test scores from an earlier predict run, if they belong to exactly these candidates."""
    path = ws.scores("test", cd.country)
    if not os.path.isfile(path) or not len(cd):
        return None
    saved = pl.read_parquet(path, columns=["i1", "j", "p1"])
    if len(saved) != len(cd) or not (saved["i1"].equals(cd.cands["i1"]) and saved["j"].equals(cd.cands["j"])):
        return None
    return saved["p1"].to_numpy()


def write_outputs(ws: Workspace, output_dir: str, cands: pl.DataFrame, pred: pl.DataFrame):
    os.makedirs(output_dir, exist_ok=True)
    s1 = pl.read_parquet(ws.norm("test", 1), columns=["entity_id"]).rename({"entity_id": "s1"})
    cand_lists = cands.sort(["s1", "rank"]).group_by("s1", maintain_order=True).agg(
        pl.col("eid").unique(maintain_order=True).str.join(",").alias("ids"))
    pred_lists = pred.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(
        pl.col("eid").unique(maintain_order=True).str.join(",").alias("ids"))
    for name, header, lists in (("candidate_pairs.tsv", "candidate_entity_ids", cand_lists),
                                ("matching_results.tsv", "matched_entity_ids", pred_lists)):
        out = s1.join(lists, on="s1", how="left", maintain_order="left").with_columns(pl.col("ids").fill_null(""))
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            for sid, ids in out.iter_rows():
                fh.write(f"{sid}\t{ids}\n")
        _log(f"wrote {path}: {len(out):,} rows, {int((out['ids'] != '').sum()):,} non-empty")

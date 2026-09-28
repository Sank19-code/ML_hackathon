"""
Pairwise features for (Source 1, candidate) pairs, computed in vectorised chunks.

String similarities use rapidfuzz.process.cpdist (C++, multithreaded) on the canonical
fields produced by normalize.py; set/number comparisons use polars list operations;
IDF-weighted overlaps use sparse idf rows and a numba kernel (cosine plus
containment in both directions, so rare shared words count for more).

Feature groups
  name   – on the raw name, the full cleaned name (keeps "services", "trading", legal
           forms) and the core name (legal forms, honorifics and generator filler words
           removed): ratio, token sort / set, partial, Jaro-Winkler, compact-string ratios
           (domain names), alias comparison, IDF cosine / containment of name tokens,
           char 3-gram TF-IDF cosine.
  address– full / street / locality similarities; separate match and conflict flags for
           house number, number set, compound id (D-12), locality (city) and state; IDF
           cosine / containment of address tokens.
  record – source (S2/S3), domain / transliterated / alias flags, token counts,
           legal-form agreement, missing-address flags.
  block  – blocking cosines (name / address / char), cheap score, rank and number of
           candidates of the Source 1 record.
"""
import time
from typing import Dict

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

try:
    from .blocking import build_tfidf, char_features, pair_stats, word_features, NS_CODES
except ImportError:
    from blocking import build_tfidf, char_features, pair_stats, word_features, NS_CODES

# (column, scorer, feature name)
STRING_FEATURES = [
    ("n_core", fuzz.ratio, "core_ratio"),
    ("n_core", fuzz.token_sort_ratio, "core_tsort"),
    ("n_core", fuzz.token_set_ratio, "core_tset"),
    ("n_core", fuzz.partial_ratio, "core_partial"),
    ("n_core", JaroWinkler.normalized_similarity, "core_jw"),
    ("n_cmp", fuzz.ratio, "cmp_ratio"),
    ("n_cmp", fuzz.partial_ratio, "cmp_partial"),
    ("n_fc", fuzz.ratio, "fc_ratio"),
    ("n_full", fuzz.ratio, "full_ratio"),
    ("n_full", fuzz.token_set_ratio, "full_tset"),
    ("n_full", fuzz.token_sort_ratio, "full_tsort"),
    ("name_lc", fuzz.ratio, "raw_ratio"),
    ("name_lc", fuzz.token_set_ratio, "raw_tset"),
    ("a_norm", fuzz.ratio, "addr_ratio"),
    ("a_norm", fuzz.token_set_ratio, "addr_tset"),
    ("a_norm", fuzz.token_sort_ratio, "addr_tsort"),
    ("a_street", fuzz.ratio, "street_ratio"),
    ("a_street", fuzz.token_set_ratio, "street_tset"),
    ("a_loc", fuzz.token_set_ratio, "loc_tset"),
    ("a_loc", fuzz.ratio, "loc_ratio"),
    ("a_house", fuzz.ratio, "house_ratio"),
]
# cross comparisons: (s1 column, candidate column, scorer, name)
CROSS_FEATURES = [
    ("n_cmp", "n_fc", fuzz.partial_ratio, "cmp_fc_partial"),  # domain incl. legal suffix
    ("n_fc", "n_cmp", fuzz.partial_ratio, "fc_cmp_partial"),
    ("n_core", "n_alias", fuzz.token_set_ratio, "core_alias_tset"),
]
BLOCK_FEATURES = ["cos_name", "cos_addr", "cos_char", "cheap", "rank", "n_cands"]
IDF_FEATURES = [
    "name_cos", "name_cont_ab", "name_cont_ba", "name_nf_a", "name_nf_b",
    "addr_cos", "addr_cont_ab", "addr_cont_ba", "addr_nf_a", "addr_nf_b",
    "char_cos", "char_cont_ab", "char_cont_ba",
]
SET_FEATURES = [
    "house_in_b", "house_b_in_a", "house_eq", "num_inter", "num_a", "num_b", "num_b_extra",
    "ids_inter", "ids_conflict", "state_match", "loc_match",
    "ntok_a", "ntok_b", "legal_eq", "legal_empty_b", "first_tok_eq", "last_tok_eq",
<<<<<<< HEAD
    "house_delta", "house_delta_sib", "nums_sib", "nums_sib_any", "house_trunc",
=======
    "house_delta", "house_delta_sib", "nums_sib", "nums_sib_any",
>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10
]
# The generator creates "sibling" distractor businesses: same name and street, house number =
# the entity's number + one of these offsets (99.9 % of the negative near-duplicates with a
# small number difference in training have a positive offset from this set, while genuine
# house-number noise is symmetric). Learned from the training ground truth.
SIBLING_OFFSETS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
RECORD_FEATURES = ["a_empty_b", "street_empty_b", "is_s2", "is_domain_b", "is_native_b",
                   "has_alias_b", "len_a", "len_b", "acr_prefix", "acr_rest"]
<<<<<<< HEAD
# content core: the core name minus the locality words of either record ("Tourcoing Ecole SARL" vs
# "Tourcoing Amis SARL" -> "ecole" vs "amis"); separates a neighbouring business that shares the
# town-derived part of the name from a noisy copy. Record-driven, no word lists.
CORE_FEATURES = ["cc_jacc", "cc_disjoint", "cc_diff_idf"]
FEATURE_NAMES = ([f[2] for f in STRING_FEATURES] + [f[3] for f in CROSS_FEATURES] + BLOCK_FEATURES
                 + IDF_FEATURES + SET_FEATURES + RECORD_FEATURES + CORE_FEATURES)
=======
FEATURE_NAMES = ([f[2] for f in STRING_FEATURES] + [f[3] for f in CROSS_FEATURES] + BLOCK_FEATURES
                 + IDF_FEATURES + SET_FEATURES + RECORD_FEATURES)
>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10

NEEDED_COLUMNS = ["entity_id", "business_name", "n_core", "n_cmp", "n_full", "n_alias", "n_legal",
                  "a_norm", "a_street", "a_loc", "a_house", "a_nums", "a_ids", "a_state", "a_empty",
                  "is_domain", "is_native", "has_alias"]


def prepare_side(df: pl.DataFrame) -> pl.DataFrame:
    """Derived string columns used by the features; keep only what is needed."""
    return df.select(NEEDED_COLUMNS).with_columns(
        pl.col("business_name").str.to_lowercase().alias("name_lc"),
        pl.col("n_full").str.replace_all(" ", "").alias("n_fc"),
    ).drop("business_name")


<<<<<<< HEAD
def impute_state(s1: pl.DataFrame, s23: pl.DataFrame, min_share: float = 0.9, min_support: int = 5):
    """
    Fill a missing state / region from the record's locality, using a locality -> state table learned
    from the country's own records that carry both (no gazetteer). The noise drops the last address
    component far more often in some countries (a third of the French Source 2/3 records end with the
    city), which would otherwise turn state_match into 'unknown' for them.
    """
    both = pl.concat([s1.select("a_loc", "a_state"), s23.select("a_loc", "a_state")]).filter(
        (pl.col("a_loc") != "") & (pl.col("a_state") != ""))
    if len(both) == 0:
        return s1, s23
    tab = (both.group_by(["a_loc", "a_state"]).agg(pl.len().alias("n"))
           .with_columns(pl.col("n").sum().over("a_loc").alias("tot"))
           .filter((pl.col("n") == pl.col("n").max().over("a_loc")) & (pl.col("tot") >= min_support)
                   & (pl.col("n") >= min_share * pl.col("tot")))
           .unique("a_loc").select("a_loc", pl.col("a_state").alias("_st")))

    def fill(df):
        return (df.join(tab, on="a_loc", how="left", maintain_order="left")
                .with_columns(pl.when(pl.col("a_state") == "").then(pl.col("_st").fill_null(""))
                              .otherwise(pl.col("a_state")).alias("a_state")).drop("_st"))
    return fill(s1), fill(s23)


class PairFeaturizer:
    """Holds per-country frames and idf matrices so that pair chunks featurise cheaply."""

    def __init__(self, s1: pl.DataFrame, s23: pl.DataFrame):
        n1, n2 = len(s1), len(s23)
        f1, f2 = word_features(s1), word_features(s23)
        name_ns = [NS_CODES["w"]]
        addr_ns = [NS_CODES[k] for k in ("s", "l", "n", "x")]
        big = 10 ** 12
        self.An, self.Bn = build_tfidf(f1.filter(pl.col("ns").is_in(name_ns)),
                                       f2.filter(pl.col("ns").is_in(name_ns)),
                                       n1, n2, {"w": 1.0}, big, 1, normalize=False)
        self.Aa, self.Ba = build_tfidf(f1.filter(pl.col("ns").is_in(addr_ns)),
                                       f2.filter(pl.col("ns").is_in(addr_ns)),
                                       n1, n2, {"s": 1.0, "l": 1.0, "n": 1.0, "x": 1.0}, big, 1,
                                       normalize=False)
        del f1, f2
        self.Ac, self.Bc = build_tfidf(char_features(s1), char_features(s23), n1, n2, {"g": 1.0}, big, 1,
                                       normalize=False)
        self.s1 = prepare_side(s1)
        self.s23 = prepare_side(s23)
        self.s1, self.s23 = impute_state(self.s1, self.s23)
        # name-token idf of the country (both sides) for the content-core features
        toks = pl.concat([self.s1.select("n_core"), self.s23.select("n_core")]).with_row_index("r").with_columns(
            pl.col("n_core").str.split(" ").list.unique()).explode("n_core").filter(pl.col("n_core") != "")
        n_rec = n1 + n2
        self.idf = (toks.group_by("n_core").agg(pl.len().alias("df"))
                    .select(pl.col("n_core").alias("tok"), (np.log(n_rec) - pl.col("df").log()).cast(pl.Float32).alias("idf")))
        del toks

    def content_core(self, I: np.ndarray, J: np.ndarray) -> Dict[str, np.ndarray]:
        """cc_jacc / cc_disjoint / cc_diff_idf for pairs (I[i], J[i]); -1 when a core is empty."""
        split = lambda s: s.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
        a, b = self.s1[I], self.s23[J]
        df = pl.DataFrame({"ta": split(a["n_core"]), "tb": split(b["n_core"]),
                           "la": split(a["a_loc"]), "lb": split(b["a_loc"])})
        df = df.select(pl.col("ta").list.set_difference(pl.col("la")).list.set_difference(pl.col("lb")).alias("ca"),
                       pl.col("tb").list.set_difference(pl.col("la")).list.set_difference(pl.col("lb")).alias("cb"))
        df = df.with_columns(pl.col("ca").list.set_intersection("cb").list.len().alias("ni"),
                             pl.col("ca").list.set_union("cb").alias("u"),
                             pl.col("ca").list.set_symmetric_difference("cb").alias("d"),
                             ((pl.col("ca").list.len() == 0) | (pl.col("cb").list.len() == 0)).alias("empty"))
        nu = df["u"].list.len().to_numpy()
        ni = df["ni"].to_numpy()
        empty = df["empty"].to_numpy()

        def idf_sum(col):
            ex = df.select(col).with_row_index("r").explode(col).filter(pl.col(col).is_not_null())
            s = ex.join(self.idf, left_on=col, right_on="tok", how="left").group_by("r").agg(
                pl.col("idf").fill_null(float(np.log(len(self.s1) + len(self.s23)))).sum())
            out = np.zeros(len(df), dtype=np.float32)
            out[s["r"].to_numpy()] = s["idf"].to_numpy()
            return out

        su, sd = idf_sum("u"), idf_sum("d")
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(empty, -1.0, ni / np.maximum(nu, 1)).astype(np.float32)
            dis = np.where(empty, -1.0, (ni == 0).astype(np.float32)).astype(np.float32)
            dif = np.where(empty | (su <= 0), -1.0, sd / np.where(su > 0, su, 1.0)).astype(np.float32)
        return {"cc_jacc": jac, "cc_disjoint": dis, "cc_diff_idf": dif}

    def featurize(self, pairs: pl.DataFrame) -> np.ndarray:
        """pairs: frame with i1, j and BLOCK_FEATURES columns. Returns float32 matrix."""
        I = pairs["i1"].to_numpy().astype(np.int64)
        J = pairs["j"].to_numpy().astype(np.int64)
        a = self.s1[I]
        b = self.s23[J]
        cols: Dict[str, np.ndarray] = {}
        empty_cache: Dict[str, np.ndarray] = {}

        def empty(side, frame, col):
            key = side + col
            if key not in empty_cache:
                empty_cache[key] = (frame[col].str.len_chars() == 0).to_numpy()
            return empty_cache[key]

        lists: Dict[str, list] = {}

        def as_list(side, frame, col):
            key = side + col
            if key not in lists:
                lists[key] = frame[col].to_list()
            return lists[key]

        for col, scorer, name in STRING_FEATURES:
            v = process.cpdist(as_list("a", a, col), as_list("b", b, col), scorer=scorer, workers=-1,
                               dtype=np.float32)
            if scorer is JaroWinkler.normalized_similarity:
                v = v * 100.0
            v[empty("a", a, col) | empty("b", b, col)] = -1.0
            cols[name] = v
        for ca, cb, scorer, name in CROSS_FEATURES:
            v = process.cpdist(as_list("a", a, ca), as_list("b", b, cb), scorer=scorer, workers=-1,
                               dtype=np.float32)
            v[empty("a", a, ca) | empty("b", b, cb)] = -1.0
            cols[name] = v
        for c in BLOCK_FEATURES:
            cols[c] = pairs[c].to_numpy().astype(np.float32)
        for pre, A, B in (("name", self.An, self.Bn), ("addr", self.Aa, self.Ba)):
            st = pair_stats(A, B, I, J)
            cols[pre + "_cos"], cols[pre + "_cont_ab"], cols[pre + "_cont_ba"] = st[:, 0], st[:, 1], st[:, 2]
            cols[pre + "_nf_a"], cols[pre + "_nf_b"] = st[:, 3], st[:, 4]
        st = pair_stats(self.Ac, self.Bc, I, J)
        cols["char_cos"], cols["char_cont_ab"], cols["char_cont_ba"] = st[:, 0], st[:, 1], st[:, 2]

        split = lambda s: s.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        sets = pl.DataFrame({
            "ha": a["a_house"], "hb": b["a_house"], "na": split(a["a_nums"]), "nb": split(b["a_nums"]),
            "xa": split(a["a_ids"]), "xb": split(b["a_ids"]), "sa": a["a_state"], "sb": b["a_state"],
            "la": split(a["a_loc"]), "lb": split(b["a_loc"]),
            "ta": a["n_core"].str.split(" "), "tb": b["n_core"].str.split(" "),
            "ga": a["n_legal"], "gb": b["n_legal"],
        })
        f = sets.select(
            # 1 = match, 0 = conflict, -1 = unknown (one side missing)
            pl.when((pl.col("ha") == "") | (pl.col("nb").list.len() == 0)).then(-1)
            .when(pl.col("nb").list.contains(pl.col("ha"))).then(1).otherwise(0).alias("house_in_b"),
            pl.when((pl.col("hb") == "") | (pl.col("na").list.len() == 0)).then(-1)
            .when(pl.col("na").list.contains(pl.col("hb"))).then(1).otherwise(0).alias("house_b_in_a"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(-1)
            .when(pl.col("ha") == pl.col("hb")).then(1).otherwise(0).alias("house_eq"),
            pl.col("na").list.set_intersection("nb").list.len().alias("num_inter"),
            pl.col("na").list.len().alias("num_a"),
            pl.col("nb").list.len().alias("num_b"),
            pl.col("nb").list.set_difference("na").list.len().alias("num_b_extra"),
            pl.col("xa").list.set_intersection("xb").list.len().alias("ids_inter"),
            ((pl.col("xa").list.len() > 0) & (pl.col("xb").list.len() > 0)
             & (pl.col("xa").list.set_intersection("xb").list.len() == 0)).cast(pl.Int8).alias("ids_conflict"),
            pl.when((pl.col("sa") == "") | (pl.col("sb") == "")).then(-1)
            .when(pl.col("sa") == pl.col("sb")).then(1).otherwise(0).alias("state_match"),
            pl.when((pl.col("la").list.len() == 0) | (pl.col("lb").list.len() == 0)).then(-1)
            .when(pl.col("la").list.set_intersection("lb").list.len() > 0).then(1).otherwise(0).alias("loc_match"),
            pl.col("ta").list.len().alias("ntok_a"),
            pl.col("tb").list.len().alias("ntok_b"),
            pl.when((pl.col("ga") == "") | (pl.col("gb") == "")).then(-1)
            .when(pl.col("ga") == pl.col("gb")).then(1).otherwise(0).alias("legal_eq"),
            (pl.col("gb") == "").cast(pl.Int8).alias("legal_empty_b"),
            (pl.col("ta").list.first() == pl.col("tb").list.first()).cast(pl.Int8).alias("first_tok_eq"),
            (pl.col("ta").list.last() == pl.col("tb").list.last()).cast(pl.Int8).alias("last_tok_eq"),
            # signed house-number difference (candidate - Source 1) and the sibling-offset flags
            (pl.col("hb").cast(pl.Int64, strict=False) - pl.col("ha").cast(pl.Int64, strict=False))
            .clip(-100000, 100000).cast(pl.Float32).alias("house_delta"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(-1)
            .when((pl.col("hb").cast(pl.Int64, strict=False) - pl.col("ha").cast(pl.Int64, strict=False))
                  .is_in(SIBLING_OFFSETS)).then(1).otherwise(0).alias("house_delta_sib"),
            # clipped house number: one side lost its first or last digit (2424 -> 424, 17177 -> 1717)
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(-1)
            .when((pl.col("ha") != pl.col("hb"))
                  & ((pl.col("hb") == pl.col("ha").str.slice(1)) | (pl.col("hb") == pl.col("ha").str.head(-1))
                     | (pl.col("ha") == pl.col("hb").str.slice(1)) | (pl.col("ha") == pl.col("hb").str.head(-1))))
            .then(1).otherwise(0).alias("house_trunc"),
            pl.when((pl.col("ha") == "") | (pl.col("nb").list.len() == 0)).then(-1)
            .when(~pl.col("nb").list.contains(pl.col("ha"))
                  & (pl.col("nb").list.set_intersection(pl.concat_list(
                      [(pl.col("ha").cast(pl.Int64, strict=False) + k).cast(pl.String) for k in SIBLING_OFFSETS]))
                     .list.len() > 0)).then(1).otherwise(0).alias("nums_sib"),
        )
        for c in f.columns:
            cols[c] = f[c].to_numpy().astype(np.float32)
        cols["nums_sib_any"] = _sibling_any(sets.select("na", "nb"))
        cols["a_empty_b"] = b["a_empty"].to_numpy().astype(np.float32)
        cols["street_empty_b"] = empty("b", b, "a_street").astype(np.float32)
        cols["is_s2"] = b["entity_id"].str.starts_with("S2-").to_numpy().astype(np.float32)
        cols["is_domain_b"] = b["is_domain"].to_numpy().astype(np.float32)
        cols["is_native_b"] = b["is_native"].to_numpy().astype(np.float32)
        cols["has_alias_b"] = b["has_alias"].to_numpy().astype(np.float32)
        cols["len_a"] = a["n_core"].str.len_chars().to_numpy().astype(np.float32)
        cols["len_b"] = b["n_core"].str.len_chars().to_numpy().astype(np.float32)
        cols["acr_prefix"], cols["acr_rest"] = _acronym_features(a, b, as_list)
        cols.update(self.content_core(I, J))
        out = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
        for k, n in enumerate(FEATURE_NAMES):
            out[:, k] = cols[n]
        return out


def _sibling_any(nums: pl.DataFrame) -> np.ndarray:
    """
    1 if some candidate number x is not among the Source 1 numbers but x - k is, for a sibling
    offset k (covers compound numbers such as 116-47 -> 116-49 or 22/235 -> 22/236), 0 if not,
    -1 when either side has no number.
    """
    out = np.where((nums["na"].list.len() == 0) | (nums["nb"].list.len() == 0), -1.0, 0.0).astype(np.float32)
    ex = (nums.with_columns(pl.int_range(0, pl.len()).alias("r")).explode("nb")
          .filter(pl.col("nb").is_not_null() & ~pl.col("na").list.contains(pl.col("nb"))))
    if len(ex):
        x = pl.col("nb").cast(pl.Int64, strict=False)
        hit = ex.filter(pl.col("na").list.set_intersection(
            pl.concat_list([(x - k).cast(pl.String) for k in SIBLING_OFFSETS])).list.len() > 0)["r"].unique().to_numpy()
        out[hit] = 1.0
    return out


def _acronym_features(a: pl.DataFrame, b: pl.DataFrame, as_list):
    """
    Domain / glued names built from initials ("jiprivate.com" for Jai Infrastructure Private,
    "rpcartons.com" for Rajni ... Cartons): length of the common prefix between the
    candidate's glued name and the Source 1 initials, and how well the rest of the glued
    name matches the Source 1 name. -1 when the candidate is not a single glued token.
    """
    n = len(a)
    pref = np.full(n, -1.0, dtype=np.float32)
    rest = np.full(n, -1.0, dtype=np.float32)
    single = (b["n_full"].str.contains(" ").not_() & (b["n_cmp"].str.len_chars() >= 4)).to_numpy()
    idx = np.where(single)[0]
    if len(idx) == 0:
        return pref, rest
    full_a = as_list("a", a, "n_full")
    fc_a = as_list("a", a, "n_fc")
    cmp_b = as_list("b", b, "n_cmp")
    rests_a, rests_b = [], []
    for i in idx:
        stem = cmp_b[i]
        initials = "".join(t[0] for t in full_a[i].split() if t)
        k = 0
        while k < len(initials) and k < len(stem) and initials[k] == stem[k]:
            k += 1
        pref[i] = k
        rests_a.append(fc_a[i])
        rests_b.append(stem[k:] if k >= 2 else stem)
    rest[idx] = process.cpdist(rests_b, rests_a, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)
    return pref, rest


=======
class PairFeaturizer:
    """Holds per-country frames and idf matrices so that pair chunks featurise cheaply."""

    def __init__(self, s1: pl.DataFrame, s23: pl.DataFrame):
        n1, n2 = len(s1), len(s23)
        f1, f2 = word_features(s1), word_features(s23)
        name_ns = [NS_CODES["w"]]
        addr_ns = [NS_CODES[k] for k in ("s", "l", "n", "x")]
        big = 10 ** 12
        self.An, self.Bn = build_tfidf(f1.filter(pl.col("ns").is_in(name_ns)),
                                       f2.filter(pl.col("ns").is_in(name_ns)),
                                       n1, n2, {"w": 1.0}, big, 1, normalize=False)
        self.Aa, self.Ba = build_tfidf(f1.filter(pl.col("ns").is_in(addr_ns)),
                                       f2.filter(pl.col("ns").is_in(addr_ns)),
                                       n1, n2, {"s": 1.0, "l": 1.0, "n": 1.0, "x": 1.0}, big, 1,
                                       normalize=False)
        del f1, f2
        self.Ac, self.Bc = build_tfidf(char_features(s1), char_features(s23), n1, n2, {"g": 1.0}, big, 1,
                                       normalize=False)
        self.s1 = prepare_side(s1)
        self.s23 = prepare_side(s23)

    def featurize(self, pairs: pl.DataFrame) -> np.ndarray:
        """pairs: frame with i1, j and BLOCK_FEATURES columns. Returns float32 matrix."""
        I = pairs["i1"].to_numpy().astype(np.int64)
        J = pairs["j"].to_numpy().astype(np.int64)
        a = self.s1[I]
        b = self.s23[J]
        cols: Dict[str, np.ndarray] = {}
        empty_cache: Dict[str, np.ndarray] = {}

        def empty(side, frame, col):
            key = side + col
            if key not in empty_cache:
                empty_cache[key] = (frame[col].str.len_chars() == 0).to_numpy()
            return empty_cache[key]

        lists: Dict[str, list] = {}

        def as_list(side, frame, col):
            key = side + col
            if key not in lists:
                lists[key] = frame[col].to_list()
            return lists[key]

        for col, scorer, name in STRING_FEATURES:
            v = process.cpdist(as_list("a", a, col), as_list("b", b, col), scorer=scorer, workers=-1,
                               dtype=np.float32)
            if scorer is JaroWinkler.normalized_similarity:
                v = v * 100.0
            v[empty("a", a, col) | empty("b", b, col)] = -1.0
            cols[name] = v
        for ca, cb, scorer, name in CROSS_FEATURES:
            v = process.cpdist(as_list("a", a, ca), as_list("b", b, cb), scorer=scorer, workers=-1,
                               dtype=np.float32)
            v[empty("a", a, ca) | empty("b", b, cb)] = -1.0
            cols[name] = v
        for c in BLOCK_FEATURES:
            cols[c] = pairs[c].to_numpy().astype(np.float32)
        for pre, A, B in (("name", self.An, self.Bn), ("addr", self.Aa, self.Ba)):
            st = pair_stats(A, B, I, J)
            cols[pre + "_cos"], cols[pre + "_cont_ab"], cols[pre + "_cont_ba"] = st[:, 0], st[:, 1], st[:, 2]
            cols[pre + "_nf_a"], cols[pre + "_nf_b"] = st[:, 3], st[:, 4]
        st = pair_stats(self.Ac, self.Bc, I, J)
        cols["char_cos"], cols["char_cont_ab"], cols["char_cont_ba"] = st[:, 0], st[:, 1], st[:, 2]

        split = lambda s: s.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        sets = pl.DataFrame({
            "ha": a["a_house"], "hb": b["a_house"], "na": split(a["a_nums"]), "nb": split(b["a_nums"]),
            "xa": split(a["a_ids"]), "xb": split(b["a_ids"]), "sa": a["a_state"], "sb": b["a_state"],
            "la": split(a["a_loc"]), "lb": split(b["a_loc"]),
            "ta": a["n_core"].str.split(" "), "tb": b["n_core"].str.split(" "),
            "ga": a["n_legal"], "gb": b["n_legal"],
        })
        f = sets.select(
            # 1 = match, 0 = conflict, -1 = unknown (one side missing)
            pl.when((pl.col("ha") == "") | (pl.col("nb").list.len() == 0)).then(-1)
            .when(pl.col("nb").list.contains(pl.col("ha"))).then(1).otherwise(0).alias("house_in_b"),
            pl.when((pl.col("hb") == "") | (pl.col("na").list.len() == 0)).then(-1)
            .when(pl.col("na").list.contains(pl.col("hb"))).then(1).otherwise(0).alias("house_b_in_a"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(-1)
            .when(pl.col("ha") == pl.col("hb")).then(1).otherwise(0).alias("house_eq"),
            pl.col("na").list.set_intersection("nb").list.len().alias("num_inter"),
            pl.col("na").list.len().alias("num_a"),
            pl.col("nb").list.len().alias("num_b"),
            pl.col("nb").list.set_difference("na").list.len().alias("num_b_extra"),
            pl.col("xa").list.set_intersection("xb").list.len().alias("ids_inter"),
            ((pl.col("xa").list.len() > 0) & (pl.col("xb").list.len() > 0)
             & (pl.col("xa").list.set_intersection("xb").list.len() == 0)).cast(pl.Int8).alias("ids_conflict"),
            pl.when((pl.col("sa") == "") | (pl.col("sb") == "")).then(-1)
            .when(pl.col("sa") == pl.col("sb")).then(1).otherwise(0).alias("state_match"),
            pl.when((pl.col("la").list.len() == 0) | (pl.col("lb").list.len() == 0)).then(-1)
            .when(pl.col("la").list.set_intersection("lb").list.len() > 0).then(1).otherwise(0).alias("loc_match"),
            pl.col("ta").list.len().alias("ntok_a"),
            pl.col("tb").list.len().alias("ntok_b"),
            pl.when((pl.col("ga") == "") | (pl.col("gb") == "")).then(-1)
            .when(pl.col("ga") == pl.col("gb")).then(1).otherwise(0).alias("legal_eq"),
            (pl.col("gb") == "").cast(pl.Int8).alias("legal_empty_b"),
            (pl.col("ta").list.first() == pl.col("tb").list.first()).cast(pl.Int8).alias("first_tok_eq"),
            (pl.col("ta").list.last() == pl.col("tb").list.last()).cast(pl.Int8).alias("last_tok_eq"),
            # signed house-number difference (candidate - Source 1) and the sibling-offset flags
            (pl.col("hb").cast(pl.Int64, strict=False) - pl.col("ha").cast(pl.Int64, strict=False))
            .clip(-100000, 100000).cast(pl.Float32).alias("house_delta"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(-1)
            .when((pl.col("hb").cast(pl.Int64, strict=False) - pl.col("ha").cast(pl.Int64, strict=False))
                  .is_in(SIBLING_OFFSETS)).then(1).otherwise(0).alias("house_delta_sib"),
            pl.when((pl.col("ha") == "") | (pl.col("nb").list.len() == 0)).then(-1)
            .when(~pl.col("nb").list.contains(pl.col("ha"))
                  & (pl.col("nb").list.set_intersection(pl.concat_list(
                      [(pl.col("ha").cast(pl.Int64, strict=False) + k).cast(pl.String) for k in SIBLING_OFFSETS]))
                     .list.len() > 0)).then(1).otherwise(0).alias("nums_sib"),
        )
        for c in f.columns:
            cols[c] = f[c].to_numpy().astype(np.float32)
        cols["nums_sib_any"] = _sibling_any(sets.select("na", "nb"))
        cols["a_empty_b"] = b["a_empty"].to_numpy().astype(np.float32)
        cols["street_empty_b"] = empty("b", b, "a_street").astype(np.float32)
        cols["is_s2"] = b["entity_id"].str.starts_with("S2-").to_numpy().astype(np.float32)
        cols["is_domain_b"] = b["is_domain"].to_numpy().astype(np.float32)
        cols["is_native_b"] = b["is_native"].to_numpy().astype(np.float32)
        cols["has_alias_b"] = b["has_alias"].to_numpy().astype(np.float32)
        cols["len_a"] = a["n_core"].str.len_chars().to_numpy().astype(np.float32)
        cols["len_b"] = b["n_core"].str.len_chars().to_numpy().astype(np.float32)
        cols["acr_prefix"], cols["acr_rest"] = _acronym_features(a, b, as_list)
        return np.column_stack([cols[n] for n in FEATURE_NAMES]).astype(np.float32)


def _sibling_any(nums: pl.DataFrame) -> np.ndarray:
    """
    1 if some candidate number x is not among the Source 1 numbers but x - k is, for a sibling
    offset k (covers compound numbers such as 116-47 -> 116-49 or 22/235 -> 22/236), 0 if not,
    -1 when either side has no number.
    """
    out = np.where((nums["na"].list.len() == 0) | (nums["nb"].list.len() == 0), -1.0, 0.0).astype(np.float32)
    ex = (nums.with_columns(pl.int_range(0, pl.len()).alias("r")).explode("nb")
          .filter(pl.col("nb").is_not_null() & ~pl.col("na").list.contains(pl.col("nb"))))
    if len(ex):
        x = pl.col("nb").cast(pl.Int64, strict=False)
        hit = ex.filter(pl.col("na").list.set_intersection(
            pl.concat_list([(x - k).cast(pl.String) for k in SIBLING_OFFSETS])).list.len() > 0)["r"].unique().to_numpy()
        out[hit] = 1.0
    return out


def _acronym_features(a: pl.DataFrame, b: pl.DataFrame, as_list):
    """
    Domain / glued names built from initials ("jiprivate.com" for Jai Infrastructure Private,
    "rpcartons.com" for Rajni ... Cartons): length of the common prefix between the
    candidate's glued name and the Source 1 initials, and how well the rest of the glued
    name matches the Source 1 name. -1 when the candidate is not a single glued token.
    """
    n = len(a)
    pref = np.full(n, -1.0, dtype=np.float32)
    rest = np.full(n, -1.0, dtype=np.float32)
    single = (b["n_full"].str.contains(" ").not_() & (b["n_cmp"].str.len_chars() >= 4)).to_numpy()
    idx = np.where(single)[0]
    if len(idx) == 0:
        return pref, rest
    full_a = as_list("a", a, "n_full")
    fc_a = as_list("a", a, "n_fc")
    cmp_b = as_list("b", b, "n_cmp")
    rests_a, rests_b = [], []
    for i in idx:
        stem = cmp_b[i]
        initials = "".join(t[0] for t in full_a[i].split() if t)
        k = 0
        while k < len(initials) and k < len(stem) and initials[k] == stem[k]:
            k += 1
        pref[i] = k
        rests_a.append(fc_a[i])
        rests_b.append(stem[k:] if k >= 2 else stem)
    rest[idx] = process.cpdist(rests_b, rests_a, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)
    return pref, rest


>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10
def featurize_to_memmap(fz: PairFeaturizer, cands: pl.DataFrame, path: str, chunk: int = 1_000_000,
                        verbose: bool = True) -> np.memmap:
    """Featurise all candidate pairs into a float16 .npy memmap (rows aligned with cands)."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(len(cands), len(FEATURE_NAMES)))
    t0 = time.time()
    for start in range(0, len(cands), chunk):
        part = cands.slice(start, chunk)
        out[start:start + len(part)] = fz.featurize(part).astype(np.float16)
        if verbose:
            print(f"    features {start + len(part):,}/{len(cands):,} ({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    return out

"""Unit tests for the entity-resolution pipeline (run: python -m pytest test_pipeline.py or python test_pipeline.py)."""
import os
import sys
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from normalize import normalize_address, normalize_name, set_translit_dict  # noqa: E402
from blocking import generate_candidates, pair_dot, pair_stats, build_tfidf, word_features  # noqa: E402
from features import FEATURE_NAMES, PairFeaturizer  # noqa: E402
from metrics import compute_macro_f05, entity_scores  # noqa: E402
from decide import one_to_one, apply_thresholds, expected_f_select  # noqa: E402
from data import normalize_frame  # noqa: E402


def _frame(rows, prefix):
    return pl.DataFrame({
        "entity_id": [f"{prefix}-{i}" for i in range(len(rows))],
        "business_name": [r[0] for r in rows],
        "business_address": [r[1] for r in rows],
        "country": [r[2] for r in rows],
    })


class TestNormalize(unittest.TestCase):
    def test_name_noise_is_undone(self):
        self.assertEqual(normalize_name("Drexnexarc dba Highland Capital, LLC")[1], "highland capital")
        self.assertEqual(normalize_name("-- Holloway Peak Inc")[1], "holloway peak")
        self.assertEqual(normalize_name("Kinley Pediatrics L.L.C.")[0], "kinley pediatrics llc")
        self.assertEqual(normalize_name("Integrated Trading Ass0ciates")[1], "integrated trading associates")
        self.assertEqual(normalize_name("M/s LAXMI INVESTMENT PVT LTD.")[1], "laxmi investment")
        full, core, _, _, cmp_, is_domain, _, _ = normalize_name("wilfordhancock.com")
        self.assertEqual((cmp_, is_domain), ("wilfordhancock", 1))
        self.assertEqual(normalize_name("PLANTATION CENTER PRIVATE LÍMITED (ID: 93609)")[0],
                         "plantation center pvt ltd")

    def test_transliteration(self):
        set_translit_dict({"लक्ष्मी": "laxmi", "इन्वेस्टमेंट": "investment", "प्रा": "pvt", "लि": "ltd"})
        self.assertEqual(normalize_name("लक्ष्मी इन्वेस्टमेंट प्रा. लि.")[1], "laxmi investment")
        # unknown words fall back to anyascii instead of disappearing
        self.assertTrue(normalize_name("सन")[1])

    def test_address(self):
        norm, street, loc, state, nums, house, empty, ids = normalize_address(
            "#111 Mansfield Ave, Shelby, Ohio", "US")
        self.assertEqual((street, loc, state, house), ("mansfield ave", "shelby", "oh", "111"))
        self.assertEqual(normalize_address("31, Mumbai City, महाराष्ट्र", "India")[3], "maharashtra")
        self.assertEqual(normalize_address("TN", "India")[3], "tamil nadu")
        self.assertEqual(normalize_address("12 R. Lyderic, Lille, Nord", "France")[3], "hauts de france")
        self.assertEqual(normalize_address("D-12, Kalkaji, New Delhi, Delhi", "India")[7], "d12")
        self.assertEqual(normalize_address("", "US")[6], 1)


class TestBlockingAndFeatures(unittest.TestCase):
    def setUp(self):
        s1 = _frame([("Robbie Garza Laboratories", "61 Scarborough Village Drive, Dayton, OH", "US"),
                     ("Beam Boral Co", "111 Mansfield Avenue, Shelby, OH", "US"),
                     ("Primary Care Group", "49 Ridge Drive, Mountain Brook, AL", "US")], "S1")
        s23 = _frame([("Robbie Garza Laboratories Inc.", "61A SCARBOROUGH VILLAGE DR, DAYTON, OH", "US"),
                      ("Beam Boal Co", "#111 Mansfield Ave, Shelby, Ohio", "US"),
                      ("primarycaregroup.com", "49 RIDGE DR, MOUNTAIN BROOK, AL", "US"),
                      ("Primary Care Group", "12 Oak Street, Dallas, TX", "US")], "S2")
        self.s1 = normalize_frame(s1, None, 1)
        self.s23 = normalize_frame(s23, None, 1)

    def test_candidates_contain_true_matches(self):
        c = generate_candidates(self.s1, self.s23, k_comb=3, k_name=2, k_addr=2, max_df_word=100,
                                max_df_char=100, verbose=False)
        pairs = set(zip(c["i1"].to_list(), c["j"].to_list()))
        for i in range(3):
            self.assertIn((i, i), pairs)
        top = c.filter(pl.col("rank") == 1)
        self.assertEqual(sorted(zip(top["i1"].to_list(), top["j"].to_list())), [(0, 0), (1, 1), (2, 2)])

    def test_features_shape(self):
        c = generate_candidates(self.s1, self.s23, k_comb=3, k_name=2, k_addr=2, max_df_word=100,
                                max_df_char=100, verbose=False).with_columns(
            pl.len().over("i1").cast(pl.UInt16).alias("n_cands"))
        X = PairFeaturizer(self.s1, self.s23).featurize(c)
        self.assertEqual(X.shape, (len(c), len(FEATURE_NAMES)))
        self.assertFalse(np.isnan(X).any())

    def test_acronym_feature(self):
        s1 = normalize_frame(_frame([("Jai Infrastructure Private Limited", "67/11 Strand Road, Kolkata", "India")], "S1"), None, 1)
        s2 = normalize_frame(_frame([("jiprivate.com", "67/11 Strand Road, Calcutta, WB", "India"),
                                     ("Jai Infrastructure Pvt Ltd", "67/11 Strand Road, Kolkata", "India")], "S2"), None, 1)
        pairs = pl.DataFrame({"i1": [0, 0], "j": [0, 1], "cos_name": [0.0, 1.0], "cos_addr": [1.0, 1.0],
                              "cos_char": [0.0, 1.0], "cheap": [0.5, 1.0], "rank": [2, 1], "n_cands": [2, 2]})
        X = PairFeaturizer(s1, s2).featurize(pairs)
        pref = X[:, FEATURE_NAMES.index("acr_prefix")]
        rest = X[:, FEATURE_NAMES.index("acr_rest")]
        # initials of the full cleaned name "jai infrastructure pvt ltd" = "jipl" share "jip"
        self.assertEqual(pref[0], 3)
        self.assertGreaterEqual(rest[0], 0)
        self.assertEqual((pref[1], rest[1]), (-1, -1))  # multi-token name: not applicable

<<<<<<< HEAD
    def test_clipped_house_number(self):
        # the noise clips house numbers (2424 -> 424) and renames the business: only the crossed
        # exact<->clipped key links the pair; the pair feature flags it
        s1 = normalize_frame(_frame([("Glad Vasquez DDS", "2424 Peach Avenue, Marshfield, WI", "US"),
                                     ("Other Co", "12 Elm Street, Austin, TX", "US")], "S1"), None, 1)
        s2 = normalize_frame(_frame([("Calodelta", "424 PEACH AVENUE, MARSHIELD, WI", "US"),
                                     ("Zeta", "9 Oak Rd, Boise, ID", "US")], "S2"), None, 1)
        side1, side2 = word_features(s1, side=1), word_features(s2, side=2)
        self.assertGreater(side1.height, word_features(s1).height)  # retrieval-only keys added
        c = generate_candidates(s1, s2, k_comb=1, k_name=1, k_addr=1, max_df_word=10, max_df_char=10,
                                verbose=False).with_columns(pl.len().over("i1").cast(pl.UInt16).alias("n_cands"))
        self.assertIn((0, 0), set(zip(c["i1"].to_list(), c["j"].to_list())))
        X = PairFeaturizer(s1, s2).featurize(c.filter((pl.col("i1") == 0) & (pl.col("j") == 0)))
        self.assertEqual(X[0, FEATURE_NAMES.index("house_trunc")], 1)

=======
>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10
    def test_sparse_kernels(self):
        f = word_features(self.s1)
        A, B = build_tfidf(f, f, len(self.s1), len(self.s1), {"w": 1.0}, 100, 1)
        d = pair_dot(A, B, np.arange(3), np.arange(3))
        self.assertTrue(np.allclose(d[d > 0], 1.0, atol=1e-5))
        st = pair_stats(A, B, np.arange(3), np.arange(3))
        self.assertTrue(np.allclose(st[st[:, 0] > 0, 1], 1.0, atol=1e-5))


<<<<<<< HEAD
class TestV6(unittest.TestCase):
    def test_content_core(self):
        # the town word shared by both names is masked: "ecole" vs "amis" is a different business
        s1 = normalize_frame(_frame([("Tourcoing Ecole SARL", "12 Rue Lyderic, Tourcoing, Nord", "France")], "S1"), None, 1)
        s2 = normalize_frame(_frame([("Tourcoing Amis SARL", "12 Rue Lyderic, Tourcoing, Nord", "France"),
                                     ("Ecole Tourcoing", "12 R. Lyderic, Tourcoing", "France")], "S2"), None, 1)
        fz = PairFeaturizer(s1, s2)
        cc = fz.content_core(np.array([0, 0]), np.array([0, 1]))
        self.assertEqual((cc["cc_disjoint"][0], cc["cc_jacc"][0]), (1.0, 0.0))
        self.assertEqual((cc["cc_disjoint"][1], cc["cc_jacc"][1]), (0.0, 1.0))

    def test_block_params_open_set(self):
        from pipeline import DEFAULT_CONFIG, block_params
        self.assertEqual(block_params(DEFAULT_CONFIG, 0.18)["k_comb"], 40)
        self.assertEqual(block_params(DEFAULT_CONFIG, 0.0)["k_comb"], DEFAULT_CONFIG["block"]["k_comb"])

    def test_exact_expected_f(self):
        from decide import exact_f_select, fit_isotonic, apply_calibration
        # two 0.6 candidates: exact E[F] of predicting both is 0.627, one alone is better
        df = pl.DataFrame({"s1": [0, 0], "eid": ["a", "b"], "q": [0.6, 0.6]})
        _, ents = exact_f_select(df, return_ef=True, min_p=0.0)
        self.assertEqual(ents["_kb"][0], 2)
        self.assertGreater(ents["_ef"][0], 0.62)
        # a near-certain singleton-free entity predicts nothing
        _, ents = exact_f_select(pl.DataFrame({"s1": [1], "eid": ["c"], "q": [0.05]}), return_ef=True, min_p=0.0)
        self.assertEqual(ents["_kb"][0], 0)
        cal = fit_isotonic(np.array([0.1, 0.2, 0.8, 0.9] * 50), np.array([0, 0, 1, 1] * 50))
        q = apply_calibration(np.array([0.05, 0.95]), cal)
        self.assertTrue(q[0] < 0.1 and q[1] > 0.9)

    def test_shape_buckets_and_offsets(self):
        from decide import shape_buckets, apply_bucket_offsets
        from features import SIBLING_OFFSETS
        b = shape_buckets(pl.Series(["12", "12", "12"]), pl.Series(["13", "12", "12"]),
                          pl.Series(["ecole", "ecole", "ecole"]), pl.Series(["ecole", "amis", "ecole"]), SIBLING_OFFSETS)
        self.assertEqual(b.to_list(), ["sibling", "samehouse_swap", ""])
        q = apply_bucket_offsets(np.array([0.9, 0.9, 0.9]), b, {"sibling": -2.0})
        self.assertLess(q[0], 0.6)
        self.assertAlmostEqual(q[2], 0.9, 6)

    def test_state_imputation(self):
        from features import impute_state
        s1 = pl.DataFrame({"a_loc": ["bordeaux"] * 6, "a_state": ["nouvelle aquitaine"] * 6})
        s2 = pl.DataFrame({"a_loc": ["bordeaux", "nowhere"], "a_state": ["", ""]})
        _, out = impute_state(s1, s2)
        self.assertEqual(out["a_state"].to_list(), ["nouvelle aquitaine", ""])


=======
>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10
class TestMetricAndDecisions(unittest.TestCase):
    def test_metric(self):
        gt = {"a": {"x", "y"}, "b": set(), "c": {"z"}}
        pred = {"a": {"x", "y"}, "b": set(), "c": {"z", "w"}}
        # a=1, b=1 (singleton predicted empty), c: P=0.5 R=1 -> 0.625/0.625... F0.5 = 0.5556
        self.assertAlmostEqual(compute_macro_f05(gt, pred), (1 + 1 + (1.25 * 0.5) / (0.25 * 0.5 + 1)) / 3, 6)
        e = entity_scores(pl.DataFrame({"s1": ["a", "b", "c"]}),
                          pl.DataFrame({"s1": ["a", "a", "c"], "eid": ["x", "y", "z"]}),
                          pl.DataFrame({"s1": ["a", "a", "c", "c"], "eid": ["x", "y", "z", "w"]}))
        self.assertAlmostEqual(e["f05"].mean(), compute_macro_f05(gt, pred), 6)

    def test_one_to_one_and_thresholds(self):
        df = pl.DataFrame({"s1": ["a", "b", "a"], "eid": ["x", "x", "y"], "src": ["S2", "S2", "S3"],
                           "p": [0.9, 0.6, 0.4]})
        best = one_to_one(df)
        self.assertEqual(sorted(best["s1"].to_list()), ["a", "a"])
        kept = apply_thresholds(best, {"S2": 0.5, "S3": 0.5}, 0.5)
        self.assertEqual(kept["eid"].to_list(), ["x"])
        ef = expected_f_select(best)
        self.assertIn("x", ef["eid"].to_list())


if __name__ == "__main__":
    unittest.main()

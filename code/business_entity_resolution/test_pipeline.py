import unittest
import numpy as np
import polars as pl
from business_entity_resolution.src.normalize import clean_name, clean_compact_name, extract_addr_numbers, extract_addr_words
from business_entity_resolution.src.blocking import generate_blocking_keys, build_country_inverted_index, retrieve_candidates_for_s1
from business_entity_resolution.src.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.src.metrics import compute_macro_f05
from business_entity_resolution.src.model import EntityResolutionModel


class TestEntityResolutionPipeline(unittest.TestCase):

    def test_normalization(self):
        name = "Maure Williams Colombier Inc."
        cleaned = clean_name(name)
        self.assertNotIn("inc", cleaned.split())
        self.assertEqual(clean_compact_name("Maure Williams!"), "maurewilliams")
        
        addr = "9300 Perseverance Way, Suite 400, Seattle, WA 98101"
        nums = extract_addr_numbers(addr)
        self.assertIn("9300", nums)
        self.assertIn("400", nums)
        self.assertIn("98101", nums)

    def test_blocking_keys(self):
        keys = generate_blocking_keys("Acme Corp LLC", "123 Main St, Springfield")
        key_types = {k[0] for k in keys}
        self.assertTrue(any(t in key_types for t in ['cmp', 'sort', 'num_n1', 'num_str']))

    def test_feature_computation(self):
        s1_name = "Microsoft Corporation"
        s1_addr = "One Microsoft Way, Redmond, WA 98052"
        s2_name = "Microsoft Corp"
        s2_addr = "1 Microsoft Way, Redmond, WA"
        
        feats = compute_pair_features(s1_name, s1_addr, s2_name, s2_addr, "s2_12345")
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(x, (int, float, np.floating)) for x in feats))

    def test_metric_f05(self):
        gt = {
            "s1_1": {"s2_1", "s3_1"},
            "s1_2": set(), # singleton
            "s1_3": {"s2_2"}
        }
        pred = {
            "s1_1": {"s2_1", "s3_1"},
            "s1_2": set(), # correct singleton prediction -> 1.0
            "s1_3": {"s2_2", "s3_99"} # 1 TP, 1 FP
        }
        score = compute_macro_f05(gt, pred)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_model_initialization(self):
        model = EntityResolutionModel(optimal_threshold=0.65)
        self.assertEqual(model.optimal_threshold, 0.65)


if __name__ == "__main__":
    unittest.main()

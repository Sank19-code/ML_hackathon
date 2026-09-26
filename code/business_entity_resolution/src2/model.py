import os
import pickle
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np

try:
    from .metrics import compute_macro_f05
    from .features import FEATURE_NAMES
except ImportError:
    from metrics import compute_macro_f05
    from features import FEATURE_NAMES


class EntityResolutionModel:
    def __init__(self, optimal_threshold: float = 0.55):
        # LightGBM: fast, low-memory, high accuracy on tabular string features
        self.model = lgb.LGBMClassifier(
            n_estimators=800,       # More trees with lower learning rate
            learning_rate=0.03,     # Slower LR for better generalization
            num_leaves=63,          # Prevents overfitting on noise
            max_depth=7,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.2,          # L1 for feature selection
            reg_lambda=0.5,         # L2 for smoothness
            scale_pos_weight=1.5,   # Moderate balance without distorting probability calibration
            random_state=42,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )
        self.optimal_threshold = optimal_threshold
        self.feature_names = FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM binary classifier on candidate pair features."""
        pos = np.sum(y)
        neg = len(y) - pos
        print(f"Training LightGBM: {len(X)} pairs | Positives: {pos} ({pos/len(y)*100:.2f}%) | Negatives: {neg}")
        self.model.fit(X, y, feature_name=self.feature_names)
        print("Training complete.")
        # Print top feature importances
        importances = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("Top 8 features by gain:")
        for fname, imp in importances[:8]:
            print(f"  {fname}: {imp:.1f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability for candidate pairs."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        val_s1_ids: List[str],
        val_cand_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_ground_truth: Dict[str, Set[str]]
    ) -> float:
        """
        Grid-search the decision threshold τ* that maximises macro F_0.5,
        enforcing the global 1-to-1 unique assignment constraint for S2/S3 entities.
        """
        print("Optimising threshold for macro F_0.5 with 1-to-1 matching constraint...")

        best_score = -1.0
        best_tau = 0.55

        # Fine-grained search from 0.30 to 0.85
        for tau in np.arange(0.30, 0.86, 0.01):
            # Enforce 1-to-1 constraint: each candidate mid is assigned to its best S1 with prob >= tau
            best_match_for_mid: Dict[str, Tuple[str, float]] = {}
            for (s1_id, cand_id), prob in zip(val_cand_pairs, val_probs):
                p = float(prob)
                if p >= tau:
                    if cand_id not in best_match_for_mid or p > best_match_for_mid[cand_id][1]:
                        best_match_for_mid[cand_id] = (s1_id, p)

            preds: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_s1_ids}
            for cand_id, (s1_id, _) in best_match_for_mid.items():
                if s1_id in preds:
                    preds[s1_id].add(cand_id)

            score = compute_macro_f05(val_ground_truth, preds)
            if score > best_score:
                best_score = score
                best_tau = float(tau)

        self.optimal_threshold = best_tau
        print(f"  Optimal tau* = {best_tau:.3f}  ->  Validation Macro F_0.5 = {best_score:.4f}")
        return best_tau

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.optimal_threshold,
                         'features': self.feature_names}, f)
        print(f"Model saved -> {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(optimal_threshold=data['threshold'])
        instance.model = data['model']
        instance.feature_names = data.get('features', FEATURE_NAMES)
        return instance

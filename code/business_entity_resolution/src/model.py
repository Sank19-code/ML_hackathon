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
    def __init__(self, optimal_threshold: float = 0.65):
        # LightGBM: fast, low-memory, excellent on tabular similarity features
        self.model = lgb.LGBMClassifier(
            n_estimators=500,       # More trees for better recall of subtle matches
            learning_rate=0.05,     # Slower LR + more estimators → lower variance
            num_leaves=127,         # More leaves captures complex interactions
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,          # L1 for sparsity
            reg_lambda=0.1,         # L2 for smoothness
            class_weight='balanced', # Handles imbalanced pos/neg candidate pool
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
        print("Top 5 features by gain:")
        for fname, imp in importances[:5]:
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
        Grid-search the decision threshold τ* that maximises macro F_0.5.

        F_0.5 weights precision 2× over recall — so the optimal τ* is
        typically higher than the standard 0.5.
        """
        print("Optimising threshold for macro F_0.5...")

        # Group per S1 entity
        s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {
            s1_id: [] for s1_id in val_s1_ids
        }
        for (s1_id, cand_id), prob in zip(val_cand_pairs, val_probs):
            if s1_id in s1_to_scored_cands:
                s1_to_scored_cands[s1_id].append((cand_id, float(prob)))

        best_score = -1.0
        best_tau = 0.65

        # Fine-grained search from 0.40 to 0.90
        for tau in np.arange(0.40, 0.91, 0.01):
            preds: Dict[str, Set[str]] = {}
            for s1_id, scored_list in s1_to_scored_cands.items():
                preds[s1_id] = {cid for cid, p in scored_list if p >= tau}

            score = compute_macro_f05(val_ground_truth, preds)
            if score > best_score:
                best_score = score
                best_tau = float(tau)

        self.optimal_threshold = best_tau
        print(f"  Optimal τ* = {best_tau:.3f}  →  Validation Macro F_0.5 = {best_score:.4f}")
        return best_tau

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.optimal_threshold,
                         'features': self.feature_names}, f)
        print(f"Model saved → {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(optimal_threshold=data['threshold'])
        instance.model = data['model']
        instance.feature_names = data.get('features', FEATURE_NAMES)
        return instance

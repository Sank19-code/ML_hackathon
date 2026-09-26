"""
Gradient-boosted pair classifier (LightGBM, MIT licence) with cross-fitting.

Training rows are candidate pairs from the training universe. Folds are assigned by
Source 1 entity (hash of the entity id), so every candidate of an entity sits in the
same fold and no information leaks across the split. Each fold model scores the
other fold, giving out-of-fold probabilities for every training pair; those are used
to build the second-stage (group) features and to tune the decision thresholds.
At test time the fold models are averaged.
"""
import os
import pickle
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "verbose": -1,
    "num_threads": 15,
    "seed": 42,
}


def fold_of(ids: Sequence[str], n_folds: int = 2, salt: int = 7) -> np.ndarray:
    """Deterministic fold assignment from the numeric part of the entity id."""
    nums = np.array([int(s.split("-", 1)[1]) for s in ids], dtype=np.int64)
    return ((nums * 2654435761 + salt) % 1000003) % n_folds


class FoldEnsemble:
    def __init__(self, feature_names: List[str], params: Optional[Dict] = None, num_rounds: int = 600):
        self.feature_names = list(feature_names)
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.num_rounds = num_rounds
        self.models: List[lgb.Booster] = []

    def fit_fold(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> lgb.Booster:
        dtrain = lgb.Dataset(X, label=y, feature_name=self.feature_names, free_raw_data=True)
        valid = []
        if X_val is not None:
            valid = [lgb.Dataset(X_val, label=y_val, reference=dtrain)]
        booster = lgb.train(self.params, dtrain, num_boost_round=self.num_rounds, valid_sets=valid,
                            callbacks=[lgb.log_evaluation(200)] if valid else None)
        self.models.append(booster)
        return booster

    def dataset(self, X: np.ndarray, y: np.ndarray) -> lgb.Dataset:
        """Bin the full training matrix once (the raw float matrix can then be freed)."""
        ds = lgb.Dataset(X, label=y, feature_name=self.feature_names, free_raw_data=True,
                         params={"max_bin": self.params["max_bin"], "verbose": -1})
        return ds.construct()

    def fit_subset(self, full: lgb.Dataset, rows: np.ndarray) -> lgb.Booster:
        booster = lgb.train(self.params, full.subset(rows).construct(), num_boost_round=self.num_rounds)
        self.models.append(booster)
        return booster

    def predict(self, X: np.ndarray, fold: Optional[int] = None) -> np.ndarray:
        if fold is not None:
            return self.models[fold].predict(X, num_threads=self.params["num_threads"])
        p = np.zeros(len(X), dtype=np.float64)
        for m in self.models:
            p += m.predict(X, num_threads=self.params["num_threads"])
        return p / len(self.models)

    def importance(self, top: int = 20):
        imp = np.zeros(len(self.feature_names))
        for m in self.models:
            imp += m.feature_importance("gain")
        order = np.argsort(-imp)[:top]
        tot = imp.sum() or 1.0
        return [(self.feature_names[i], imp[i] / tot) for i in order]

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"feature_names": self.feature_names, "params": self.params,
                         "num_rounds": self.num_rounds,
                         "models": [m.model_to_string() for m in self.models]}, f)

    @classmethod
    def load(cls, path: str) -> "FoldEnsemble":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["feature_names"], d["params"], d["num_rounds"])
        obj.models = [lgb.Booster(model_str=s) for s in d["models"]]
        return obj

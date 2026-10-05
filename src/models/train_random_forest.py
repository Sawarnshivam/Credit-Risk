"""Random Forest builder.

Trees handle the mixed numeric feature matrix (ordinal-encoded categoricals,
NaNs imputed) without scaling. RF is a strong, low-tuning baseline for tabular
data. It is the slowest of the tree models here, so depth and estimator count
are kept modest for a single-core box.
"""
from __future__ import annotations

from typing import Optional

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(class_weight: Optional[str] = None, seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        max_features="sqrt",
        max_samples=0.6,  # per-tree bootstrap fraction; caps peak memory on a 4 GB box
        n_jobs=1,         # single core available; -1 adds joblib overhead here
        class_weight=class_weight,
        random_state=seed,
    )

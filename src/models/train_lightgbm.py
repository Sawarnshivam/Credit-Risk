"""LightGBM builder.

LightGBM is fast, memory-light, handles NaNs natively and usually gives the
best accuracy/compute trade-off on this dataset, which is why it ends up being
the model taken through tuning, calibration and SHAP.
"""
from __future__ import annotations

from typing import Optional

from lightgbm import LGBMClassifier

DEFAULT_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
)


def build_lightgbm(params: Optional[dict] = None, class_weight: Optional[str] = None, seed: int = 42) -> LGBMClassifier:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return LGBMClassifier(
        **p,
        objective="binary",
        class_weight=class_weight,
        n_jobs=1,  # 1 core available; OpenMP can otherwise over-allocate in a container
        random_state=seed,
        verbosity=-1,
    )

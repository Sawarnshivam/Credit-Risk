"""XGBoost builder.

Gradient-boosted trees are the workhorse for this kind of tabular credit data.
``hist`` tree method keeps training fast on one core. ``scale_pos_weight`` is
the XGBoost way of applying class weights and is toggled by the imbalance
experiments.
"""
from __future__ import annotations

from typing import Optional

from xgboost import XGBClassifier

DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
)


def build_xgboost(params: Optional[dict] = None, scale_pos_weight: float = 1.0, seed: int = 42) -> XGBClassifier:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return XGBClassifier(
        **p,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,  # 1 core available; OpenMP can otherwise over-allocate in a container
        random_state=seed,
    )

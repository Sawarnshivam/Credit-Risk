"""Logistic Regression baseline.

The baseline is a linear model on top of the ColumnTransformer (median-impute +
scale for numerics, most-frequent-impute + one-hot for categoricals). It gives
a reference point that the tree models have to beat, and because it is linear
and calibrated-ish out of the box it is a sanity check on the feature set.
"""
from __future__ import annotations

from typing import List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_linear_preprocessor


def build_logreg(
    num_cols: List[str],
    cat_cols: List[str],
    class_weight: Optional[str] = None,
    seed: int = 42,
) -> Pipeline:
    """Return a full preprocess + logistic-regression pipeline."""
    pre = build_linear_preprocessor(num_cols, cat_cols)
    clf = LogisticRegression(
        solver="lbfgs",
        C=0.1,  # l2 is the default penalty
        max_iter=200,
        class_weight=class_weight,
        random_state=seed,
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])

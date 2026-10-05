"""SHAP explainability for the final tree model (Phase 12).

TreeExplainer is exact and fast for gradient-boosted trees. We produce global
importance (mean |SHAP|), a summary/beeswarm plot, a bar plot, dependence plots
for the top features, and per-applicant local explanations.

Important framing: SHAP attributes the *model's* output to features. It does not
establish that a feature causes default. Language in the report reflects that.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import shap


def compute_shap(model, X: pd.DataFrame):
    """Return (explainer, shap_values) for a fitted tree model on X."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # LightGBM binary returns a single array; some versions return a list.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values


def global_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    row_idx: int,
    top_k: int = 5,
):
    """Top risk-increasing and risk-reducing features for one applicant."""
    vals = shap_values[row_idx]
    order = np.argsort(vals)
    reducing = [(feature_names[i], float(vals[i])) for i in order[:top_k]]
    increasing = [(feature_names[i], float(vals[i])) for i in order[::-1][:top_k]]
    return increasing, reducing

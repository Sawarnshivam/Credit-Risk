"""Tests for evaluation metrics, threshold logic, and model builders."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    full_metrics,
    gini_coefficient,
    ks_statistic,
    threshold_metrics,
)
from src.evaluation.threshold import choose_bands, sweep_thresholds


def _separable():
    # a perfectly separable problem: scores order the classes exactly
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y, s


def test_perfect_scores_give_auc_one():
    y, s = _separable()
    m = full_metrics(y, s, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["gini"] == 1.0  # gini = 2*auc - 1


def test_ks_between_zero_and_one():
    y, s = _separable()
    ks = ks_statistic(y, s)
    assert 0.0 <= ks <= 1.0
    assert ks == 1.0  # perfectly separable


def test_gini_matches_auc_relationship():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=500)
    s = rng.random(size=500)
    from sklearn.metrics import roc_auc_score
    assert abs(gini_coefficient(y, s) - (2 * roc_auc_score(y, s) - 1)) < 1e-6


def test_full_metrics_has_expected_keys():
    y, s = _separable()
    m = full_metrics(y, s)
    for k in ["roc_auc", "pr_auc", "ks", "gini", "brier", "precision", "recall", "f1"]:
        assert k in m


def test_threshold_metrics_recall_moves_with_threshold():
    y, s = _separable()
    hi = threshold_metrics(y, s, 0.95)   # very strict -> low recall
    lo = threshold_metrics(y, s, 0.05)   # very loose -> high recall
    assert lo["recall"] >= hi["recall"]


def test_sweep_and_bands():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=1000)
    s = rng.random(size=1000)
    sweep = sweep_thresholds(y, s, [0.2, 0.4, 0.6], cost_fn=5.0, cost_fp=1.0)
    assert {"threshold", "precision", "recall", "f1",
            "expected_cost_per_app"}.issubset(sweep.columns)
    bands = choose_bands(sweep)
    assert bands["approve_below"] <= bands["reject_above"]


def test_model_builders_return_fitted_types():
    from src.models.train_baseline import build_logreg
    from src.models.train_lightgbm import build_lightgbm
    from src.models.train_random_forest import build_random_forest
    from src.models.train_xgboost import build_xgboost

    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)[::-1]})
    y = np.array([0, 1] * 20)

    for build in [lambda: build_random_forest(seed=0),
                  lambda: build_xgboost(seed=0),
                  lambda: build_lightgbm(seed=0)]:
        model = build()
        model.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        assert p.shape == (40,)
        assert ((p >= 0) & (p <= 1)).all()

    # logreg builder takes column lists (it wraps a preprocessor)
    lr = build_logreg(num_cols=["a", "b"], cat_cols=[], seed=0)
    lr.fit(X, y)
    assert lr.predict_proba(X).shape == (40, 2)

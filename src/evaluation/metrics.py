"""Metrics used throughout the project.

Accuracy is deliberately avoided as a headline metric: with ~8% defaults a
model that predicts "never defaults" is ~92% accurate and useless. The metrics
here are the ones that matter for ranking and for probability quality:

* ROC-AUC  - ranking / discrimination across all thresholds.
* PR-AUC   - ranking quality focused on the positive (default) class, which is
             more informative than ROC-AUC under heavy imbalance.
* KS       - max separation between the cumulative good/bad distributions, a
             standard scorecard metric.
* Gini     - 2*AUC - 1, the usual credit-scoring restatement of AUC.
* Brier    - mean squared error of the predicted probabilities (calibration).
* Precision/Recall/F1 - reported at a chosen decision threshold.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between positive and negative score CDFs."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_true = y_true[order]
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = np.cumsum(pos) / n_pos
    cum_neg = np.cumsum(neg) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(2.0 * roc_auc_score(y_true, y_score) - 1.0)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """Precision / recall / F1 and confusion counts at a given threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def full_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """All headline metrics for a set of predicted probabilities."""
    m = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "gini": gini_coefficient(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    m.update(threshold_metrics(y_true, y_score, threshold))
    return m

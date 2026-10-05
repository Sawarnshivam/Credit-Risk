"""Threshold selection and a simple cost framework (Phase 11).

0.5 is not a sensible cut-off for an 8%-prevalence problem. We sweep thresholds
and, using an explicit cost for each error type, find the threshold that
minimises expected cost. Two-sided bands (approve / review / reject) are then
derived so the system outputs a decision, not just a class.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import threshold_metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: List[float],
    cost_fn: float,
    cost_fp: float,
) -> pd.DataFrame:
    """Return a table of precision/recall/F1/approval-rate/expected-cost."""
    n = len(y_true)
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_true, y_score, t)
        approve_rate = (np.asarray(y_score) < t).mean()
        expected_cost = (m["fn"] * cost_fn + m["fp"] * cost_fp) / n
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "approval_rate": float(approve_rate),
                "rejection_rate": float(1 - approve_rate),
                "expected_cost_per_app": float(expected_cost),
            }
        )
    return pd.DataFrame(rows)


def choose_bands(sweep: pd.DataFrame) -> Dict[str, float]:
    """Pick a reject threshold (min expected cost) and an approve threshold.

    Approve band: highest threshold whose recall is still low enough that few
    defaulters slip in among approvals -- we take the F1-optimal point as the
    review boundary and the cost-optimal point as the reject boundary. The
    lower of the two is the approve line, the higher is the reject line.
    """
    reject = float(sweep.loc[sweep["expected_cost_per_app"].idxmin(), "threshold"])
    approve = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
    lo, hi = sorted([approve, reject])
    # Ensure a genuine review band exists.
    if hi - lo < 1e-6:
        hi = min(1.0, lo + 0.05)
    return {"approve_below": lo, "reject_above": hi}

"""Probability calibration.

For lending we care about the *value* of the predicted probability, not only
the ranking: a "0.20 probability of default" should mean roughly 20% of such
applicants default. Tree ensembles are often mis-calibrated, so we compare the
raw model against Platt (sigmoid) and isotonic calibration and pick by Brier
score / reliability on a held-out set.

We use ``FrozenEstimator`` + ``CalibratedClassifierCV(cv="prefit")`` so the
base model is trained once on the training split and only the calibrator is fit
on a separate calibration split (no leakage, no retraining).
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(base_estimator, X_cal, y_cal, method: str = "isotonic"):
    """Fit a calibrator on top of an already-trained estimator."""
    cal = CalibratedClassifierCV(FrozenEstimator(base_estimator), method=method)
    cal.fit(X_cal, y_cal)
    return cal

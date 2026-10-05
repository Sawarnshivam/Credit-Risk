"""Plotting and MLflow logging helpers used by the training scripts."""
from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _ensure(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def plot_roc(y_true, y_score, label: str, out: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(out), dpi=110)
    plt.close()


def plot_pr(y_true, y_score, label: str, out: str) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    base = np.mean(y_true)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=label)
    plt.axhline(base, color="k", ls="--", alpha=0.4, label=f"baseline={base:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(out), dpi=110)
    plt.close()


def plot_confusion(y_true, y_pred, label: str, out: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["repaid", "default"]).plot(
        cmap="Blues", values_format="d"
    )
    plt.title(f"Confusion matrix - {label}")
    plt.tight_layout()
    plt.savefig(_ensure(out), dpi=110)
    plt.close()


def plot_calibration(curves: Dict[str, tuple], out: str) -> None:
    """curves: {label: (y_true, y_score)}."""
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for label, (y_true, y_score) in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default fraction")
    plt.title("Calibration (reliability) curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(out), dpi=110)
    plt.close()


def log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: Dict[str, float],
    model=None,
    artifact_paths: Optional[list] = None,
    experiment: str = "credit-risk",
) -> None:
    """Log one run to MLflow. Kept intentionally minimal."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # drop NaN
        for p in artifact_paths or []:
            if os.path.exists(p):
                mlflow.log_artifact(p)

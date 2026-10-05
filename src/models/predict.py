"""Reusable prediction pipeline (Phase 15).

Loads the saved final (calibrated) model, the feature order it was trained on,
and the risk bands derived in the threshold analysis, then maps a raw applicant
record to:

    {"probability_of_default": float, "risk_category": str, "decision": str}

The numbers are produced by the trained model at run time - nothing is
hard-coded. Risk bands come from reports/tables/risk_bands.json.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_config
from src.data.preprocessing import clean_application
from src.features.feature_engineering import application_features


class CreditRiskPredictor:
    def __init__(self, cfg_path: str = "configs/config.yaml"):
        self.cfg = load_config(cfg_path)
        mdir = self.cfg["paths"]["models_dir"]
        self.model = joblib.load(os.path.join(mdir, "final_model.joblib"))
        self.encoder = joblib.load(os.path.join(mdir, "tree_encoder.joblib"))
        self.feature_order = json.load(open(os.path.join(mdir, "feature_order.json")))
        bands_path = os.path.join(self.cfg["paths"]["tables_dir"], "risk_bands.json")
        self.bands = json.load(open(bands_path))

    def _prepare(self, applicant: Dict) -> pd.DataFrame:
        """Turn a raw application dict into the model's feature vector.

        Only application-level features are reconstructed here; relational
        aggregate columns default to NaN if not supplied, which the tree model
        handles. This keeps single-record scoring dependency-free.
        """
        df = pd.DataFrame([applicant])
        df = clean_application(df)
        df = application_features(df)
        # Encode the categorical application columns to the same ordinal codes
        # the model was trained on; unseen categories map to -1.
        df = self.encoder.transform(df)
        # Align to the trained feature order; unseen relational aggregates become
        # NaN, which the tree model handles natively.
        X = df.reindex(columns=self.feature_order)
        return X.astype("float32")

    def predict(self, applicant: Dict) -> Dict:
        X = self._prepare(applicant)
        pd_hat = float(self.model.predict_proba(X)[:, 1][0])
        if pd_hat < self.bands["approve_below"]:
            cat, decision = "LOW", "APPROVE"
        elif pd_hat >= self.bands["reject_above"]:
            cat, decision = "HIGH", "REJECT"
        else:
            cat, decision = "MEDIUM", "REVIEW"
        return {
            "probability_of_default": round(pd_hat, 4),
            "risk_category": cat,
            "decision": decision,
        }

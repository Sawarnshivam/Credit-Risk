"""Preprocessing for the application data.

Two kinds of preprocessing live here:

1. ``clean_application`` - deterministic, domain-specific cleaning that fixes
   known encoding problems in the Home Credit application table (sentinel
   values, an impossible gender category, a couple of extreme outliers).
   This is not "learned" from the data, so applying it to train and test
   separately does not leak anything.

2. Fitted transformers used by the models:
   - ``build_linear_preprocessor`` returns a ColumnTransformer that imputes and
     scales numeric columns and one-hot encodes categoricals. Used by the
     Logistic Regression baseline. It is fit on training data only.
   - ``TreeEncoder`` ordinal-encodes categoricals (learning the mapping on
     train) and leaves numeric columns and their NaNs untouched, which is what
     LightGBM / XGBoost want.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Sentinel used in DAYS_EMPLOYED for people who are not employed (pensioners,
# unemployed). It is ~1000 years and clearly not a real value.
DAYS_EMPLOYED_ANOMALY = 365243


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the application table.

    Returns a copy; safe to call on train and test independently.
    """
    df = df.copy()

    # DAYS_EMPLOYED sentinel -> NaN, keep a flag because the anomaly itself is
    # informative (it marks pensioners/unemployed).
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # A handful of rows have CODE_GENDER == 'XNA'. Treat as missing.
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # A few AMT_INCOME_TOTAL values are absurdly large (e.g. 1.17e8). Cap at the
    # 99.9th percentile computed on this frame to stop them dominating scaling.
    if "AMT_INCOME_TOTAL" in df.columns:
        cap = df["AMT_INCOME_TOTAL"].quantile(0.999)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=cap)

    return df


def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio features can divide by zero. Turn +/-inf into NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


def split_feature_types(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) excluding id/target columns."""
    feats = [c for c in df.columns if c not in exclude]
    # Treat any non-numeric column as categorical. Under pandas 3 text columns
    # can be dtype 'str' rather than 'object', so key off numeric-ness.
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in feats if c not in cat]
    return num, cat


def build_linear_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """ColumnTransformer for the linear baseline (fit on train only)."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


@dataclass
class TreeEncoder:
    """Ordinal-encode object columns for tree models; pass numerics through.

    Unknown categories at transform time map to -1, missing categories stay NaN.
    Fit on training data only.
    """

    cat_cols: List[str] = field(default_factory=list)
    _encoder: OrdinalEncoder = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> "TreeEncoder":
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        if self.cat_cols:
            self._encoder.fit(df[self.cat_cols].astype("object"))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.cat_cols:
            out[self.cat_cols] = self._encoder.transform(out[self.cat_cols].astype("object"))
            out[self.cat_cols] = out[self.cat_cols].astype("float32")
        return replace_infinities(out)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

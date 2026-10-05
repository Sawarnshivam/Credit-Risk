"""Loading helpers for the Home Credit tables.

The application tables fit in memory comfortably. The relational tables are
large (POS_CASH is ~10M rows, bureau_balance ~27M), so anything that needs the
raw relational data reads it in chunks from the callers in feature_engineering.
Here we only expose simple loaders plus a dtype-downcasting helper that keeps
the application frame small.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Read the YAML config into a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def raw_path(cfg: dict, key: str) -> str:
    """Absolute path to a raw CSV given its logical key in the config."""
    return os.path.join(cfg["paths"]["raw_dir"], cfg["data"]["files"][key])


def file_exists(cfg: dict, key: str) -> bool:
    return os.path.exists(raw_path(cfg, key))


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in place to save memory.

    float64 -> float32, int64 -> the smallest signed int that fits. Object
    columns are left as-is (converted to category later where useful).
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_application(cfg: dict, which: str = "train") -> pd.DataFrame:
    """Load application_train or application_test with downcast dtypes."""
    key = "application_train" if which == "train" else "application_test"
    df = pd.read_csv(raw_path(cfg, key))
    return downcast(df)


def load_table(cfg: dict, key: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load an arbitrary raw table (used for small tables / inspection)."""
    return downcast(pd.read_csv(raw_path(cfg, key), nrows=nrows))


@lru_cache(maxsize=1)
def load_column_descriptions(cfg_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Load the column description reference shipped with the dataset."""
    cfg = load_config(cfg_path)
    path = os.path.join(cfg["paths"]["raw_dir"], "HomeCredit_columns_description.csv")
    return pd.read_csv(path, encoding="latin1")

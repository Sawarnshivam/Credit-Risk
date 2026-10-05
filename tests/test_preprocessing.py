"""Tests for the cleaning / encoding helpers.

These use tiny hand-built frames so they run in well under a second and don't
touch the real data files.
"""
import numpy as np
import pandas as pd

from src.data.preprocessing import (
    TreeEncoder,
    clean_application,
    replace_infinities,
    split_feature_types,
)


def _toy_application():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_EMPLOYED": [-1200, 365243, -300],   # middle row is the sentinel
            "CODE_GENDER": ["M", "XNA", "F"],
            "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 117_000_000.0],  # last is absurd
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
            "TARGET": [0, 1, 0],
        }
    )


def test_days_employed_sentinel_becomes_nan_with_flag():
    out = clean_application(_toy_application())
    assert out["DAYS_EMPLOYED"].isna().sum() == 1
    # the flag marks exactly the sentinel row
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0]


def test_gender_xna_becomes_nan():
    out = clean_application(_toy_application())
    assert out["CODE_GENDER"].isna().sum() == 1
    assert set(out["CODE_GENDER"].dropna()) == {"M", "F"}


def test_income_is_capped():
    out = clean_application(_toy_application())
    # the 117M outlier must be pulled down to the frame's 99.9th percentile
    assert out["AMT_INCOME_TOTAL"].max() < 117_000_000.0


def test_clean_application_does_not_mutate_input():
    df = _toy_application()
    before = df["AMT_INCOME_TOTAL"].copy()
    _ = clean_application(df)
    pd.testing.assert_series_equal(df["AMT_INCOME_TOTAL"], before)


def test_split_feature_types_detects_categoricals():
    df = _toy_application()
    num, cat = split_feature_types(df, exclude=["SK_ID_CURR", "TARGET"])
    assert "CODE_GENDER" in cat and "NAME_CONTRACT_TYPE" in cat
    assert "AMT_INCOME_TOTAL" in num and "DAYS_EMPLOYED" in num
    # id / target must be excluded from both lists
    assert "SK_ID_CURR" not in num + cat
    assert "TARGET" not in num + cat


def test_replace_infinities():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})
    out = replace_infinities(df)
    assert out["a"].isna().sum() == 2


def test_tree_encoder_encodes_to_numeric_and_handles_unknown():
    train = pd.DataFrame({"g": ["A", "B", "A"], "x": [1.0, 2.0, 3.0]})
    enc = TreeEncoder(cat_cols=["g"])
    tr = enc.fit_transform(train)
    # categorical column is now numeric
    assert pd.api.types.is_numeric_dtype(tr["g"])
    # an unseen category at transform time maps to the -1 sentinel
    test = pd.DataFrame({"g": ["Z"], "x": [9.0]})
    te = enc.transform(test)
    assert te["g"].iloc[0] == -1

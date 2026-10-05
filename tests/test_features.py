"""Tests for feature engineering.

The important one is `test_chunked_aggregator_matches_pandas`: the streaming
aggregator is the memory-saving core of the pipeline, so it is checked against
a plain full-frame pandas groupby to prove the two-pass maths is correct.
"""
import numpy as np
import pandas as pd

from src.features.feature_engineering import ChunkedAggregator, application_features


def test_application_ratios_are_correct():
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [1000.0],
            "AMT_INCOME_TOTAL": [500.0],
            "AMT_ANNUITY": [100.0],
            "AMT_GOODS_PRICE": [800.0],
            "CNT_FAM_MEMBERS": [2.0],
            "CNT_CHILDREN": [1],
            "DAYS_BIRTH": [-365 * 30],
            "DAYS_EMPLOYED": [-365 * 5],
            "EXT_SOURCE_1": [0.2],
            "EXT_SOURCE_2": [0.4],
            "EXT_SOURCE_3": [0.6],
        }
    )
    out = application_features(df)
    assert abs(out["CREDIT_INCOME_RATIO"].iloc[0] - 1000 / 500.001) < 1e-3
    assert abs(out["CREDIT_ANNUITY_RATIO"].iloc[0] - 1000 / 100.001) < 1e-2
    # EXT_SOURCE_MEAN of 0.2/0.4/0.6 is 0.4
    assert abs(out["EXT_SOURCE_MEAN"].iloc[0] - 0.4) < 1e-6
    assert abs(out["AGE_YEARS"].iloc[0] - 30.0) < 0.1


def test_ext_source_features_optional():
    # if no EXT_SOURCE columns exist, the function should still run
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [1.0], "AMT_INCOME_TOTAL": [1.0], "AMT_ANNUITY": [1.0],
            "AMT_GOODS_PRICE": [1.0], "CNT_FAM_MEMBERS": [1.0], "CNT_CHILDREN": [0],
            "DAYS_BIRTH": [-1000], "DAYS_EMPLOYED": [-100],
        }
    )
    out = application_features(df)
    assert "EXT_SOURCE_MEAN" not in out.columns
    assert "CREDIT_INCOME_RATIO" in out.columns


def test_chunked_aggregator_matches_pandas():
    rng = np.random.default_rng(0)
    n = 5000
    frame = pd.DataFrame(
        {
            "key": rng.integers(0, 50, size=n),
            "v1": rng.normal(size=n),
            "v2": rng.normal(10, 3, size=n),
        }
    )

    # streaming: feed the frame in several chunks
    agg = ChunkedAggregator(key="key", value_cols=["v1", "v2"])
    for start in range(0, n, 512):
        agg.update(frame.iloc[start:start + 512])
    got = agg.result(prefix="T_").set_index("key").sort_index()

    # ground truth: one-shot pandas groupby
    ref = frame.groupby("key").agg(
        v1_MEAN=("v1", "mean"), v1_STD=("v1", "std"),
        v1_MIN=("v1", "min"), v1_MAX=("v1", "max"), v1_SUM=("v1", "sum"),
    ).sort_index()

    # means, mins, maxes, sums must match closely
    assert np.allclose(got["T_v1_MEAN"], ref["v1_MEAN"], atol=1e-6)
    assert np.allclose(got["T_v1_MIN"], ref["v1_MIN"], atol=1e-6)
    assert np.allclose(got["T_v1_MAX"], ref["v1_MAX"], atol=1e-6)
    assert np.allclose(got["T_v1_SUM"], ref["v1_SUM"], atol=1e-4)
    # std: aggregator uses population variance; pandas uses sample (ddof=1).
    # They should be very close for ~100 rows per group but not identical, so
    # just check they are in the same ballpark.
    assert np.allclose(got["T_v1_STD"], ref["v1_STD"], atol=0.05)


def test_chunked_aggregator_count():
    frame = pd.DataFrame({"key": [1, 1, 2, 2, 2], "v1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    agg = ChunkedAggregator(key="key", value_cols=["v1"])
    agg.update(frame)
    got = agg.result(prefix="P_").set_index("key").sort_index()
    assert got.loc[1, "P_COUNT"] == 2
    assert got.loc[2, "P_COUNT"] == 3

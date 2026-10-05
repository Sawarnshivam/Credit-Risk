"""Feature engineering.

Two parts:

* ``application_features`` builds interpretable ratios on the main application
  table (loan-to-income, annuity-to-income, age, employment ratios, ...).

* The relational helpers turn the one-row-per-loan / one-row-per-month history
  tables (bureau, bureau_balance, previous_application, POS_CASH,
  credit_card) into one row per customer (SK_ID_CURR).

The relational tables are large and this environment has ~4 GB of RAM, so the
aggregations are done with a streaming two-pass combiner (``ChunkedAggregator``)
that reads the CSV in chunks and never holds a full table in memory. It
computes count / sum / mean / min / max / std per group without ever needing
the whole column at once (mean and std come from running sum, sum-of-squares
and count).

I looked at doing this in PySpark (it is in the intended stack), but on a
single-core, 4 GB machine the JVM/Spark overhead was not worth it - the data
fits comfortably with chunked pandas. The aggregation logic here is the same
logic I would express as a Spark ``groupBy().agg()``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ID = "SK_ID_CURR"


# --------------------------------------------------------------------------- #
# Application-level features
# --------------------------------------------------------------------------- #
def application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable ratio / interaction features to the application frame.

    Assumes ``clean_application`` has already run (DAYS_EMPLOYED sentinel fixed).
    """
    df = df.copy()
    eps = 1e-3  # guard against divide-by-zero; infinities are cleaned later

    # Affordability ratios -----------------------------------------------------
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + eps)  # ~loan term
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + eps)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)
    df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (1.0 + df["CNT_CHILDREN"])

    # Age / employment ---------------------------------------------------------
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).astype("float32")
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).astype("float32")
    df["EMPLOYED_TO_AGE_RATIO"] = -df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] - eps)

    # External scores are the strongest single features in this dataset; a few
    # cheap combinations of them tend to help.
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext].max(axis=1)
        df["EXT_SOURCE_STD"] = df[ext].std(axis=1)
        df["EXT_SOURCE_PROD"] = df[ext].prod(axis=1)

    return df


# --------------------------------------------------------------------------- #
# Streaming aggregation over a large relational table
# --------------------------------------------------------------------------- #
class ChunkedAggregator:
    """Two-pass streaming group aggregation.

    For each chunk we compute partial sums (sum, sum of squares, count, min,
    max) per group, accumulate the small partial frames, then combine them once
    at the end into count/mean/std/min/max/sum. Memory stays proportional to the
    number of groups, not the number of rows.
    """

    def __init__(self, key: str, value_cols: List[str]):
        self.key = key
        self.value_cols = value_cols
        self._partials: List[pd.DataFrame] = []

    def update(self, chunk: pd.DataFrame) -> None:
        cols = [c for c in self.value_cols if c in chunk.columns]
        if not cols:
            return
        sub = chunk[[self.key] + cols].copy()
        sq = sub[cols].pow(2)
        sq.columns = [f"{c}__sq" for c in cols]
        sub = pd.concat([sub, sq], axis=1)

        agg_map: Dict[str, str] = {}
        for c in cols:
            agg_map[c] = "sum"
        grouped = sub.groupby(self.key, sort=False)
        g_sum = grouped[cols].sum(min_count=1)
        g_cnt = grouped[cols].count()
        g_min = grouped[cols].min()
        g_max = grouped[cols].max()
        g_sq = sub.groupby(self.key, sort=False)[[f"{c}__sq" for c in cols]].sum(min_count=1)
        g_size = grouped.size().rename("__size")

        part = pd.concat(
            [
                g_sum.add_suffix("__sum"),
                g_cnt.add_suffix("__cnt"),
                g_min.add_suffix("__min"),
                g_max.add_suffix("__max"),
                g_sq,
                g_size,
            ],
            axis=1,
        )
        self._partials.append(part.reset_index())

    def result(self, prefix: str) -> pd.DataFrame:
        if not self._partials:
            return pd.DataFrame(columns=[self.key])
        allp = pd.concat(self._partials, ignore_index=True)
        cols = [c for c in self.value_cols]
        sum_cols = [f"{c}__sum" for c in cols]
        cnt_cols = [f"{c}__cnt" for c in cols]
        min_cols = [f"{c}__min" for c in cols]
        max_cols = [f"{c}__max" for c in cols]
        sq_cols = [f"{c}__sq" for c in cols]

        g = allp.groupby(self.key, sort=False)
        out = pd.DataFrame(index=g.size().index)
        out["__size"] = g["__size"].sum()

        s_sum = g[sum_cols].sum(min_count=1)
        s_cnt = g[cnt_cols].sum(min_count=1)
        s_min = g[min_cols].min()
        s_max = g[max_cols].max()
        s_sq = g[sq_cols].sum(min_count=1)

        result = {}
        for c in cols:
            n = s_cnt[f"{c}__cnt"]
            total = s_sum[f"{c}__sum"]
            mean = total / n.replace(0, np.nan)
            var = (s_sq[f"{c}__sq"] / n.replace(0, np.nan)) - mean.pow(2)
            var = var.clip(lower=0)
            result[f"{prefix}{c}_MEAN"] = mean
            result[f"{prefix}{c}_STD"] = np.sqrt(var)
            result[f"{prefix}{c}_MIN"] = s_min[f"{c}__min"]
            result[f"{prefix}{c}_MAX"] = s_max[f"{c}__max"]
            result[f"{prefix}{c}_SUM"] = total
        res = pd.DataFrame(result, index=out.index)
        res[f"{prefix}COUNT"] = out["__size"].astype("float32")
        return res.reset_index().rename(columns={"index": self.key}).astype(
            {self.key: "int64"}
        )


def _iter_csv(path: str, usecols: List[str], chunksize: int, dtype=None) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=dtype)


# --------------------------------------------------------------------------- #
# Table-specific aggregations
# --------------------------------------------------------------------------- #
def aggregate_bureau(path: str, chunksize: int) -> pd.DataFrame:
    """Customer-level features from the credit-bureau table.

    Counts of active / closed / bad-debt credits, overdue amounts, credit sums,
    and how recent the bureau records are.
    """
    numeric = [
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
    ]
    indicators = ["ACTIVE", "CLOSED", "BAD_DEBT"]
    agg = ChunkedAggregator(ID, numeric + indicators)
    usecols = [ID, "CREDIT_ACTIVE"] + numeric
    for chunk in _iter_csv(path, usecols, chunksize):
        chunk["ACTIVE"] = (chunk["CREDIT_ACTIVE"] == "Active").astype("float32")
        chunk["CLOSED"] = (chunk["CREDIT_ACTIVE"] == "Closed").astype("float32")
        chunk["BAD_DEBT"] = (chunk["CREDIT_ACTIVE"] == "Bad debt").astype("float32")
        agg.update(chunk)
    out = agg.result("BURO_")
    # A few directly interpretable rollups.
    out["BURO_ACTIVE_COUNT"] = out["BURO_ACTIVE_SUM"]
    out["BURO_CLOSED_COUNT"] = out["BURO_CLOSED_SUM"]
    out["BURO_BAD_DEBT_COUNT"] = out["BURO_BAD_DEBT_SUM"]
    out["BURO_ACTIVE_RATE"] = out["BURO_ACTIVE_SUM"] / out["BURO_COUNT"].replace(0, np.nan)
    return out


def aggregate_bureau_balance(bb_path: str, bureau_path: str, chunksize: int) -> pd.DataFrame:
    """Monthly bureau-balance delinquency features, rolled up to the customer.

    bureau_balance is keyed by SK_ID_BUREAU, so we aggregate per bureau credit
    first, then join SK_ID_CURR from the bureau table and aggregate again.
    """
    # Pass 1: aggregate bureau_balance per SK_ID_BUREAU.
    bb_agg = ChunkedAggregator("SK_ID_BUREAU", ["MONTHS_BALANCE", "DPD", "IS_CLOSED", "IS_UNKNOWN"])
    for chunk in _iter_csv(bb_path, ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], chunksize):
        status = chunk["STATUS"]
        chunk["DPD"] = status.isin(["1", "2", "3", "4", "5"]).astype("float32")
        chunk["IS_CLOSED"] = (status == "C").astype("float32")
        chunk["IS_UNKNOWN"] = (status == "X").astype("float32")
        bb_agg.update(chunk)
    per_bureau = bb_agg.result("BB_")

    # Join SK_ID_CURR (bureau is small enough to read in one shot with 2 cols).
    bureau_map = pd.read_csv(bureau_path, usecols=[ID, "SK_ID_BUREAU"])
    per_bureau = per_bureau.merge(bureau_map, on="SK_ID_BUREAU", how="left")

    # Pass 2: keep only interpretable per-bureau summaries, then roll up to the
    # customer with mean/max (share of delinquent months, worst delinquency,
    # share closed/unknown, oldest record, history length).
    keep = [
        "BB_DPD_MEAN", "BB_DPD_MAX", "BB_IS_CLOSED_MEAN", "BB_IS_UNKNOWN_MEAN",
        "BB_MONTHS_BALANCE_MIN", "BB_COUNT",
    ]
    keep = [c for c in keep if c in per_bureau.columns]
    grouped = per_bureau.groupby(ID)[keep]
    out = grouped.agg(["mean", "max"])
    out.columns = [f"{a}_{b.upper()}" for a, b in out.columns]
    return out.reset_index()


def aggregate_previous_application(path: str, chunksize: int) -> pd.DataFrame:
    """Features from prior Home Credit applications (approved/refused counts,
    requested vs granted amounts, down-payment rates)."""
    numeric = [
        "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE", "RATE_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT",
    ]
    indicators = ["APPROVED", "REFUSED", "CANCELED", "UNUSED"]
    agg = ChunkedAggregator(ID, numeric + indicators + ["ASK_TO_GRANT"])
    usecols = [ID, "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT"] + [
        c for c in numeric if c not in ("AMT_APPLICATION", "AMT_CREDIT")
    ]
    for chunk in _iter_csv(path, usecols, chunksize):
        st = chunk["NAME_CONTRACT_STATUS"]
        chunk["APPROVED"] = (st == "Approved").astype("float32")
        chunk["REFUSED"] = (st == "Refused").astype("float32")
        chunk["CANCELED"] = (st == "Canceled").astype("float32")
        chunk["UNUSED"] = (st == "Unused offer").astype("float32")
        chunk["ASK_TO_GRANT"] = chunk["AMT_CREDIT"] / (chunk["AMT_APPLICATION"] + 1e-3)
        agg.update(chunk)
    out = agg.result("PREV_")
    out["PREV_APPROVED_COUNT"] = out["PREV_APPROVED_SUM"]
    out["PREV_REFUSED_COUNT"] = out["PREV_REFUSED_SUM"]
    out["PREV_REFUSED_RATE"] = out["PREV_REFUSED_SUM"] / out["PREV_COUNT"].replace(0, np.nan)
    return out


def aggregate_pos_cash(path: str, chunksize: int) -> pd.DataFrame:
    """POS / cash-loan monthly balances: months of history and days-past-due."""
    numeric = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
    agg = ChunkedAggregator(ID, numeric + ["IS_DPD"])
    usecols = [ID] + numeric
    for chunk in _iter_csv(path, usecols, chunksize):
        chunk["IS_DPD"] = (chunk["SK_DPD"] > 0).astype("float32")
        agg.update(chunk)
    out = agg.result("POS_")
    out["POS_DPD_RATE"] = out["POS_IS_DPD_SUM"] / out["POS_COUNT"].replace(0, np.nan)
    return out


def aggregate_credit_card(path: str, chunksize: int) -> pd.DataFrame:
    """Credit-card monthly balances: utilisation, drawings, days-past-due."""
    numeric = [
        "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_CURRENT", "AMT_RECIVABLE",
        "SK_DPD", "SK_DPD_DEF",
    ]
    agg = ChunkedAggregator(ID, numeric + ["UTILISATION", "IS_DPD"])
    usecols = [ID] + numeric
    for chunk in _iter_csv(path, usecols, chunksize):
        chunk["UTILISATION"] = chunk["AMT_BALANCE"] / (chunk["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-3)
        chunk["IS_DPD"] = (chunk["SK_DPD"] > 0).astype("float32")
        agg.update(chunk)
    out = agg.result("CC_")
    out["CC_DPD_RATE"] = out["CC_IS_DPD_SUM"] / out["CC_COUNT"].replace(0, np.nan)
    return out

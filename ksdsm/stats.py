"""Rainfall statistics compared in the paper (SDSM 'Summary Statistics' equivalents).

Monthly statistics are computed per calendar month over all years, giving 12 values:
  mean    mean daily rainfall (mm/day)
  max     mean of the yearly maximum daily rainfall in that month (mm)
  sum     mean monthly total (mm)
  dryspell, wetspell  mean length of dry / wet spells (days), spells assigned to the
          month in which they start
  wetday  fraction of wet days
Annual totals (one value per year) are returned by `annual_totals`.
"""
import numpy as np
import pandas as pd

WET_THRESHOLD = 0.3  # mm/day, SDSM default


def _spells(wet: pd.Series):
    """Return DataFrame of spells: start date, length, is_wet (NaN days break spells)."""
    v = wet.to_numpy()
    out, start, cur = [], None, None
    idx = wet.index
    for i, x in enumerate(v):
        if pd.isna(x):
            if cur is not None:
                out.append((idx[start], i - start, cur))
            start, cur = None, None
            continue
        x = bool(x)
        if cur is None:
            start, cur = i, x
        elif x != cur:
            out.append((idx[start], i - start, cur))
            start, cur = i, x
    if cur is not None:
        out.append((idx[start], len(v) - start, cur))
    return pd.DataFrame(out, columns=["start", "length", "wet"])


def monthly_stats(series: pd.Series, threshold: float = WET_THRESHOLD) -> pd.DataFrame:
    """12-row table (index 1..12) of the monthly statistics for a daily series."""
    s = series.astype(float)
    wet = (s >= threshold).where(s.notna())
    df = pd.DataFrame({"p": s, "wet": wet})
    df["year"], df["month"] = df.index.year, df.index.month
    g = df.groupby("month")
    res = pd.DataFrame(index=pd.RangeIndex(1, 13, name="month"))
    res["mean"] = g["p"].mean()
    ym = df.groupby(["year", "month"])
    res["max"] = ym["p"].max().groupby("month").mean()
    res["sum"] = ym["p"].sum(min_count=1).groupby("month").mean()
    res["wetday"] = g["wet"].mean()
    sp = _spells(wet)
    sp["month"] = pd.DatetimeIndex(sp["start"]).month
    res["dryspell"] = sp[~sp.wet].groupby("month")["length"].mean()
    res["wetspell"] = sp[sp.wet].groupby("month")["length"].mean()
    return res


def annual_totals(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.year).sum(min_count=300)


def ensemble_monthly_stats(ens: pd.DataFrame, threshold: float = WET_THRESHOLD) -> pd.DataFrame:
    """Average monthly statistics over ensemble members (columns)."""
    tabs = [monthly_stats(ens[c], threshold) for c in ens.columns]
    return sum(tabs) / len(tabs)

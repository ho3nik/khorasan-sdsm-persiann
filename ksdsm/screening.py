"""Predictor screening: correlation and partial correlation with p-values, by season.

Mirrors SDSM's 'Screen Variables' step. For precipitation (a conditional process),
predictors are screened against the wet-day indicator and against the fourth-root of
wet-day amounts.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .model import group_key
from .stats import WET_THRESHOLD


def _partial_corr(x, y, Z):
    if Z.shape[1] == 0:
        return stats.pearsonr(x, y)
    A = np.column_stack([np.ones(len(x)), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    return r, 2 * stats.t.sf(abs(t), dof)


def screen(y: pd.Series, X: pd.DataFrame, grouping: str = "season",
           threshold: float = WET_THRESHOLD) -> pd.DataFrame:
    """Table of r, partial r and p-values per group, predictor and process."""
    df = X.join(y.rename("y"), how="inner").dropna()
    keys = group_key(df.index, grouping)
    rows = []
    for k in np.unique(keys):
        d = df[keys == k]
        wet = d["y"] >= threshold
        for proc in ("occurrence", "amount"):
            dd = d if proc == "occurrence" else d[wet]
            t = wet.astype(float).to_numpy() if proc == "occurrence" else dd["y"].to_numpy() ** 0.25
            if len(dd) < 10 or np.std(t) == 0:
                continue
            for c in X.columns:
                others = dd[[o for o in X.columns if o != c]].to_numpy()
                r, p = stats.pearsonr(dd[c], t)
                pr, pp = _partial_corr(dd[c].to_numpy(), t, others)
                rows.append(dict(group=k, process=proc, predictor=c, r=r, p=p, partial_r=pr, partial_p=pp))
    return pd.DataFrame(rows)


def select_predictors(table: pd.DataFrame, k: int = 5, alpha: float = 0.05) -> list:
    """Pick the k predictors with the largest mean |partial r| among significant entries."""
    sig = table[table.partial_p < alpha].copy()
    if sig.empty:
        sig = table.copy()
    sig["a"] = sig.partial_r.abs()
    return list(sig.groupby("predictor")["a"].mean().sort_values(ascending=False).index[:k])

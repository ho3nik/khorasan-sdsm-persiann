"""SDSM-style conditional regression model for daily precipitation.

Follows the structure of SDSM (Wilby et al., 2002; Wilby and Dawson, 2004):

* predictors are standardised (z-scores) against a baseline period;
* the model is fitted separately for each month, season or the whole year;
* occurrence: linear regression of the wet-day indicator (0/1) on the predictors;
  a day is generated as wet when a uniform random number <= predicted probability;
* amount: linear regression of the fourth root of wet-day rainfall on the predictors;
  amounts are generated as (prediction + Gaussian noise)**4;
* `variance_inflation` scales the noise standard deviation (1 = SDSM default of 12),
  `bias_correction` scales generated amounts (1 = none);
* an ensemble of stochastic series is generated.

Differences from SDSM: parameters are fitted by ordinary least squares (SDSM uses a
dual-simplex / least-squares option), and `fit(..., tune_bias=True)` sets the bias
factor so the calibration-period mean wet-day amount matches the observed one,
where SDSM users tune it by hand.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .stats import WET_THRESHOLD

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def group_key(index: pd.DatetimeIndex, grouping: str) -> np.ndarray:
    if grouping == "month":
        return index.month.to_numpy()
    if grouping == "season":
        return np.array([SEASONS[m] for m in index.month])
    if grouping == "annual":
        return np.full(len(index), "ALL")
    raise ValueError(f"grouping must be month, season or annual, not {grouping!r}")


@dataclass
class Standardizer:
    """z-score each predictor using the mean/std of a baseline period."""
    mean: pd.Series = None
    std: pd.Series = None

    def fit(self, X: pd.DataFrame, start=None, end=None):
        base = X.loc[str(start) if start else None: str(end) if end else None]
        self.mean, self.std = base.mean(), base.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.mean) / self.std


def _ols(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(len(y) - A.shape[1], 1)
    return beta, float(np.sqrt(resid @ resid / dof))


@dataclass
class SDSMPrecip:
    predictors: list
    grouping: str = "season"
    threshold: float = WET_THRESHOLD
    transform_power: float = 0.25          # fourth-root transform of amounts
    variance_inflation: float = 1.0
    bias_correction: float = 1.0
    params_: dict = field(default_factory=dict)

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.predictors].to_numpy(float)

    def fit(self, y: pd.Series, X: pd.DataFrame, tune_bias: bool = False, seed: int = 0):
        """y: observed daily rainfall (NaN = missing); X: standardised predictors."""
        df = X[self.predictors].join(y.rename("y"), how="inner").dropna()
        keys = group_key(df.index, self.grouping)
        self.params_ = {}
        for k in np.unique(keys):
            d = df[keys == k]
            Xk = self._design(d)
            wet = (d["y"] >= self.threshold).to_numpy(float)
            b_occ, _ = _ols(Xk, wet)
            w = wet.astype(bool)
            if w.sum() < len(self.predictors) + 2:   # too few wet days: climatology only
                b_amt = np.r_[np.mean(d["y"][w] ** self.transform_power) if w.any() else 0, np.zeros(Xk.shape[1])]
                se = float(np.std(d["y"][w] ** self.transform_power)) if w.any() else 0.0
            else:
                b_amt, se = _ols(Xk[w], d["y"].to_numpy()[w] ** self.transform_power)
            self.params_[k] = {"occ": b_occ, "amt": b_amt, "se": se}
        if tune_bias:
            sim = self.generate(X.loc[df.index], n_ensemble=20, seed=seed)
            obs_wet = df["y"][df["y"] >= self.threshold].mean()
            sim_wet = sim.where(sim >= self.threshold).stack().mean()
            if sim_wet > 0:
                self.bias_correction *= obs_wet / sim_wet
        return self

    def wet_probability(self, X: pd.DataFrame) -> pd.Series:
        keys = group_key(X.index, self.grouping)
        A = np.column_stack([np.ones(len(X)), self._design(X)])
        p = np.full(len(X), np.nan)
        for k, prm in self.params_.items():
            m = keys == k
            p[m] = A[m] @ prm["occ"]
        return pd.Series(np.clip(p, 0, 1), index=X.index)

    def generate(self, X: pd.DataFrame, n_ensemble: int = 20, seed: int = 0) -> pd.DataFrame:
        """Return an ensemble (columns e00..) of daily rainfall series for predictors X."""
        rng = np.random.default_rng(seed)
        X = X[self.predictors].dropna()
        keys = group_key(X.index, self.grouping)
        A = np.column_stack([np.ones(len(X)), self._design(X)])
        p = self.wet_probability(X).to_numpy()
        mu = np.zeros(len(X)); se = np.zeros(len(X))
        for k, prm in self.params_.items():
            m = keys == k
            mu[m] = A[m] @ prm["amt"]; se[m] = prm["se"]
        out = {}
        for e in range(n_ensemble):
            wet = rng.random(len(X)) <= p
            z = mu + rng.standard_normal(len(X)) * se * np.sqrt(self.variance_inflation)
            amt = np.clip(z, 0, None) ** (1 / self.transform_power) * self.bias_correction
            amt = np.maximum(amt, self.threshold)  # a generated wet day is at least the threshold
            out[f"e{e:02d}"] = np.where(wet, amt, 0.0)
        return pd.DataFrame(out, index=X.index)

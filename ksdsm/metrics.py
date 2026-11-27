"""Goodness-of-fit metrics used in the paper: r, R², RMSE and Nash-Sutcliffe efficiency."""
import numpy as np


def _clean(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def rmse(obs, sim):
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((o - s) ** 2)))


def pearson_r(obs, sim):
    o, s = _clean(obs, sim)
    if o.std() == 0 or s.std() == 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1])


def r2(obs, sim):
    """Coefficient of determination of the linear regression of obs on sim (= r²)."""
    r = pearson_r(obs, sim)
    return r * r


def nse(obs, sim):
    """Nash-Sutcliffe efficiency: 1 is perfect, 0 equals the observed mean, < 0 is worse."""
    o, s = _clean(obs, sim)
    den = np.sum((o - o.mean()) ** 2)
    return float("nan") if den == 0 else float(1 - np.sum((o - s) ** 2) / den)


def all_metrics(obs, sim):
    return {"r": pearson_r(obs, sim), "r2": r2(obs, sim), "rmse": rmse(obs, sim), "nse": nse(obs, sim)}

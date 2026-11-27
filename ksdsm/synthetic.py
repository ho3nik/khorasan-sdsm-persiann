"""Synthetic test data so the whole pipeline runs without the real (non-public) inputs.

Creates daily predictors with a seasonal cycle and AR(1) noise for an 'NCEP' record
(1961-2005), a 'GCM' historical record (1961-2005, 365-day calendar) and GCM scenarios
(2006-2050) with a drift, plus a rainfall predictand at a few sites generated from a
known logistic occurrence / gamma amount model. Nothing here is real Khorasan data.
"""
import numpy as np
import pandas as pd

from .io import daily_index, write_predictor_dir, write_predictand

VARS = ["mslpgl", "p500gl", "shumgl", "p_zgl", "tempgl"]


def _predictors(index, rng, drift=0.0, bias=0.0):
    doy = index.dayofyear.to_numpy()
    years = (index.year - index.year[0]).to_numpy()
    out = {}
    for j, v in enumerate(VARS):
        phase = 2 * np.pi * (doy / 365.25 + j / len(VARS))
        e = np.zeros(len(index)); eps = rng.standard_normal(len(index))
        for t in range(1, len(index)):
            e[t] = 0.7 * e[t - 1] + eps[t] * np.sqrt(1 - 0.49)
        out[v] = np.sin(phase) + e + drift * years / 45 * (1 if j % 2 else -1) + bias
    return pd.DataFrame(out, index=index)


def rainfall(X, rng, scale=1.0):
    """Wet if logistic(-1 + 1.2 mslp' - 0.8 p500 + 0.6 shum) > U; amount gamma(0.7, 6*exp(.3 shum))."""
    lin = -1.0 - 1.2 * X["mslpgl"] + 0.8 * X["shumgl"] - 0.5 * X["p500gl"]
    p = 1 / (1 + np.exp(-lin))
    wet = rng.random(len(X)) < p
    amt = rng.gamma(0.7, 6 * scale * np.exp(0.3 * X["shumgl"]))
    return pd.Series(np.where(wet, np.maximum(amt, 0.3), 0.0), index=X.index)


def make_demo(root, n_sites=6, seed=1):
    rng = np.random.default_rng(seed)
    root = __import__("pathlib").Path(root)
    sites = pd.DataFrame({"site": [f"S{i+1}" for i in range(n_sites)],
                          "lat": rng.uniform(34.2, 37.3, n_sites).round(2),
                          "lon": rng.uniform(57.0, 60.8, n_sites).round(2)})
    sites.to_csv(root / "sites.csv", index=False)
    rain = {}
    for i, s in enumerate(sites.site):
        nidx = daily_index("1961-01-01", 45 * 365 + 11, "standard")
        ncep = _predictors(nidx, rng)
        write_predictor_dir(ncep, root / "predictors" / s / "ncep", "ncep")
        gidx = daily_index("1961-01-01", 45 * 365, "365_day")
        write_predictor_dir(_predictors(gidx, rng, bias=0.1), root / "predictors" / s / "historical", "canesm2")
        for k, sc in enumerate(["rcp26", "rcp45", "rcp85"]):
            fidx = daily_index("2006-01-01", 45 * 365, "365_day")
            write_predictor_dir(_predictors(fidx, rng, drift=0.3 * (k + 1)), root / "predictors" / s / sc, "canesm2")
        r = rainfall(ncep, rng, scale=0.7 + 0.1 * i).loc["1975":"2005"]
        r[rng.random(len(r)) < 0.01] = np.nan       # 1% missing days
        rain[s] = r
    write_predictand(pd.DataFrame(rain), root / "predictand.csv")
    return root

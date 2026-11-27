import numpy as np
import pandas as pd
import xarray as xr

from ksdsm import metrics, stats, io
from ksdsm.maps import idw
from ksdsm.model import SDSMPrecip, Standardizer
from ksdsm.screening import screen, select_predictors
from ksdsm.persiann import crop, select_cells, extract_cells


def test_metrics_known_values():
    o = np.array([1.0, 2, 3, 4]); s = o.copy()
    assert metrics.nse(o, s) == 1 and metrics.rmse(o, s) == 0
    assert abs(metrics.nse(o, np.full(4, o.mean()))) < 1e-12      # mean predictor -> NSE 0
    assert metrics.nse(o, o[::-1]) < 0
    assert abs(metrics.pearson_r(o, 2 * o + 1) - 1) < 1e-12


def test_spells_and_wetdays():
    idx = pd.date_range("2000-01-01", periods=10)
    s = pd.Series([0, 0, 1, 1, 1, 0, 1, 0, 0, 0], index=idx, dtype=float)
    t = stats.monthly_stats(s, threshold=0.5)
    assert t.loc[1, "wetday"] == 0.4
    assert t.loc[1, "wetspell"] == 2.0          # spells of 3 and 1
    assert abs(t.loc[1, "dryspell"] - 2.0) < 1e-12  # spells of 2, 1, 3
    assert t.loc[1, "sum"] == 4.0


def test_365_day_calendar_skips_feb29():
    idx = io.daily_index("2000-02-27", 4, "365_day")
    assert list(idx.strftime("%m-%d")) == ["02-27", "02-28", "03-01", "03-02"]


def _toy(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-01", periods=n)
    X = pd.DataFrame(rng.standard_normal((n, 3)), index=idx, columns=["a", "b", "noise"])
    p = np.clip(0.3 + 0.2 * X.a, 0.02, 0.98)
    wet = rng.random(n) < p
    amt = (1.2 + 0.3 * X.b + 0.2 * rng.standard_normal(n)).clip(0.8) ** 4
    return X, pd.Series(np.where(wet, amt, 0.0), index=idx)


def test_model_recovers_occurrence_and_amounts():
    X, y = _toy()
    m = SDSMPrecip(["a", "b"], grouping="annual").fit(y, X)
    occ = m.params_["ALL"]["occ"]
    assert abs(occ[0] - 0.3) < 0.03 and abs(occ[1] - 0.2) < 0.03
    sim = m.generate(X, n_ensemble=10, seed=1)
    assert abs((sim >= 0.3).mean().mean() - (y >= 0.3).mean()) < 0.02
    assert abs(sim.mean().mean() / y.mean() - 1) < 0.1


def test_screening_ranks_real_predictors_first():
    X, y = _toy()
    chosen = select_predictors(screen(y, X, "annual"), k=2)
    assert set(chosen) == {"a", "b"}


def test_standardizer():
    X = pd.DataFrame({"v": np.arange(100.0)}, index=pd.date_range("1961-01-01", periods=100))
    Z = Standardizer().fit(X).transform(X)
    assert abs(Z.v.mean()) < 1e-12 and abs(Z.v.std(ddof=0) - 1) < 1e-12


def test_idw_exact_at_sites():
    z = idw([35, 36], [58, 59], [1.0, 3.0], np.array([35.0, 36.0]), np.array([58.0, 59.0]))
    assert z[0, 0] == 1.0 and z[1, 1] == 3.0 and 1 < z[0, 1] < 3


def test_persiann_helpers_on_fake_grid():
    t = pd.date_range("1983-01-01", periods=5)
    lat = np.arange(30, 40, 0.25); lon = np.arange(54, 64, 0.25) % 360
    da = xr.DataArray(np.random.default_rng(0).gamma(0.5, 2, (5, len(lon), len(lat))),
                      coords=dict(time=t, lon=lon, lat=lat), dims=("time", "lon", "lat"), name="precipitation")
    c = crop(da.to_dataset())["precipitation"].transpose("time", "lat", "lon")
    assert float(c.lat.min()) >= 33.5 and float(c.lon.max()) <= 61.5
    cells = select_cells(c, 5)
    assert len(cells) == 5 and cells[["lat", "lon"]].drop_duplicates().shape[0] == 5
    df = extract_cells(c, cells)
    assert df.shape == (5, 5)

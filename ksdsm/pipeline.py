"""End-to-end workflow: screen -> calibrate -> validate (NCEP) -> base period (GCM) ->
future scenarios -> metrics tables and IDW maps. Driven by a TOML config file."""
import tomllib
from pathlib import Path
import numpy as np
import pandas as pd

from . import io, metrics, stats
from .model import SDSMPrecip, Standardizer
from .screening import screen, select_predictors

STATS = ["mean", "max", "sum", "dryspell", "wetspell", "wetday"]


def _yrs(df, span):
    return df.loc[f"{span[0]}-01-01":f"{span[1]}-12-31"]


def load_config(path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    base = Path(path).parent
    for k in ("predictand", "sites", "predictor_root", "out_dir"):
        if k in cfg and not Path(cfg[k]).is_absolute():
            cfg[k] = str((base / cfg[k]).resolve())
    return cfg


def load_site_predictors(cfg, site):
    root = Path(cfg["predictor_root"]) / site
    sb = cfg.get("standardize_base", [1961, 1990])
    ncep = io.read_predictor_dir(root / "ncep", cfg.get("ncep_start", "1961-01-01"))
    ncep = Standardizer().fit(ncep, sb[0], sb[1]).transform(ncep)
    cal = cfg.get("gcm_calendar", "365_day")
    hist = io.read_predictor_dir(root / "historical", cfg.get("gcm_hist_start", "1961-01-01"), cal)
    std = Standardizer().fit(hist, sb[0], sb[1])   # GCM standardised against its own baseline
    gcm = {"historical": std.transform(hist)}
    for sc in cfg.get("scenarios", []):
        fut = io.read_predictor_dir(root / sc, cfg.get("gcm_future_start", "2006-01-01"), cal)
        gcm[sc] = std.transform(fut)
    return ncep, gcm


def compare(obs: pd.Series, ens: pd.DataFrame, threshold) -> list:
    """Rows of metrics for each statistic. Monthly statistics are compared over their 12
    calendar-month values; annual totals over years (ensemble-mean series)."""
    so = stats.monthly_stats(obs, threshold)
    ss = stats.ensemble_monthly_stats(ens, threshold)
    rows = [dict(statistic=s, **metrics.all_metrics(so[s], ss[s])) for s in STATS]
    ao = stats.annual_totals(obs)
    asim = stats.annual_totals(ens.mean(axis=1))
    j = ao.index.intersection(asim.index)
    rows.append(dict(statistic="annual_total", **metrics.all_metrics(ao[j], asim[j])))
    return rows


def run(cfg: dict) -> dict:
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    y_all = io.read_predictand(cfg["predictand"])
    P = cfg["periods"]; M = cfg.get("model", {})
    thr = M.get("threshold", stats.WET_THRESHOLD)
    n_ens = M.get("n_ensemble", 20)
    met_rows, chg_rows, pred_rows, screens = [], [], [], []
    for i, site in enumerate(y_all.columns):
        y = y_all[site]
        ncep, gcm = load_site_predictors(cfg, site)
        mo = {**M, **M.get("overrides", {}).get(site, {})}
        y_cal, X_cal = _yrs(y, P["calibration"]), _yrs(ncep, P["calibration"])
        if mo.get("predictors"):
            preds = list(mo["predictors"])
        else:
            tab = screen(y_cal, X_cal, mo.get("grouping", "season"), thr)
            tab.insert(0, "site", site); screens.append(tab)
            preds = select_predictors(tab, mo.get("n_predictors", 4))
        pred_rows.append(dict(site=site, predictors=" ".join(preds)))
        model = SDSMPrecip(preds, grouping=mo.get("grouping", "season"), threshold=thr,
                           variance_inflation=mo.get("variance_inflation", 1.0),
                           bias_correction=mo.get("bias_correction", 1.0))
        model.fit(y_cal, X_cal, tune_bias=mo.get("tune_bias", False), seed=i)
        # 1) validation with reanalysis predictors (years outside calibration)
        val = model.generate(_yrs(ncep, P["validation"]), n_ens, seed=100 + i)
        for r in compare(_yrs(y, P["validation"]), val, thr):
            met_rows.append(dict(site=site, stage="validation_ncep", **r))
        # 2) base period with GCM historical predictors
        hist = model.generate(_yrs(gcm["historical"], P["base"]), n_ens, seed=200 + i)
        y_base = _yrs(y, P["base"])
        for r in compare(y_base, hist, thr):
            met_rows.append(dict(site=site, stage="base_gcm", **r))
        # 3) future scenarios: change in mean annual total vs observed base period
        obs_mean = stats.annual_totals(y_base).mean()
        for sc in cfg.get("scenarios", []):
            Xf = pd.concat([gcm["historical"], gcm[sc]]).sort_index()
            fut = model.generate(_yrs(Xf, P["future"]), n_ens, seed=300 + i)
            fm = stats.annual_totals(fut.mean(axis=1)).mean()
            chg_rows.append(dict(site=site, scenario=sc, base_mm=obs_mean, future_mm=fm,
                                 change_mm=fm - obs_mean, change_pct=100 * (fm - obs_mean) / obs_mean))
    met = pd.DataFrame(met_rows); chg = pd.DataFrame(chg_rows)
    met.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    chg.to_csv(out / "changes.csv", index=False, float_format="%.3f")
    pd.DataFrame(pred_rows).to_csv(out / "predictors.csv", index=False)
    if screens:
        pd.concat(screens).to_csv(out / "screening.csv", index=False, float_format="%.4f")
    if cfg.get("sites") and len(y_all.columns) >= 3 and cfg.get("maps", True):
        _maps(cfg, met, chg, out)
    return {"metrics": met, "changes": chg}


def _maps(cfg, met, chg, out):
    from .maps import plot_idw
    sites = io.read_sites(cfg["sites"])
    d = out / "maps"; d.mkdir(exist_ok=True)
    for (stage, st), g in met.groupby(["stage", "statistic"]):
        for m in ("r", "r2", "rmse", "nse"):
            plot_idw(sites, g.set_index("site")[m], f"{stage} {st} {m}", d / f"{stage}_{st}_{m}.png")
    for sc, g in chg.groupby("scenario"):
        plot_idw(sites, g.set_index("site")["change_mm"], f"Change in mean annual rainfall (mm), {sc}",
                 d / f"change_{sc}.png", cmap="RdBu")


def summarise(met: pd.DataFrame) -> pd.DataFrame:
    """Share of sites with NSE < 0 and > 0.6, and mean RMSE, per stage and statistic."""
    g = met.groupby(["stage", "statistic"])
    return pd.DataFrame({"mean_rmse": g.rmse.mean(), "mean_r": g.r.mean(),
                         "frac_nse_neg": g.nse.apply(lambda x: (x < 0).mean()),
                         "frac_nse_gt_0.6": g.nse.apply(lambda x: (x > 0.6).mean())})

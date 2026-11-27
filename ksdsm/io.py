"""Readers and writers for predictand and predictor files.

Predictand CSV: a `date` column (YYYY-MM-DD) plus one column per site (gauge or
PERSIANN-CDR cell), daily rainfall in mm; -99 or empty means missing (as in SDSM).
A sites CSV gives `site,lat,lon` for mapping.

Predictors: SDSM-format `.dat` files, one value per line, one file per variable, as
distributed per grid box by the Canadian climate scenarios service for NCEP/NCAR and
CanESM2 (e.g. `ncepmslpgl.dat`, `canesm2mslpgl.dat`). The model prefix is stripped so
NCEP and GCM files share variable names (`mslpgl`). CanESM2 uses a 365-day calendar.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

MISSING = -99.0
_PREFIX = re.compile(r"^(ncep|canesm2|can|cgcm3|h3|had)", re.I)


def read_predictand(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.astype(float).where(lambda x: (x != MISSING) & (x >= 0))
    return df


def write_predictand(df: pd.DataFrame, path):
    df.fillna(MISSING).rename_axis("date").to_csv(path, float_format="%.2f")


def read_sites(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("site")


def daily_index(start: str, n: int, calendar: str = "standard") -> pd.DatetimeIndex:
    """n consecutive days from start; for a 365-day calendar, 29 Feb is skipped."""
    if calendar == "standard":
        return pd.date_range(start, periods=n, freq="D")
    days = pd.date_range(start, periods=int(n * 1.01) + 400, freq="D")
    days = days[~((days.month == 2) & (days.day == 29))]
    return days[:n]


def read_predictor_dir(directory, start: str, calendar: str = "standard") -> pd.DataFrame:
    """Read every *.dat file in a directory into one DataFrame of daily predictors."""
    cols = {}
    for f in sorted(Path(directory).glob("*.dat")):
        name = _PREFIX.sub("", f.stem).lower()
        cols[name] = np.loadtxt(f, dtype=float).ravel()
    if not cols:
        raise FileNotFoundError(f"no .dat predictor files in {directory}")
    n = min(len(v) for v in cols.values())
    idx = daily_index(start, n, calendar)
    return pd.DataFrame({k: v[:n] for k, v in cols.items()}, index=idx).replace(MISSING, np.nan)


def write_predictor_dir(df: pd.DataFrame, directory, prefix: str):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    for c in df.columns:
        np.savetxt(d / f"{prefix}{c}.dat", df[c].to_numpy(), fmt="%.4f")


def long_to_wide(path, site_col="station", date_col="date", value_col="rain") -> pd.DataFrame:
    """Convert a long gauge table (one row per station-day) into the predictand layout."""
    df = pd.read_csv(path, parse_dates=[date_col])
    wide = df.pivot_table(index=date_col, columns=site_col, values=value_col, aggfunc="first")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full).rename_axis("date")

"""PERSIANN-CDR download, subsetting and cell selection.

PERSIANN-CDR (Ashouri et al., 2015) daily 0.25-degree files are served by NOAA NCEI at
https://www.ncei.noaa.gov/data/precipitation-persiann/access/<YYYY>/ (one NetCDF per
day). This module lists each year's directory, downloads the daily files, crops them to
a bounding box and saves one NetCDF per year. CHRS (https://chrsdata.eng.uci.edu/) is an
alternative source with a web interface for regional subsets.

NOTE: written in 2026 against the documented NCEI layout; the download step was not
run end to end while preparing this repository (the build machine could not reach
ncei.noaa.gov). Check the URL pattern if NCEI has changed it.
"""
import re
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

NCEI = "https://www.ncei.noaa.gov/data/precipitation-persiann/access/{year}/"
# Khorasan Razavi, with a small margin
KHORASAN_BBOX = dict(lat_min=33.5, lat_max=38.0, lon_min=56.0, lon_max=61.5)


def list_files(year: int) -> list:
    url = NCEI.format(year=year)
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode()
    names = sorted(set(re.findall(r'PERSIANN-CDR_v\d+r\d+_\d{8}_c\d{8}\.nc', html)))
    return [url + n for n in names]


def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    ren = {c: c.lower()[:3] for c in ds.coords if c.lower() in ("latitude", "longitude")}
    ds = ds.rename(ren) if ren else ds
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    return ds.sortby("lat")


def crop(ds: xr.Dataset, bbox=KHORASAN_BBOX) -> xr.Dataset:
    ds = _std_coords(ds)
    return ds.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(bbox["lon_min"], bbox["lon_max"]))


def download_year(year: int, out_dir, bbox=KHORASAN_BBOX, keep_daily: bool = False) -> Path:
    """Download one year of daily files, crop, and write <out_dir>/persiann_cdr_<year>.nc."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"daily_{year}"; tmp.mkdir(exist_ok=True)
    parts = []
    for url in list_files(year):
        f = tmp / url.rsplit("/", 1)[1]
        if not f.exists():
            urllib.request.urlretrieve(url, f)
        with xr.open_dataset(f) as ds:
            parts.append(crop(ds[["precipitation"]], bbox).load())
    out = out_dir / f"persiann_cdr_{year}.nc"
    xr.concat(parts, dim="time").to_netcdf(out)
    if not keep_daily:
        for f in tmp.glob("*.nc"):
            f.unlink()
        tmp.rmdir()
    return out


def load_years(directory, years) -> xr.DataArray:
    files = [Path(directory) / f"persiann_cdr_{y}.nc" for y in years]
    da = xr.open_mfdataset(files, combine="by_coords")["precipitation"]
    da = _std_coords(da.to_dataset())["precipitation"]
    return da.where(da >= 0)


def select_cells(da: xr.DataArray, n: int, min_valid: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Choose n cells with >= min_valid fraction of valid days, spread out by
    farthest-point sampling. Returns site,lat,lon."""
    valid = da.notnull().mean("time").load()
    if "lat" in valid.dims and valid.dims[0] != "lat":
        valid = valid.transpose("lat", "lon")
    la, lo = np.meshgrid(valid["lat"].values, valid["lon"].values, indexing="ij")
    ok = valid.values >= min_valid
    pts = np.column_stack([la[ok], lo[ok]])
    if len(pts) <= n:
        chosen = list(range(len(pts)))
    else:
        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(len(pts)))]
        d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
        while len(chosen) < n:
            i = int(np.argmax(d)); chosen.append(i)
            d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    sel = pts[chosen]
    return pd.DataFrame({"site": [f"cell{i+1:02d}" for i in range(len(sel))],
                         "lat": sel[:, 0], "lon": sel[:, 1]})


def extract_cells(da: xr.DataArray, cells: pd.DataFrame) -> pd.DataFrame:
    """Daily series at each selected cell, as a predictand DataFrame (date x site)."""
    out = {}
    for _, c in cells.iterrows():
        out[c.site] = da.sel(lat=c.lat, lon=c.lon, method="nearest").to_series()
    df = pd.DataFrame(out)
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df.rename_axis("date")

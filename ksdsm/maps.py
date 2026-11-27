"""Inverse-distance-weighted (IDW) maps of per-site scores (the paper used QGIS IDW)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def idw(lat, lon, values, grid_lat, grid_lon, power: float = 2.0):
    lat, lon, values = map(np.asarray, (lat, lon, values))
    ok = np.isfinite(values)
    lat, lon, values = lat[ok], lon[ok], values[ok]
    GL, GO = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    d = np.sqrt((GL[..., None] - lat) ** 2 + ((GO[..., None] - lon) * np.cos(np.radians(GL[..., None]))) ** 2)
    exact = d < 1e-9
    w = 1.0 / np.maximum(d, 1e-9) ** power
    z = (w * values).sum(-1) / w.sum(-1)
    hit = exact.any(-1)
    z[hit] = values[exact[hit].argmax(-1)]
    return z


def plot_idw(sites, values, title, out_path, bbox=None, res=0.05, cmap="RdBu", boundary=None):
    """sites: DataFrame with lat/lon indexed like values. boundary: optional (lon, lat) arrays."""
    bbox = bbox or dict(lat_min=sites.lat.min() - 0.5, lat_max=sites.lat.max() + 0.5,
                        lon_min=sites.lon.min() - 0.5, lon_max=sites.lon.max() + 0.5)
    glat = np.arange(bbox["lat_min"], bbox["lat_max"] + res, res)
    glon = np.arange(bbox["lon_min"], bbox["lon_max"] + res, res)
    v = values.reindex(sites.index)
    z = idw(sites.lat, sites.lon, v, glat, glon)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    m = ax.pcolormesh(glon, glat, z, cmap=cmap, shading="auto")
    ax.scatter(sites.lon, sites.lat, s=8, c="k")
    if boundary is not None:
        ax.plot(*boundary, "k-", lw=0.8)
    fig.colorbar(m, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=9); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(glat))))
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

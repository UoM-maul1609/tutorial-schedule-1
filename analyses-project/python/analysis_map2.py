#!/usr/bin/env python3
"""
Global ERA5 TCC maps with coastlines

Produces three figures:
  1) Mean Total Cloud Cover (TCC) over ANALYSIS period
  2) El Niño − La Niña composite of TCC anomalies (baseline CLIM period)
  3) Correlation map: ONI vs TCC anomalies

Inputs (local):
  - NetCDFs in DATA_DIR named like: era5_single_levels_monthly_YYYYMM.nc
    * variables: TCC (tcc or total_cloud_cover), latitude, longitude
    * time: valid_time (or time) in seconds since 1970-01-01
    * optional: land_sea_mask (or lsm) for pseudo-coastline fallback
  - ONI_CSV: two columns (Date, ONI). Duplicate headers ok; -9999 treated as missing.

If Cartopy is installed, coastlines/borders/gridlines are drawn.
Otherwise, a fallback "coastline" is contoured from the ERA5 land-sea mask (if present).

Edit the CONFIG block below to change year ranges and file paths.
"""

import glob, os, numpy as np, pandas as pd
from pathlib import Path
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# =========================
# CONFIG — edit these only
# =========================
ANALYSIS_START = 2005
ANALYSIS_END   = 2025

# Baseline for monthly climatology (used for anomalies)
CLIM_START     = 2010
CLIM_END       = 2019

# ENSO threshold (°C) for El Niño / La Niña masks
ENSO_THRESH    = 0.5

# Data location
DATA_DIR       = Path("era5_monthly_nc")
ONI_CSV        = "oni.csv"

# Candidate names in the NetCDFs
VAR_CANDS      = ["tcc", "total_cloud_cover"]
LAT_CANDS      = ["latitude", "lat"]
LON_CANDS      = ["longitude", "lon"]
TIME_CANDS     = ["valid_time", "time"]
LSM_CANDS      = ["land_sea_mask", "lsm"]  # optional for pseudo-coastline

# =========================
# Helpers
# =========================
def pick_var(nc, cands):
    for v in cands:
        if v in nc.variables:
            return v
    return None

def to_lon180(lon):
    lon180 = (lon + 180.0) % 360.0 - 180.0
    order = np.argsort(lon180)
    return lon180[order], order

def get_time(nc, tname):
    tsec = np.asarray(nc.variables[tname][:], dtype="int64")
    return pd.to_datetime(tsec, unit="s", utc=True).tz_localize(None)

def try_import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None

def plot_map(data2d, lon, lat, title, cmap, vmin=None, vmax=None, cb_label="",
             fname="map.png", lsm2d=None):
    """
    Plot with Cartopy if available; otherwise fallback with optional LSM pseudo-coastline.
    data2d shape: [lat, lon] with lon in -180..180 ascending and lat ascending.
    """
    Lon2, Lat2 = np.meshgrid(lon, lat)
    ccrs, cfeature = try_import_cartopy()

    if ccrs is not None:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 4.6))
        ax = plt.axes(projection=proj)
        im = ax.pcolormesh(Lon2, Lat2, data2d, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax)
        # coastlines/borders/gridlines
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, alpha=0.4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="k", alpha=0.25, linestyle="--")
        gl.top_labels = gl.right_labels = False
        ax.set_global()
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)
        cb.set_label(cb_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=140)
        plt.close(fig)
        return

    # Fallback (no Cartopy): draw a pseudo-coastline if LSM provided
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.6))
    im = ax.pcolormesh(Lon2, Lat2, data2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if lsm2d is not None:
        try:
            ax.contour(Lon2, Lat2, lsm2d, levels=[0.5], colors="k", linewidths=0.5)
        except Exception:
            pass
    ax.set_xlim([-180, 180]); ax.set_ylim([lat.min(), lat.max()])
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08)
    cb.set_label(cb_label)
    ax.set_title(title + " (fallback: pseudo-coastline from LSM)")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close(fig)

def corr_along_time(x, ytx):
    """
    Pearson r per grid cell.
    x   : [T]      ONI (float)
    ytx : [T,Y,X]  TCC anomalies
    returns rmap: [Y,X]
    """
    x = np.asarray(x, dtype=float)
    T, NY, NX = ytx.shape
    r = np.full((NY, NX), np.nan)
    for j in range(NY):
        for i in range(NX):
            y = ytx[:, j, i].astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 3:
                continue
            xi = x[mask] - np.nanmean(x[mask])
            yi = y[mask] - np.nanmean(y[mask])
            sx = np.nanstd(xi, ddof=1)
            sy = np.nanstd(yi, ddof=1)
            if sx == 0 or sy == 0:
                continue
            r[j, i] = np.nansum(xi * yi) / ((n - 1) * sx * sy)
    return r

# =========================
# Gather files & read grid
# =========================
patterns = [f"era5_single_levels_monthly_{y}*.nc" for y in range(ANALYSIS_START, ANALYSIS_END + 1)]
files = sorted({f for pat in patterns for f in glob.glob(str(DATA_DIR / pat))})
if not files:
    raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR} for {ANALYSIS_START}-{ANALYSIS_END}")

with Dataset(files[0]) as nc0:
    vlat = pick_var(nc0, LAT_CANDS); vlon = pick_var(nc0, LON_CANDS)
    if vlat is None or vlon is None:
        raise KeyError("latitude/longitude not found in the first file.")
    lat = np.array(nc0.variables[vlat][:], dtype=float)
    lon = np.array(nc0.variables[vlon][:], dtype=float)

    # Ensure lat ascending for plotting consistency
    flip_lat = False
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip_lat = True

    lon180, order_lon = to_lon180(lon)

    # Optional land-sea mask for pseudo-coastlines
    vlsm = pick_var(nc0, LSM_CANDS)
    lsm180 = None
    if vlsm is not None:
        lsm = np.array(nc0.variables[vlsm][...], dtype=float)
        if lsm.ndim == 3:  # some files store a singleton time dimension
            lsm = lsm[0, ...]
        if flip_lat:
            lsm = lsm[::-1, :]
        lsm180 = lsm[:, order_lon]

# =========================
# Read TCC time series
# =========================
times = []
tcc_list = []

for f in files:
    with Dataset(f) as nc:
        tname = pick_var(nc, TIME_CANDS)
        vname = pick_var(nc, VAR_CANDS)
        if tname is None or vname is None:
            continue

        tt = get_time(nc, tname)
        arr = np.array(nc.variables[vname][...], dtype=float)  # (t,y,x) or (y,x)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if flip_lat:
            arr = arr[:, ::-1, :]
        arr = arr[:, :, order_lon]

        times.extend(list(tt))
        tcc_list.append(arr)

if not tcc_list:
    raise RuntimeError("No TCC data found in the files.")
tcc = np.concatenate(tcc_list, axis=0)  # [T, Y, X]
time_index = pd.to_datetime(times)

# keep only requested analysis window (robust if extras slipped in)
mask_time = (time_index.year >= ANALYSIS_START) & (time_index.year <= ANALYSIS_END)
tcc = tcc[mask_time, :, :]
time_index = time_index[mask_time]

# =========================
# Climatology & anomalies
# =========================
df_time = pd.DataFrame({"y": time_index.year, "m": time_index.month})
mask_clim = (df_time["y"] >= CLIM_START) & (df_time["y"] <= CLIM_END)

# monthly climatology
clim = np.full((12, tcc.shape[1], tcc.shape[2]), np.nan, dtype=float)
for m in range(1, 13):
    sel = (df_time["m"].values == m) & mask_clim.values
    if np.any(sel):
        clim[m - 1] = np.nanmean(tcc[sel, :, :], axis=0)

# anomalies (TCC minus month-of-year climatology)
anom = np.empty_like(tcc)
for i, m in enumerate(df_time["m"].values):
    anom[i] = tcc[i] - clim[m - 1]

# =========================
# ONI & ENSO masks
# =========================
oni_raw = pd.read_csv(ONI_CSV, header=None, names=["date", "oni"])
oni_raw["date"] = pd.to_datetime(oni_raw["date"], errors="coerce")
oni_raw["oni"]  = pd.to_numeric(oni_raw["oni"], errors="coerce")
oni = oni_raw.dropna(subset=["date"]).copy()
oni.loc[oni["oni"] <= -99, "oni"] = np.nan  # treat sentinels as missing
oni["year"] = oni["date"].dt.year
oni["month"] = oni["date"].dt.month
oni = oni.set_index(["year", "month"]).sort_index()

# align ONI to model months
key = list(zip(df_time["y"].tolist(), df_time["m"].tolist()))
oni_aligned = oni.reindex(key)["oni"].to_numpy()
elnino_mask = (oni_aligned >=  ENSO_THRESH)
lanina_mask = (oni_aligned <= -ENSO_THRESH)

# =========================
# (1) Mean TCC map
# =========================
tcc_mean = np.nanmean(tcc, axis=0)
plot_map(
    data2d=tcc_mean,
    lon=lon180,
    lat=lat,
    title=f"ERA5 Total Cloud Cover — Mean {ANALYSIS_START}–{ANALYSIS_END}",
    cmap="Blues",
    vmin=0.0, vmax=1.0,
    cb_label="TCC (0–1)",
    fname=f"tcc_mean_{ANALYSIS_START}_{ANALYSIS_END}.png",
    lsm2d=lsm180
)
print(f"Saved tcc_mean_{ANALYSIS_START}_{ANALYSIS_END}.png")

# =========================
# (2) El Niño − La Niña composite (anomalies)
# =========================
eln = np.nanmean(anom[elnino_mask, :, :], axis=0) if np.any(elnino_mask) else np.full_like(tcc_mean, np.nan)
lan = np.nanmean(anom[lanina_mask, :, :], axis=0) if np.any(lanina_mask) else np.full_like(tcc_mean, np.nan)
comp = eln - lan

# symmetric color range (guarantee at least ±0.1)
if np.isfinite(comp).any():
    vmax = max(0.1, float(np.nanmax(np.abs(comp))))
else:
    vmax = 0.1

plot_map(
    data2d=comp,
    lon=lon180,
    lat=lat,
    title=f"TCC anomaly: El Niño − La Niña ({ANALYSIS_START}–{ANALYSIS_END}, baseline {CLIM_START}–{CLIM_END})",
    cmap="RdBu_r",
    vmin=-vmax, vmax=+vmax,
    cb_label="ΔTCC (0–1)",
    fname=f"tcc_elnino_minus_lanina_{ANALYSIS_START}_{ANALYSIS_END}.png",
    lsm2d=lsm180
)
print(f"Saved tcc_elnino_minus_lanina_{ANALYSIS_START}_{ANALYSIS_END}.png")

# =========================
# (3) Correlation map: ONI vs TCC anomalies
# =========================
valid_t = np.isfinite(oni_aligned)
Xv = oni_aligned[valid_t]
Yv = anom[valid_t, :, :]
rmap = corr_along_time(Xv, Yv)

plot_map(
    data2d=rmap,
    lon=lon180,
    lat=lat,
    title=f"Correlation: ONI vs TCC anomaly ({ANALYSIS_START}–{ANALYSIS_END})",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    cb_label="Pearson r",
    fname=f"tcc_oni_correlation_{ANALYSIS_START}_{ANALYSIS_END}.png",
    lsm2d=lsm180
)
print(f"Saved tcc_oni_correlation_{ANALYSIS_START}_{ANALYSIS_END}.png")

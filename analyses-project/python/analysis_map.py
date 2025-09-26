#!/usr/bin/env python3
"""
Quick global maps from ERA5 monthly TCC:
  (1) Mean TCC over ANALYSIS period
  (2) El Niño minus La Niña composite of TCC anomalies (vs CLIM baseline)
  (3) Correlation map: ONI vs TCC anomalies

Inputs:
  - NetCDFs in era5_monthly_nc/: era5_single_levels_monthly_YYYYMM.nc
  - oni.csv with two columns: Date, ONI  (can contain -9999 and duplicate headers)

Outputs:
  - tcc_mean_<start>_<end>.png
  - tcc_elnino_minus_lanina_<start>_<end>.png
  - tcc_oni_correlation_<start>_<end>.png
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
CLIM_START     = 2010      # monthly climatology baseline for anomalies
CLIM_END       = 2019
ENSO_THRESH    = 0.5       # ONI threshold for El Niño / La Niña
DATA_DIR       = Path("era5_monthly_nc")
ONI_CSV        = "oni.csv"
VAR_CANDS      = ["tcc", "total_cloud_cover"]
LAT_CANDS      = ["latitude", "lat"]
LON_CANDS      = ["longitude", "lon"]
TIME_CANDS     = ["valid_time", "time"]

# =========================
# Utils
# =========================
def pick_var(nc, cands):
    for v in cands:
        if v in nc.variables: return v
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

def plot_map(data2d, lon, lat, title, cmap, vmin=None, vmax=None, cb_label="", fname="map.png"):
    """
    Plot with Cartopy if available; otherwise simple pcolormesh.
    data2d must be [lat, lon] with lon in -180..180 ascending and lat ascending.
    """
    ccrs, cfeature = try_import_cartopy()
    Lon2, Lat2 = np.meshgrid(lon, lat)

    if ccrs is not None:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 4.6))
        ax = plt.axes(projection=proj)
        im = ax.pcolormesh(Lon2, Lat2, data2d, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4)
        ax.set_global()
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)
        cb.set_label(cb_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=140)
        plt.close(fig)
    else:
        # Fallback: simple geographic image
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.6))
        im = ax.pcolormesh(Lon2, Lat2, data2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlim([-180, 180]); ax.set_ylim([lat.min(), lat.max()])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08)
        cb.set_label(cb_label)
        ax.set_title(title + " (no coastlines—install cartopy for nicer maps)")
        plt.tight_layout()
        plt.savefig(fname, dpi=140)
        plt.close(fig)

# =========================
# Gather files & read grid + time series
# =========================
patterns = [f"era5_single_levels_monthly_{y}*.nc" for y in range(ANALYSIS_START, ANALYSIS_END+1)]
files = sorted({f for pat in patterns for f in glob.glob(str(DATA_DIR / pat))})
if not files:
    raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR} for {ANALYSIS_START}-{ANALYSIS_END}")

# read grid from first file
with Dataset(files[0]) as nc0:
    vlat = pick_var(nc0, LAT_CANDS); vlon = pick_var(nc0, LON_CANDS)
    if vlat is None or vlon is None:
        raise KeyError("latitude/longitude not found.")
    lat = np.array(nc0.variables[vlat][:], dtype=float)
    lon = np.array(nc0.variables[vlon][:], dtype=float)
    # Ensure increasing latitude for pcolormesh convenience
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip_lat = True
    else:
        flip_lat = False
    lon180, order_lon = to_lon180(lon)

# build time list and stacked TCC (time, lat, lon)
times = []
tcc_list = []
for f in files:
    with Dataset(f) as nc:
        tname = pick_var(nc, TIME_CANDS)
        vname = pick_var(nc, VAR_CANDS)
        if tname is None or vname is None:
            continue
        tt = get_time(nc, tname)
        arr = np.array(nc.variables[vname][...], dtype=float)  # (t, y, x) or (y, x) if single time
        if arr.ndim == 2:
            arr = arr[None, ...]
        # reorder latitude if needed
        if flip_lat:
            arr = arr[:, ::-1, :]
        # reorder longitudes to -180..180 ascending
        arr = arr[:, :, order_lon]
        times.extend(list(tt))
        tcc_list.append(arr)

if not tcc_list:
    raise RuntimeError("No TCC data found in the files.")
tcc = np.concatenate(tcc_list, axis=0)  # [T, Y, X]
time_index = pd.to_datetime(times)
# Keep rows within requested analysis window (in case extra months slipped in)
mask_time = (time_index.year >= ANALYSIS_START) & (time_index.year <= ANALYSIS_END)
tcc = tcc[mask_time, :, :]
time_index = time_index[mask_time]

# =========================
# Build monthly climatology and anomalies
# =========================
df_time = pd.DataFrame({"y": time_index.year, "m": time_index.month})
# baseline mask
mask_clim = (df_time["y"] >= CLIM_START) & (df_time["y"] <= CLIM_END)
clim = np.full((12, tcc.shape[1], tcc.shape[2]), np.nan, dtype=float)
for m in range(1, 13):
    sel = (df_time["m"].values == m) & mask_clim.values
    if np.any(sel):
        clim[m-1] = np.nanmean(tcc[sel, :, :], axis=0)

# anomalies (subtract month-of-year climatology)
anom = np.empty_like(tcc)
for i, (y, m) in enumerate(zip(df_time["y"].values, df_time["m"].values)):
    cm = clim[m-1]
    anom[i] = tcc[i] - cm

# =========================
# ONI and ENSO masks
# =========================
oni_raw = pd.read_csv(ONI_CSV, header=None, names=["date", "oni"])
oni_raw["date"] = pd.to_datetime(oni_raw["date"], errors="coerce")
oni_raw["oni"]  = pd.to_numeric(oni_raw["oni"], errors="coerce")
oni = oni_raw.dropna(subset=["date"]).copy()
oni.loc[oni["oni"] <= -99, "oni"] = np.nan
oni["year"] = oni["date"].dt.year
oni["month"] = oni["date"].dt.month
oni = oni.set_index(["year","month"]).sort_index()

# align ONI with model months
oni_aligned = oni.reindex(list(zip(df_time["y"], df_time["m"])))["oni"].to_numpy()

elnino_mask = (oni_aligned >=  ENSO_THRESH)
lanina_mask = (oni_aligned <= -ENSO_THRESH)

# =========================
# (1) Mean TCC over analysis period
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
    fname=f"tcc_mean_{ANALYSIS_START}_{ANALYSIS_END}.png"
)
print(f"Saved tcc_mean_{ANALYSIS_START}_{ANALYSIS_END}.png")

# =========================
# (2) El Niño minus La Niña composite (anomalies)
# =========================
eln = np.nanmean(anom[elnino_mask, :, :], axis=0) if np.any(elnino_mask) else np.full_like(tcc_mean, np.nan)
lan = np.nanmean(anom[lanina_mask, :, :], axis=0) if np.any(lanina_mask) else np.full_like(tcc_mean, np.nan)
comp = eln - lan
# nice symmetric range (±0.15 usually OK for TCC anomalies)
v = np.nanmax(np.abs(comp[np.isfinite(comp)])) if np.isfinite(comp).any() else 0.1
v = max(v, 0.1)
plot_map(
    data2d=comp,
    lon=lon180,
    lat=lat,
    title=f"TCC anomaly: El Niño − La Niña ({ANALYSIS_START}–{ANALYSIS_END}, baseline {CLIM_START}–{CLIM_END})",
    cmap="RdBu_r",
    vmin=-v, vmax=+v,
    cb_label="ΔTCC (0–1)",
    fname=f"tcc_elnino_minus_lanina_{ANALYSIS_START}_{ANALYSIS_END}.png"
)
print(f"Saved tcc_elnino_minus_lanina_{ANALYSIS_START}_{ANALYSIS_END}.png")

# =========================
# (3) Correlation map ONI vs TCC anomalies
# =========================
# Build correlation per grid cell
X = oni_aligned
Y = anom  # [T, Y, X]

# ensure we only use rows where ONI is finite
valid_t = np.isfinite(X)
Xv = X[valid_t]
Yv = Y[valid_t, :, :]

def corr_along_time(x, ytx):
    """
    x   : [T]      predictor (ONI)
    ytx : [T,Y,X]  TCC anomalies
    returns rmap: [Y,X]
    """
    x = np.asarray(x, dtype=float)
    T, NY, NX = ytx.shape
    r = np.full((NY, NX), np.nan)

    # loop per grid cell; simple & robust (fast enough for monthly fields)
    for j in range(NY):
        for i in range(NX):
            y = ytx[:, j, i].astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 3:
                continue
            xi = x[mask]
            yi = y[mask]

            # de-mean
            xi = xi - xi.mean()
            yi = yi - yi.mean()

            sx = xi.std(ddof=1)
            sy = yi.std(ddof=1)
            if sx == 0 or sy == 0:
                continue

            r[j, i] = np.sum(xi * yi) / ((n - 1) * sx * sy)
    return r


rmap = corr_along_time(Xv, Yv)
plot_map(
    data2d=rmap,
    lon=lon180,
    lat=lat,
    title=f"Correlation: ONI vs TCC anomaly ({ANALYSIS_START}–{ANALYSIS_END})",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    cb_label="Pearson r",
    fname=f"tcc_oni_correlation_{ANALYSIS_START}_{ANALYSIS_END}.png"
)
print(f"Saved tcc_oni_correlation_{ANALYSIS_START}_{ANALYSIS_END}.png")

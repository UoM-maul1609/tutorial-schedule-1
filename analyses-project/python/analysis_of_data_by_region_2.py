#!/usr/bin/env python3
"""
Regional ERA5 TCC anomalies (area-weighted) with ENSO colouring.

- Reads monthly single-level NetCDFs with 'valid_time' (seconds since 1970-01-01)
- Computes regional area-weighted monthly means of TCC
- Builds 2010–2019 monthly climatology (configurable) *per region*
- Produces anomalies for a chosen analysis window
- Colours points by ENSO state from an ONI CSV (two columns: Date, ONI)

Outputs:
  - CSVs: regional_means_*, regional_anomalies_*
  - PNG: tcc_regional_anoms_<start>_<end>.png
"""

import glob, numpy as np, pandas as pd
from pathlib import Path
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from math import ceil

# =========================
# CONFIG — edit these only
# =========================
ANALYSIS_START = 2005
ANALYSIS_END   = 2025

# Baseline for monthly climatology (used for anomalies)
CLIM_START = 2010
CLIM_END   = 2019

# ENSO threshold for ONI (°C)
ENSO_THRESH = 0.5

# If True and land_sea_mask/lsm is present, use OCEAN ONLY (lsm < 0.5)
OCEAN_ONLY = False

# NetCDF folder and ONI file (two columns: Date, ONI)
indir = Path("era5_monthly_nc")
oni_csv = "oni.csv"

# Variable/coord name candidates
VAR_TCC   = ["tcc", "total_cloud_cover"]
LAT_CANDS = ["latitude", "lat"]
LON_CANDS = ["longitude", "lon"]
TIME_CANDS= ["valid_time", "time"]
LSM_CANDS = ["land_sea_mask", "lsm"]  # fraction land (0 ocean, 1 land)

# Regions (lat_min, lat_max, lon_min, lon_max) — lon can be in −180..180 or 0..360
REGIONS = {
    "Nino3.4":            (-5,   5,  -170,  -120),  # 5S–5N, 170W–120W
    "MaritimeContinent":  (-10, 10,   100,   150),  # 10S–10N, 100E–150E
    "CentralPacific":     (-5,   5,   160,   -150), # 160E–150W (wrap)
    "E_Pac_Stratocu_S":   (-25, -5,  -100,   -80),  # SEP stratocu
    "NE_Pac_Stratocu_N":  (15,  30,  -140,  -120),  # NEP stratocu
    "IndianEq":           (-10, 10,    60,   100),  # Equatorial Indian
}
# =========================================================

def pick(nc, candidates):
    for v in candidates:
        if v in nc.variables:
            return v
    return None

def area_weights(lat, lon):
    wlat = np.cos(np.deg2rad(lat))
    return np.broadcast_to(wlat[:, None], (len(lat), len(lon)))

def to0360(lon_vals):
    return np.mod(lon_vals, 360.0)

def bounds_to_0360(lmin, lmax):
    return (lmin % 360.0, lmax % 360.0)

def lon_mask_0360(lon_axis_0360, lo0360, hi0360):
    # supports wrap-around (e.g., 350..20)
    if lo0360 <= hi0360:
        return (lon_axis_0360 >= lo0360) & (lon_axis_0360 <= hi0360)
    else:
        return (lon_axis_0360 >= lo0360) | (lon_axis_0360 <= hi0360)

def region_mask(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """2D boolean mask for region bounds. Handles lon wrap and lon conventions."""
    # Ensure increasing bounds
    la0, la1 = (lat_min, lat_max) if lat_min <= lat_max else (lat_max, lat_min)
    lat_m = (lat >= la0) & (lat <= la1)

    # Put lon axis and bounds into 0..360 for robust selection
    lon0360 = to0360(lon.copy())
    lo0360, hi0360 = bounds_to_0360(lon_min, lon_max)
    lon_m = lon_mask_0360(lon0360, lo0360, hi0360)

    return lat_m[:, None] & lon_m[None, :]

def weighted_mean_region(data2d, w2d, mask2d):
    mask = np.isfinite(data2d) & mask2d
    if not np.any(mask):
        return np.nan
    w = w2d * mask
    return np.nansum(w * data2d) / np.nansum(w)

# -------------------------------
# Build file list (include clim years)
# -------------------------------
FILE_START = min(ANALYSIS_START, CLIM_START)
FILE_END   = max(ANALYSIS_END,   CLIM_END)
patterns = [f"era5_single_levels_monthly_{y}*.nc" for y in range(FILE_START, FILE_END+1)]
files = sorted({f for pat in patterns for f in glob.glob(str(indir / pat))})
if not files:
    raise FileNotFoundError(f"No files in {indir} for {FILE_START}-{FILE_END}")

# -------------------------------
# Loop files → regional means
# -------------------------------
records = []
for f in files:
    with Dataset(f) as nc:
        vlat = pick(nc, LAT_CANDS); vlon = pick(nc, LON_CANDS)
        if vlat is None or vlon is None:
            raise KeyError(f"lat/lon not found in {f}")
        lat = np.array(nc.variables[vlat][:], dtype=float)
        lon = np.array(nc.variables[vlon][:], dtype=float)
        w2d = area_weights(lat, lon)

        # Optional ocean-only mask
        ocean_mask = None
        if OCEAN_ONLY:
            vlsm = pick(nc, LSM_CANDS)
            if vlsm is not None:
                lsm = np.array(nc.variables[vlsm][...], dtype=float)
                # Make 2D
                if lsm.ndim == 3:
                    lsm = lsm[0, ...]
                ocean_mask = (lsm < 0.5)

        # Precompute region masks for this grid
        region_masks = {}
        for rname, (la0, la1, lo0, lo1) in REGIONS.items():
            m = region_mask(lat, lon, la0, la1, lo0, lo1)
            if OCEAN_ONLY and ocean_mask is not None:
                m = m & ocean_mask
            region_masks[rname] = m

        # Time and TCC var
        vtime = pick(nc, TIME_CANDS)
        if vtime is None:
            raise KeyError(f"time var not found in {f} (looked for {TIME_CANDS})")
        times = pd.to_datetime(np.asarray(nc.variables[vtime][:], dtype="int64"),
                               unit="s", utc=True).tz_localize(None)

        vtcc = pick(nc, VAR_TCC)
        if vtcc is None:
            raise KeyError(f"TCC not found in {f} (candidates {VAR_TCC})")

        for it, ts in enumerate(times):
            rec = {"time": ts, "year": ts.year, "month": ts.month}
            tcc2d = np.array(nc.variables[vtcc][it, ...], dtype=float)
            for rname, m in region_masks.items():
                rec[f"{rname}"] = weighted_mean_region(tcc2d, w2d, m)
            records.append(rec)

# -------------------------------
# Tidy to MultiIndex table
# -------------------------------
df = pd.DataFrame.from_records(records).set_index("time").sort_index()
# average duplicates if any
df = df.groupby([df.index.year, df.index.month]).mean()
df.index.names = ["year", "month"]
# drop any leftover 'year','month' columns from aggregation
for col in ("year", "month"):
    if col in df.columns:
        df = df.drop(columns=col)

# Save raw regional means (full span)
out_means = f"regional_means_{FILE_START}_{FILE_END}.csv"
df.to_csv(out_means, float_format="%.6g")
print(f"Saved {out_means}")

# -------------------------------
# Build anomalies per region
# -------------------------------
idx = pd.IndexSlice
df_in = df.loc[idx[ANALYSIS_START:ANALYSIS_END, 1:12], :]

clim_src = df.loc[idx[CLIM_START:CLIM_END, 1:12], :]
if clim_src.empty:
    print(f"WARNING: No data for baseline {CLIM_START}-{CLIM_END}. "
          f"Using available years {FILE_START}-{FILE_END} as baseline.")
    clim_src = df

anom = df_in.copy()
months = anom.index.get_level_values("month")

for col in df.columns:
    clim_month = clim_src[col].groupby(level="month").mean()
    # map month → climatology value for that region
    anom[col] = df_in[col].values - months.map(clim_month).values

# Save anomalies
out_anoms = f"regional_anomalies_{ANALYSIS_START}_{ANALYSIS_END}.csv"
anom.reset_index().to_csv(out_anoms, index=False, float_format="%.6g")
print(f"Saved {out_anoms}")

# -------------------------------
# ENSO labels from ONI (two columns: Date, ONI; some rows may be headers or -9999)
# -------------------------------
oni_raw = pd.read_csv(oni_csv, header=None, names=["date", "oni"])
oni_raw["date"] = pd.to_datetime(oni_raw["date"], errors="coerce")
oni_raw["oni"]  = pd.to_numeric(oni_raw["oni"], errors="coerce")
oni = oni_raw.dropna(subset=["date"]).copy()
oni.loc[oni["oni"] <= -99, "oni"] = np.nan  # treat sentinels as missing
oni["year"]  = oni["date"].dt.year
oni["month"] = oni["date"].dt.month
oni = oni.set_index(["year", "month"]).sort_index()

oni_vals = oni.reindex(anom.index)["oni"]
state = pd.Series(index=anom.index, dtype="string")
state[(oni_vals >=  ENSO_THRESH).fillna(False)] = "ElNino"
state[(oni_vals <= -ENSO_THRESH).fillna(False)] = "LaNina"
state[state.isna()] = "Neutral"

# -------------------------------
# Print composites (mean anomaly per ENSO phase, by region)
# -------------------------------
comp = anom.join(state.rename("state")).groupby("state").mean()
comp = comp.reindex(["ElNino", "Neutral", "LaNina"])
print("\nRegional mean TCC anomalies by ENSO phase:")
print(comp)

# -------------------------------
# Plot: per-region anomalies coloured by ENSO
# -------------------------------
months_labels = [f"{y}-{m:02d}" for y, m in anom.index]
x = np.arange(len(months_labels))
col_map = {"ElNino": "tab:red", "Neutral": "tab:gray", "LaNina": "tab:blue"}

region_names = list(REGIONS.keys())
nreg = len(region_names)
ncol = 3
nrow = int(ceil(nreg / ncol))

fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0*nrow), sharex=True)
axes = np.array(axes).reshape(nrow, ncol)

for i, r in enumerate(region_names):
    ax = axes[i // ncol, i % ncol]
    y = anom[r].values
    x = np.asarray(oni_vals)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Scatter
    ax.plot(x, y, '.', alpha=0.75)
    # Zero line
    ax.axhline(0, lw=1, color='k', alpha=0.5)
    ax.set_title(r)
    ax.grid(True)

    # Fit & R^2 (only if we have 2+ points)
    if x.size >= 2:
        slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)
        yfit = intercept + slope * xfit
        ax.plot(xfit, yfit, lw=2)  # fitted line
        # Annotate R^2 in the top-left corner inside the axes
        ax.text(0.02, 0.95, f"$R^2$={r_val**2:.2f}",
                transform=ax.transAxes, va='top', ha='left')

fig.supxlabel('Oni val')

# Hide any empty subplot slots
for j in range(nreg, nrow*ncol):
    axes[j // ncol, j % ncol].set_visible(False)

# # reasonable tick density
# step = max(1, len(x)//18)
# for ax in axes[-1, :]:
#     ax.set_xticks(x[::step])
#     ax.set_xticklabels(months_labels[::step], rotation=90)

fig.suptitle(f"TCC regional anomalies ({ANALYSIS_START}–{ANALYSIS_END}) vs {CLIM_START}–{CLIM_END}")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = f"tcc_regional_anoms_correl_{ANALYSIS_START}_{ANALYSIS_END}.png"
plt.savefig(out_png, dpi=140)
print(f"Saved {out_png}")

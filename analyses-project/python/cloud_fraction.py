#!/usr/bin/env python3
"""
ERA5 single-level monthly means (2005–2025):
Global area-weighted means + standard error for
- tcc  (total cloud fraction, 0..1)
- tclw (total column cloud liquid water, kg m^-2)
- tciw (total column cloud ice water, kg m^-2)
- tcwv (total column water vapour, kg m^-2)

Outputs:
  results/cloud_tcwv_globals_ERA5_2005_2025.csv
  plots/cloud_tcwv_timeseries_2005_2025.png
"""

import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- config ----------------
YEAR_START, YEAR_END = 2005, 2025
indir = Path("era5_monthly_nc")
patterns = [f"era5_single_levels_monthly_{y}*.nc" for y in range(YEAR_START, YEAR_END + 1)]

outdir_img = Path("plots");  outdir_img.mkdir(exist_ok=True)
outdir_tab = Path("results"); outdir_tab.mkdir(exist_ok=True)

# Candidate names (short + long to be robust)
VAR_CANDS = {
    "tcc":   ["tcc", "total_cloud_cover"],
    "tclw":  ["tclw", "total_column_cloud_liquid_water"],
    "tciw":  ["tciw", "total_column_cloud_ice_water"],
    "tcwv":  ["tcwv", "total_column_water_vapour", "total_column_water_vapor"],
}
LAT_CANDS  = ["latitude", "lat"]
LON_CANDS  = ["longitude", "lon"]
TIME_CANDS = ["valid_time", "time"]  # valid_time preferred (seconds since 1970-01-01)

# -------------- helpers -----------------
def pick_first_present(nc, candidates):
    for v in candidates:
        if v in nc.variables:
            return v
    return None

def area_weights(lats, lons):
    """cos(lat) weights broadcast to (lat, lon)."""
    wlat = np.cos(np.deg2rad(np.asarray(lats)))
    return np.repeat(wlat[:, None], len(lons), axis=1)

def weighted_mean_se(data2d, w2d):
    """Area-weighted mean and standard error over lat/lon."""
    m = np.isfinite(data2d)
    if not np.any(m):
        return np.nan, np.nan
    w = w2d * m
    wsum = w.sum()
    if wsum == 0:
        return np.nan, np.nan
    wnorm = w / wsum
    mean = np.nansum(data2d * wnorm)
    var  = np.nansum(wnorm * (data2d - mean)**2)
    Neff = 1.0 / np.nansum(wnorm**2)
    se   = np.sqrt(var) / np.sqrt(Neff)
    return mean, se

# -------------- read all files ----------
files = sorted({f for pat in patterns for f in glob.glob(str(indir / pat))})
if not files:
    raise FileNotFoundError(f"No files found in {indir} for {YEAR_START}-{YEAR_END}")

records = []

for f in files:
    with Dataset(f) as nc:
        lat_name = pick_first_present(nc, LAT_CANDS)
        lon_name = pick_first_present(nc, LON_CANDS)
        if lat_name is None or lon_name is None:
            raise KeyError(f"lat/lon not found in {f}. Vars: {list(nc.variables)}")

        lats = np.array(nc.variables[lat_name][:], dtype=float)
        lons = np.array(nc.variables[lon_name][:], dtype=float)
        w2d  = area_weights(lats, lons)

        time_name = pick_first_present(nc, TIME_CANDS)
        if time_name is None:
            raise KeyError(f"No valid time variable in {f} (looked for {TIME_CANDS})")

        tvals = np.asarray(nc.variables[time_name][:], dtype="int64")
        times = pd.to_datetime(tvals, unit="s", utc=True).tz_localize(None)

        names = {k: pick_first_present(nc, v) for k, v in VAR_CANDS.items()}

        for it, t in enumerate(times):
            if not (YEAR_START <= t.year <= YEAR_END):
                continue
            row = {"year": t.year, "month": t.month}
            for key, vname in names.items():
                if vname is None:
                    row[f"{key}_mean"] = np.nan
                    row[f"{key}_se"]   = np.nan
                    continue
                arr = np.array(nc.variables[vname][...], dtype=float)
                # arr can be (time, lat, lon) or (lat, lon)
                if arr.ndim == 3:
                    data2d = arr[it, ...]
                elif arr.ndim == 2:
                    if it > 0:
                        # single slice in file; but loop thinks there are multiple times
                        # be safe and reuse (common in monthly-mean files is one timestep)
                        data2d = arr
                    else:
                        data2d = arr
                else:
                    # unexpected shape
                    row[f"{key}_mean"] = np.nan
                    row[f"{key}_se"]   = np.nan
                    continue
                mean, se = weighted_mean_se(data2d, w2d)
                row[f"{key}_mean"] = mean
                row[f"{key}_se"]   = se
            records.append(row)

# -------------- tidy & merge dup months ---
df = pd.DataFrame.from_records(records)
# If multiple files contain the same (year,month), average them:
df = df.groupby(["year", "month"], as_index=True).mean(numeric_only=True).sort_index()

# Save table
csv_path = outdir_tab / f"cloud_tcwv_globals_ERA5_{YEAR_START}_{YEAR_END}.csv"
df.to_csv(csv_path, float_format="%.6g")
print(f"Saved table → {csv_path}")

# -------------- prep plotting -------------
months = np.arange(1, 12 + 1)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
years = list(range(YEAR_START, YEAR_END + 1))

def get_series(df, var, year):
    idx = pd.MultiIndex.from_product([[year], months], names=["year","month"])
    s_mean = df.get(f"{var}_mean")
    s_se   = df.get(f"{var}_se")
    if s_mean is None or s_se is None:
        return np.full(12, np.nan), np.full(12, np.nan)
    s_mean = s_mean.reindex(idx)
    s_se   = s_se.reindex(idx)
    return s_mean.values, s_se.values

def monthly_mean_over_years(df, var):
    """Mean across years for each month (uses available years only)."""
    s = df[f"{var}_mean"].unstack(level=0)  # index=month, columns=year
    # ensure months 1..12 are present as index, reindex if needed
    s = s.reindex(index=months)
    return np.array(s.mean(axis=1, skipna=True))

# -------------- plotting ------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

series_meta = [
    ("tcc",  "Total cloud fraction (0–1)"),
    ("tclw", "Column liquid water (kg m$^{-2}$)"),
    ("tciw", "Column ice water (kg m$^{-2}$)"),
    ("tcwv", "Total column water vapour (kg m$^{-2}$)"),
]

for ax, (key, ylabel) in zip(axes, series_meta):
    # light spaghetti of all years
    for yr in years:
        y_mean, y_se = get_series(df, key, yr)
        ax.plot(months, y_mean, lw=1.0, alpha=0.35)
        # SE bands per year (optional; comment out if too busy)
        if np.isfinite(y_se).any():
            ax.fill_between(months, y_mean - y_se, y_mean + y_se, alpha=0.08)
    # multi-year monthly mean (bold)
    clim_line = monthly_mean_over_years(df, key)
    ax.plot(months, clim_line, lw=2.5, color="k", label=f"{key.upper()} 2005–2025 mean")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(1, 12)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)

axes[0].set_title("ERA5 monthly global means ± SE by month (2005–2025)\n(thin lines: individual years; thick line: 2005–2025 mean)")
axes[-1].set_xlabel("Month")
axes[0].legend(frameon=False)

fig.tight_layout()
png_path = outdir_img / f"cloud_tcwv_timeseries_{YEAR_START}_{YEAR_END}.png"
plt.savefig(png_path, dpi=150)
print(f"Saved plot → {png_path}")

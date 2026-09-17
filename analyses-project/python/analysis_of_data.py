#!/usr/bin/env python3
"""
ERA5 global-mean anomalies for TCC & TCWV, coloured by ENSO.

Fixes:
- Remove duplicate 'year'/'month' columns after MultiIndex groupby.
- Use groupby(level="month") to avoid ambiguity.
- Parse ONI CSV with two columns ('Date','ONI'), drop headers, handle -9999.

Outputs:
- global_means_<FILE_START>_<FILE_END>.csv
- anomalies_<ANALYSIS_START>_<ANALYSIS_END>.csv
- era5_global_anoms_by_ENSO_<ANALYSIS_START>_<ANALYSIS_END>.png
- Printed El Niño / Neutral / La Niña composites
"""

import glob, numpy as np, pandas as pd
from pathlib import Path
from netCDF4 import Dataset
import matplotlib.pyplot as plt

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

# Data location
indir = Path("era5_monthly_nc")
oni_csv = "oni.csv"   # two columns: Date, ONI

# Variables (single levels, monthly means)
vars_single = {
    "tcc":  ["tcc", "total_cloud_cover"],
    "tcwv": ["tcwv", "total_column_water_vapour", "total_column_water_vapor"],
}
LAT_CANDS, LON_CANDS, TIME_CANDS = ["latitude","lat"], ["longitude","lon"], ["valid_time","time"]

# =========================================================
# Helpers
# =========================================================
def pick(nc, candidates):
    for v in candidates:
        if v in nc.variables:
            return v
    return None

def area_weights(lat, lon):
    wlat = np.cos(np.deg2rad(lat))
    return np.broadcast_to(wlat[:, None], (len(lat), len(lon)))

def wmean(data2d, w2d):
    m = np.isfinite(data2d)
    w = w2d * m
    return np.nan if w.sum() == 0 else np.nansum(w * data2d) / np.nansum(w)

# =========================================================
# Build file list (include baseline years if outside analysis range)
# =========================================================
FILE_START = min(ANALYSIS_START, CLIM_START)
FILE_END   = max(ANALYSIS_END,   CLIM_END)
patterns = [f"era5_single_levels_monthly_{y}*.nc" for y in range(FILE_START, FILE_END+1)]

files = sorted({f for pat in patterns for f in glob.glob(str(indir / pat))})
if not files:
    raise FileNotFoundError(f"No files found in {indir} for {FILE_START}-{FILE_END}")

# =========================================================
# Read monthly global means
# =========================================================
rows = []
for f in files:
    with Dataset(f) as nc:
        la = pick(nc, LAT_CANDS); lo = pick(nc, LON_CANDS)
        if la is None or lo is None:
            raise KeyError(f"lat/lon not found in {f}. Have: {list(nc.variables)}")
        lat = nc.variables[la][:]; lon = nc.variables[lo][:]
        w2d = area_weights(lat, lon)

        tname = pick(nc, TIME_CANDS)
        if tname is None:
            raise KeyError(f"time variable not found in {f} (looked for {TIME_CANDS})")
        t = pd.to_datetime(np.asarray(nc.variables[tname][:], dtype="int64"),
                           unit="s", utc=True).tz_localize(None)

        names = {k: pick(nc, c) for k, c in vars_single.items()}

        for it, ts in enumerate(t):
            rec = {"time": ts, "year": ts.year, "month": ts.month}
            for k, vn in names.items():
                if vn is None:
                    rec[k] = np.nan
                else:
                    rec[k] = wmean(np.array(nc.variables[vn][it, ...], float), w2d)
            rows.append(rec)

df = pd.DataFrame(rows).set_index("time").sort_index()

# Average if duplicate monthly entries exist (e.g., multiple files with same month)
df = df.groupby([df.index.year, df.index.month]).mean()
df.index.names = ["year", "month"]

# Drop duplicate 'year'/'month' columns created by the aggregation
for col in ["year", "month"]:
    if col in df.columns:
        df = df.drop(columns=col)

# Save raw global means over full loaded span
full_means_csv = f"global_means_{FILE_START}_{FILE_END}.csv"
df.to_csv(full_means_csv, float_format="%.6g")
print(f"Saved {full_means_csv}")

# Keep only requested analysis years for outputs/plots
idx = pd.IndexSlice
df_in_range = df.loc[idx[ANALYSIS_START:ANALYSIS_END, 1:12], :]

# =========================================================
# Climatology (monthly) and anomalies
# =========================================================
clim_source = df.loc[idx[CLIM_START:CLIM_END, 1:12], :]
if clim_source.empty:
    print(f"Warning: No data for baseline {CLIM_START}-{CLIM_END}. "
          f"Using available years {FILE_START}-{FILE_END} as baseline instead.")
    clim_source = df

# IMPORTANT: group by index level (month), not a column
clim = clim_source.groupby(level="month")[["tcc", "tcwv"]].mean()

anom = df_in_range.copy()
for m in range(1, 13):
    if m in clim.index:
        anom.loc[idx[:, m], ["tcc", "tcwv"]] = (
            df_in_range.loc[idx[:, m], ["tcc", "tcwv"]].values
            - clim.loc[m, ["tcc", "tcwv"]].values
        )

# Save anomalies table
anom_csv = f"anomalies_{ANALYSIS_START}_{ANALYSIS_END}.csv"
anom.reset_index().to_csv(anom_csv, index=False, float_format="%.6g")
print(f"Saved {anom_csv}")

# =========================================================
# ONI: parse your two-column file ('Date','ONI')
# =========================================================
# The file may contain duplicate header lines and -9999 sentinels.
oni_raw = pd.read_csv(
    oni_csv,
    header=None,              # treat all rows as data
    names=["date", "oni"],    # name the two columns
)

# Coerce types; drop non-parsable header lines; clean sentinels
oni_raw["date"] = pd.to_datetime(oni_raw["date"], errors="coerce")
oni_raw["oni"]  = pd.to_numeric(oni_raw["oni"], errors="coerce")
oni = oni_raw.dropna(subset=["date"]).copy()
# Treat big negative sentinels as missing
oni.loc[oni["oni"] <= -99, "oni"] = np.nan

oni["year"]  = oni["date"].dt.year
oni["month"] = oni["date"].dt.month
oni = oni.set_index(["year", "month"]).sort_index()

# Align ONI to our anomaly index
oni_vals = oni.reindex(anom.index)["oni"]

# Label phases; keep NaNs as Neutral fallback
state = pd.Series(index=anom.index, dtype="string")
state[(oni_vals >=  ENSO_THRESH).fillna(False)] = "ElNino"
state[(oni_vals <= -ENSO_THRESH).fillna(False)] = "LaNina"
state[state.isna()] = "Neutral"

# =========================================================
# Composites (global-mean anomalies by ENSO phase)
# =========================================================
comp = (anom.join(state.rename("state"))
          .groupby("state")[["tcc", "tcwv"]].mean()
          .reindex(["ElNino", "Neutral", "LaNina"]))
print(f"\nGlobal-mean anomalies ({ANALYSIS_START}-{ANALYSIS_END}) "
      f"relative to {CLIM_START}-{CLIM_END}:")
print(comp)

# =========================================================
# Plot: monthly anomalies colored by ENSO state
# =========================================================
months_labels = [f"{y}-{m:02d}" for y, m in anom.index]
x = np.arange(len(months_labels))
colors = {"ElNino": "tab:red", "Neutral": "tab:gray", "LaNina": "tab:blue"}

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for i, var in enumerate(["tcc", "tcwv"]):
    c = [colors.get(s, "k") for s in state.values]
    axes[i].scatter(x, anom[var].values, s=12, c=c)
    axes[i].axhline(0, lw=1)
    axes[i].set_ylabel(f"{var} anomaly")
    axes[i].grid(True)

# sensible tick density even for long ranges
step = max(1, len(x)//20)
axes[-1].set_xticks(x[::step])
axes[-1].set_xticklabels(months_labels[::step], rotation=90)

axes[0].set_title(f"ERA5 global-mean anomalies ({ANALYSIS_START}–{ANALYSIS_END}) "
                  f"vs climatology {CLIM_START}–{CLIM_END}: TCC & TCWV, colored by ENSO")
plt.tight_layout()
out_png = f"era5_global_anoms_by_ENSO_{ANALYSIS_START}_{ANALYSIS_END}.png"
plt.savefig(out_png, dpi=140)
print(f"Saved {out_png}")

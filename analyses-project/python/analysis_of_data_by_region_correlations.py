# --- CONFIG: which regions to test ---
REGIONS_TO_TEST = ["Nino3.4", "MaritimeContinent", "CentralPacific", "E_Pac_Stratocu_S", "NE_Pac_Stratocu_N", "IndianEq"]

import numpy as np, pandas as pd
from math import atanh, tanh, sqrt
from scipy import stats
import matplotlib.pyplot as plt

# Align ONI with anomalies
oni_aligned = oni.reindex(anom.index)["oni"]
X = oni_aligned  # predictor
results = []

def eff_sample_size(x, y, maxlag=1):
    """Bretherton-style N_eff using lag-1 autocorr of x and y (or up to maxlag)."""
    x = pd.Series(x).dropna(); y = pd.Series(y).dropna()
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return len(xy)
    # simple AR(1) estimate
    def r1(z):
        z = z - z.mean()
        return np.corrcoef(z[:-1], z[1:])[0,1] if len(z) > 2 else 0.0
    rx = r1(xy.iloc[:,0].values)
    ry = r1(xy.iloc[:,1].values)
    n = len(xy)
    return n * (1 - rx*ry) / (1 + rx*ry)

def corr_ci(r, n_eff, alpha=0.05):
    if n_eff <= 3 or np.isnan(r):
        return (np.nan, np.nan)
    z = atanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / sqrt(max(1.0, n_eff - 3))
    zc = stats.norm.ppf(1 - alpha/2)
    lo, hi = z - zc*se, z + zc*se
    return (tanh(lo), tanh(hi))

for reg in REGIONS_TO_TEST:
    y = anom[reg]
    df_xy = pd.concat([X, y], axis=1).dropna()
    if df_xy.empty:
        results.append((reg, np.nan, np.nan, np.nan, np.nan, np.nan)); continue
    x = df_xy.iloc[:,0].values
    yy = df_xy.iloc[:,1].values

    # Pearson r
    r = np.corrcoef(x, yy)[0,1]
    n_eff = eff_sample_size(x, yy)
    # p-value using Student-t with df = n_eff-2
    if n_eff > 2 and not np.isnan(r):
        tstat = r * sqrt((n_eff - 2) / (1 - r**2 + 1e-15))
        pval = 2 * (1 - stats.t.cdf(abs(tstat), df=max(1, int(round(n_eff - 2)))))
    else:
        pval = np.nan
    lo, hi = corr_ci(r, n_eff)

    # slope (OLS): ΔTCC per 1 °C ONI
    slope, intercept, _, _, _ = stats.linregress(x, yy)

    results.append((reg, r, lo, hi, slope, pval))

corr_table = pd.DataFrame(results, columns=["region","r","r_lo95","r_hi95","slope_per_ONI","p_adj"])
print("\nCorrelation with ONI (monthly, anomalies), adj. for autocorr:")
print(corr_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# -------- OPTIONAL: lead/lag correlation curves --------
DO_LAG_CURVES = True
if DO_LAG_CURVES:
    maxlag = 6
    lags = np.arange(-maxlag, maxlag+1)
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    for reg in REGIONS_TO_TEST:
        series = []
        for L in lags:
            if L >= 0:
                xL = X.shift(L); yL = anom[reg]
            else:
                xL = X; yL = anom[reg].shift(-L)
            df_xy = pd.concat([xL, yL], axis=1).dropna()
            if len(df_xy) < 6:
                series.append(np.nan); continue
            rL = np.corrcoef(df_xy.iloc[:,0].values, df_xy.iloc[:,1].values)[0,1]
            series.append(rL)
        ax.plot(lags, series, label=reg)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Lag (months): ONI leads (+)")
    ax.set_ylabel("Corr(ONI, TCC anomaly)")
    ax.grid(True); ax.legend()
    plt.tight_layout(); plt.savefig("tcc_region_lag_correlations.png", dpi=140)
    print("Saved tcc_region_lag_correlations.png")

# -------- OPTIONAL: seasonal correlations (DJF vs JJA) --------
DO_SEASONS = True
if DO_SEASONS:
    season = pd.Series(index=anom.index, dtype="string")
    # simple DJF/JJA tags (others = shoulder seasons)
    months = anom.index.get_level_values("month")
    season[(months==12) | (months<=2)] = "DJF"
    season[(months>=6) & (months<=8)]  = "JJA"

    out = []
    for reg in REGIONS_TO_TEST:
        for s in ["DJF","JJA"]:
            mask = (season==s)
            df_xy = pd.concat([X[mask], anom[reg][mask]], axis=1).dropna()
            if len(df_xy) < 6:
                out.append((reg, s, np.nan, np.nan)); continue
            r = np.corrcoef(df_xy.iloc[:,0], df_xy.iloc[:,1])[0,1]
            n_eff = eff_sample_size(df_xy.iloc[:,0], df_xy.iloc[:,1])
            lo, hi = corr_ci(r, n_eff)
            out.append((reg, s, r, n_eff))
    seas_tbl = pd.DataFrame(out, columns=["region","season","r","n_eff"])
    print("\nSeasonal correlation (DJF/JJA):")
    print(seas_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

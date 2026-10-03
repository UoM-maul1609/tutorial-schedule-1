# ENSO and Cloud Cover using ERA5

## Launch
| Notebook | What it's for | Launch |
|---|---|---|
| **`student_project.ipynb`** | The main project notebook: run it from the top. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UoM-maul1609/tutorial-schedule-1/HEAD?labpath=analyses-project%2Fpython%2Fstudent_project.ipynb) |
| **`methods_demos.ipynb`** | *Optional.* Short demos of the methods used in the tutorials: thresholds, composites, why *r* needs *n* (coins in a bucket), calculating a p-value, effective sample size, a correlation map made from pure noise, and a "shift test" for your own region. Needs only `oni.csv`, so it runs in seconds. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UoM-maul1609/tutorial-schedule-1/HEAD?labpath=analyses-project%2Fpython%2Fmethods_demos.ipynb) |

Both notebooks open in the same Binder environment, so once one is running you can open the other from the file browser on the left.

> **Tutors:** launch a Binder link yourself 15–30 minutes before each session.
> The first launch after any change to the repo triggers a slow image rebuild;
> every launch after that reuses the cached image and starts in seconds. This
> matters most in the first session, when every student launches at once.

## The important idea
The notebook is a **toolbox, not a checklist**. Explore the core analyses, then choose one aspect of ENSO and cloud cover that interests you. Your report should answer **one focused question** using about **3–5 figures**, in at most **2,000 words**, not reproduce every notebook output.

Possible directions include cloud level, region, season, persistence or lag.

## What's in `student_project.ipynb`
| Section | Content |
|---|---|
| 0 · 0b | Setup and load the data (run once) |
| 1 | Global seasonal cycle |
| 2 | Global ENSO maps: El Niño − La Niña composite and correlation with the ONI |
| 3 | Global-mean anomalies by ENSO phase |
| 3b | *Optional:* is global cloud cover different from one year to the next? (t-tests with naive and autocorrelation-adjusted p-values, multiple-comparison warning, optional detrending) |
| 4 | Regional analysis for 12 numbered regions |
| 5 | ONI vs regional anomaly: scatter plots showing *r*, *R²* and *n* |
| 6 | *Optional:* correlation table with *n*, effective sample size *n*<sub>eff</sub>, 95% range and adjusted p-value; lag and seasonal correlations |
| STOP | Choose your investigation before doing more analysis |

To study another variable, change one line in section 0: `VARIABLE = "lcc"` (or `"mcc"`, `"hcc"`, `"skt"`, `"tcwv"`, …).

## Data
- **ERA5** monthly means (January 2005 – August 2025): ECMWF / Copernicus Climate Change Service (Hersbach et al. 2020). The first code cell downloads the files from this repository's release into `era5_monthly_nc/` if they aren't already there.
- **ONI**: NOAA Climate Prediction Center, supplied in `oni.csv`.

Figures you generate are your own figures, but acknowledge/cite the underlying datasets.

## Guidance
- The **Student Guide**, the tutorial slides (Tutorials 1–8, files `W1_…` to `W8_…`) and the **Writing Workbook** are in `../guidance/`.

**Binder sessions are temporary:** download your figures, results and edited notebook before you close the tab.

#!/usr/bin/env python3
import cdsapi
from pathlib import Path

# ------------------------------
# Config — edit these as needed
# ------------------------------
outdir = Path("./era5_monthly_nc")
outdir.mkdir(parents=True, exist_ok=True)

# Date window
start_year = 2007
end_year   = 2025
months     = [1,2,3,4,5,6,7,8,9,10,11,12]                 # 1..12; set to list(range(1,13)) for full-year

# Domain / grid (matches your script)
area = [90, 0, -90, 360]         # N, W, S, E
grid = [0.75, 0.75]

# Single-level variables (same set you used)
single_level_variables = [
    "geopotential",
    "land_sea_mask",
    "skin_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "sea_ice_cover",
    "snow_depth",
    "surface_pressure",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "total_cloud_cover",
    "total_column_cloud_liquid_water",
    "total_column_cloud_ice_water",
    "total_column_water_vapour",
    ]

# ------------------------------
# Download ERA5 monthly means — SINGLE LEVELS
# ------------------------------
c = cdsapi.Client()

for year in range(start_year, end_year + 1):
    for m in months:
        ym = f"{year:04d}-{m:02d}"
        target = outdir / f"era5_single_levels_monthly_{year}{m:02d}.nc"
        print(f"[single-levels] {ym} → {target.name}")

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": single_level_variables,
                "year": f"{year:04d}",
                "month": f"{m:02d}",
                # For monthly_averaged_reanalysis, CDS expects a single 'time' of 00:00
                "time": "00:00",
                "format": "netcdf",
                "area": area,
                "grid": grid,
            },
            str(target),
        )

print("Done: single-level monthly means saved as NetCDF.")

# --------------------------------------------------------------------
# OPTIONAL: Pressure-level monthly means (if you need upper-air fields)
# --------------------------------------------------------------------
# Uncomment and set the variables/levels you want.

# pressure_level_variables = [
#     "geopotential",
#     "temperature",
#     "u_component_of_wind",
#     "v_component_of_wind",
#     "vertical_velocity",
#     "relative_humidity",
#     "vertical_velocity",
#     "fraction_of_cloud_cover",
#     "specific_cloud_liquid_water_content",
#     "specific_cloud_ice_water_content",
#     "specific_rain_water_content",
#     "specific_snow_water_content",
#     "specific_humidity",
#     ]
# pressure_levels = ["1000","925","850","700","600","500","400","300","250","200","150","100"]
# 
# for year in range(start_year, end_year + 1):
#     for m in months:
#         ym = f"{year:04d}-{m:02d}"
#         target = outdir / f"era5_pressure_levels_monthly_{year}{m:02d}.nc"
#         print(f"[pressure-levels] {ym} → {target.name}")
# 
#         c.retrieve(
#             "reanalysis-era5-pressure-levels-monthly-means",
#             {
#                 "product_type": "monthly_averaged_reanalysis",
#                 "variable": pressure_level_variables,
#                 "pressure_level": pressure_levels,
#                 "year": f"{year:04d}",
#                 "month": f"{m:02d}",
#                 "time": "00:00",
#                 "format": "netcdf",
#                 "area": area,
#                 "grid": grid,
#             },
#             str(target),
#         )
# 
# print("Done: pressure-level monthly means saved as NetCDF.")

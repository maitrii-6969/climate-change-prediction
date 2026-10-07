import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = r"C:\climate_project"
IMD_DIR = os.path.join(ROOT, "data", "raw", "imd")
OUT_DIR = os.path.join(ROOT, "data", "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = {
    "Mumbai": (19.1, 72.9), "Delhi": (28.6, 77.2), "Chennai": (13.1, 80.3),
    "Kolkata": (22.6, 88.4), "Bengaluru": (12.9, 77.6), "Hyderabad": (17.4, 78.5),
    "Jaipur": (26.9, 75.8), "Ahmedabad": (23.0, 72.6), "Bhopal": (23.3, 77.4),
    "Nagpur": (21.1, 79.1), "Guwahati": (26.1, 91.7), "Thiruvananthapuram": (8.5, 76.9),
}

city_rows, annual_rows = [], []

for year in range(2015, 2025):
    ds = xr.open_dataset(os.path.join(IMD_DIR, f"RF25_ind{year}_rfp25.nc"))
    rf = ds["RAINFALL"].values.astype("float32")      # (days, lat, lon)
    rf[rf < 0] = np.nan                               # missing-value flags
    lats, lons = ds["LATITUDE"].values, ds["LONGITUDE"].values
    days = pd.date_range(f"{year}-01-01", periods=rf.shape[0], freq="D")

    # India mean of the annual total, over land cells only
    annual_cell = np.nansum(rf, axis=0)
    land = ~np.all(np.isnan(rf), axis=0)
    annual_rows.append({"year": year, "india_mean_mm": float(annual_cell[land].mean())})

    # City series: average of the 3x3 grid cells around the nearest cell
    for name, (lat, lon) in CITIES.items():
        i = int(np.abs(lats - lat).argmin())
        j = int(np.abs(lons - lon).argmin())
        block = rf[:, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
        daily = np.nanmean(block.reshape(block.shape[0], -1), axis=1)
        monthly = pd.Series(daily, index=days).resample("MS").sum(min_count=1)
        for month, mm in monthly.items():
            city_rows.append({"month": month.date(), "city": name,
                              "rainfall_mm": round(float(mm), 1)})
    ds.close()
    print(year, "done")

pd.DataFrame(city_rows).to_csv(os.path.join(OUT_DIR, "imd_city_monthly.csv"), index=False)
annual = pd.DataFrame(annual_rows)
annual.to_csv(os.path.join(OUT_DIR, "imd_india_annual.csv"), index=False)
print(annual.round(0).to_string(index=False))

# 17_modis_city_extract.py - read the 360 real MODIS GeoTIFFs and build monthly series.
# Output 1: modis_city_monthly.csv   (city, date, ndvi, evi, lst_c)  - mean of the 3x3 pixels around each city
# Output 2: modis_india_monthly.csv  (date, ndvi, evi, lst_c)        - India-wide mean of valid pixels
# MODIS no-data is 0.0 in these files, so 0.0 is treated as missing. Values are already scaled (NDVI/EVI) and in deg C (LST).
import os, glob, re
import numpy as np, pandas as pd, rasterio
import rain_common as rc

MODIS_DIR = os.environ.get("MODIS_DIR", r"C:\climate_project\data\raw\modis")
VARS = {"ndvi": "NDVI", "evi": "EVI", "lst_c": "LST"}

city_rows, india_rows = [], []
for var, tag in VARS.items():
    files = sorted(glob.glob(os.path.join(MODIS_DIR, "**", f"{tag}_*.tif"), recursive=True))
    print(f"{tag}: {len(files)} files")
    for f in files:
        m = re.search(r"_(\d{4})_(\d{2})\.tif$", f)
        date = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        with rasterio.open(f) as r:
            a = r.read(1).astype("float64")
            a[(a == 0.0) | ~np.isfinite(a)] = np.nan
            india_rows.append((var, date, float(np.nanmean(a))))
            for city, (lat, lon) in rc.CITIES.items():
                row, col = r.index(lon, lat)
                win = a[max(row-1, 0):row+2, max(col-1, 0):col+2]
                city_rows.append((var, city, date, float(np.nanmean(win)) if np.isfinite(win).any() else np.nan))

c = pd.DataFrame(city_rows, columns=["var", "city", "date", "v"]).pivot_table(index=["city", "date"], columns="var", values="v").reset_index()
i = pd.DataFrame(india_rows, columns=["var", "date", "v"]).pivot_table(index="date", columns="var", values="v").reset_index()
print("\nCity-months:", len(c), "| missing NDVI:", int(c.ndvi.isna().sum()), "EVI:", int(c.evi.isna().sum()), "LST:", int(c.lst_c.isna().sum()))
print(c.groupby("city")[["ndvi", "evi", "lst_c"]].mean().round(3).to_string())
c.to_csv(os.path.join(rc.OUT_DIR, "modis_city_monthly.csv"), index=False, float_format="%.4f")
i.to_csv(os.path.join(rc.OUT_DIR, "modis_india_monthly.csv"), index=False, float_format="%.4f")
print("\nSaved to", rc.OUT_DIR)

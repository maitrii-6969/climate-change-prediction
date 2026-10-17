# rain_common.py - shared helpers for scripts 12, 13, 14
# Builds a monthly rainfall table per city from the real IMD gridded files.
import os, glob
import numpy as np
import pandas as pd
import xarray as xr

IMD_DIR = os.environ.get("IMD_DIR", r"C:\climate_project\data\raw\imd")
OUT_DIR = os.environ.get("OUT_DIR", r"C:\climate_project\outputs\results")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = {
    "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36), "Bangalore": (12.97, 77.59), "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79), "Ahmedabad": (23.02, 72.57), "Bhopal": (23.26, 77.41),
    "Nagpur": (21.15, 79.09),
}
TRAIN_END, VAL_YEAR, TEST_START = 2021, 2022, 2023


def load_monthly():
    """Return DataFrame [city, date, rain_mm] of monthly totals (3x3 cell mean around city)."""
    files = sorted(glob.glob(os.path.join(IMD_DIR, "RF25_ind*_rfp25.nc")))
    if not files:
        raise FileNotFoundError(f"No IMD files in {IMD_DIR}")
    rows = []
    for f in files:
        ds = xr.open_dataset(f)
        da = ds["RAINFALL"]
        da = da.where(da >= 0)                      # negative = missing flag
        times = pd.to_datetime(ds["TIME"].values) if np.issubdtype(ds["TIME"].dtype, np.datetime64) else None
        if times is None:                            # fall back: day-of-year from file year
            year = int(os.path.basename(f).split("ind")[1][:4])
            times = pd.date_range(f"{year}-01-01", periods=da.sizes["TIME"], freq="D")
        da = da.assign_coords(TIME=times)
        for city, (lat, lon) in CITIES.items():
            li = int(np.abs(ds["LATITUDE"].values - lat).argmin())
            lo = int(np.abs(ds["LONGITUDE"].values - lon).argmin())
            sub = da.isel(LATITUDE=slice(max(li-1, 0), li+2), LONGITUDE=slice(max(lo-1, 0), lo+2))
            daily = sub.mean(dim=["LATITUDE", "LONGITUDE"], skipna=True).to_series()
            monthly = daily.resample("MS").sum(min_count=1)
            for d, v in monthly.items():
                rows.append((city, d, v))
        ds.close()
    df = pd.DataFrame(rows, columns=["city", "date", "rain_mm"]).sort_values(["city", "date"])
    return df.reset_index(drop=True)


def make_features(df):
    """Lag/rolling/seasonal features. Target = NEXT month's rainfall. Climatology uses train years only."""
    out = []
    clim = (df[df.date.dt.year <= TRAIN_END].groupby(["city", df.date.dt.month])["rain_mm"]
              .mean().rename("clim"))
    for city, g in df.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        r = g["rain_mm"]
        f = pd.DataFrame({"city": city, "date": g["date"]})
        for L in (1, 2, 3, 12):
            f[f"lag{L}"] = r.shift(L - 1)           # lag1 = this month, observed at time t
        f["roll3"] = r.rolling(3).mean()
        f["roll6"] = r.rolling(6).mean()
        nm = (g["date"].dt.month % 12) + 1          # month of the target
        f["m_sin"] = np.sin(2 * np.pi * nm / 12)
        f["m_cos"] = np.cos(2 * np.pi * nm / 12)
        f["target_date"] = g["date"] + pd.offsets.MonthBegin(1)
        f["clim_next"] = [clim.get((city, m), np.nan) for m in nm]
        f["seasonal_naive"] = r.shift(11)           # same month as target, last year
        f["y"] = r.shift(-1)
        out.append(f)
    d = pd.concat(out, ignore_index=True)
    d["city_id"] = d["city"].astype("category").cat.codes
    d = d.dropna(subset=["lag12", "roll6", "y", "seasonal_naive"]).reset_index(drop=True)
    return d


FEATS = ["lag1", "lag2", "lag3", "lag12", "roll3", "roll6", "m_sin", "m_cos", "clim_next", "city_id"]


def split(d):
    yr = d["target_date"].dt.year
    return d[yr <= TRAIN_END], d[yr == VAL_YEAR], d[yr >= TEST_START]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return rmse, mae, r2

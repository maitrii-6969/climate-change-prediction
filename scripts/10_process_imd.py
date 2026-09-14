# 10_process_imd.py
# Processes real IMD rainfall data and combines with ERA5
# Run: python scripts/10_process_imd.py

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

ROOT       = r"C:\climate_project"
IMD_DIR    = os.path.join(ROOT, "data", "raw", "imd")
IDX_DIR    = os.path.join(ROOT, "data", "indices")
MODELS_DIR = os.path.join(ROOT, "models", "saved")
OUT_DIR    = os.path.join(ROOT, "outputs", "maps")
os.makedirs(IDX_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

CITIES = {
    "Mumbai":    {"lat": 19.1, "lon": 72.9},
    "Delhi":     {"lat": 28.6, "lon": 77.2},
    "Chennai":   {"lat": 13.1, "lon": 80.3},
    "Kolkata":   {"lat": 22.6, "lon": 88.4},
    "Bangalore": {"lat": 12.9, "lon": 77.6},
    "Hyderabad": {"lat": 17.4, "lon": 78.5},
    "Jaipur":    {"lat": 26.9, "lon": 75.8},
    "Ahmedabad": {"lat": 23.0, "lon": 72.6},
    "Bhopal":    {"lat": 23.3, "lon": 77.4},
    "Nagpur":    {"lat": 21.1, "lon": 79.1},
}

print("="*60)
print("PROCESSING REAL IMD RAINFALL DATA")
print("="*60)

# ── Load all IMD years ────────────────────────────────────────────────────────
print("\n[1/4] Loading IMD rainfall files (2015-2024)...")
monthly_data = []

for yr in range(2015, 2025):
    fpath = os.path.join(IMD_DIR, f"RF25_ind{yr}_rfp25.nc")
    ds    = xr.open_dataset(fpath)
    rf    = ds["RAINFALL"]
    dates = pd.date_range(f"{yr}-01-01", periods=len(rf.TIME), freq="D")
    rf_da = xr.DataArray(
        rf.values,
        dims=["time","lat","lon"],
        coords={
            "time": dates,
            "lat":  ds["LATITUDE"].values,
            "lon":  ds["LONGITUDE"].values,
        }
    )
    monthly = rf_da.resample(time="1MS").sum()
    monthly_data.append(monthly)
    ann = float(rf_da.sum(dim="time").mean())
    print(f"   {yr}: {len(rf.TIME)} days | India mean: {ann:.0f} mm")
    ds.close()

combined = xr.concat(monthly_data, dim="time")
print(f"\n   Combined: {combined.shape} (months × lat × lon)")

# Save monthly IMD stack
out_nc = os.path.join(IDX_DIR, "imd_rainfall_monthly.nc")
combined.to_dataset(name="rainfall").to_netcdf(out_nc)
print(f"   ✅ Saved: imd_rainfall_monthly.nc")

# ── Build combined ERA5 + IMD dataset ────────────────────────────────────────
print("\n[2/4] Loading ERA5 data and combining with IMD...")
lst_ds = xr.open_dataset(os.path.join(IDX_DIR, "era5_lst_india.nc"))
t2m_ds = xr.open_dataset(os.path.join(IDX_DIR, "era5_t2m_india.nc"))
lst    = lst_ds["lst"]
t2m    = t2m_ds["t2m"]

if "valid_time" in lst.coords:
    lst = lst.rename({"valid_time":"time"})
    t2m = t2m.rename({"valid_time":"time"})

# Filter ERA5 to 2015-2024
era5_times = pd.DatetimeIndex(lst.time.values)
mask = (era5_times.year >= 2015) & (era5_times.year <= 2024)
lst  = lst.isel(time=mask)
t2m  = t2m.isel(time=mask)

print(f"   ERA5 LST:  {lst.shape}")
print(f"   IMD Rain:  {combined.shape}")

# ── Build ML dataset ─────────────────────────────────────────────────────────
print("\n[3/4] Building combined feature dataset...")
rows = []
times = pd.DatetimeIndex(lst.time.values)

for city, coords in CITIES.items():
    lst_ts  = lst.sel(latitude=coords["lat"],  longitude=coords["lon"],  method="nearest").values
    t2m_ts  = t2m.sel(latitude=coords["lat"],  longitude=coords["lon"],  method="nearest").values
    rain_ts = combined.sel(lat=coords["lat"],   lon=coords["lon"],        method="nearest").values

    lst_norm  = (lst_ts  - lst_ts.min())  / (lst_ts.max()  - lst_ts.min()  + 1e-8)
    rain_norm = (rain_ts - rain_ts.min()) / (rain_ts.max() - rain_ts.min() + 1e-8)

    for i, date in enumerate(times):
        rows.append({
            "date":        date,
            "city":        city,
            "year":        date.year,
            "month":       date.month,
            # Real ERA5 features
            "lst":         lst_ts[i],
            "t2m":         t2m_ts[i],
            "lst_norm":    lst_norm[i],
            # Real IMD rainfall features
            "rainfall":    rain_ts[i],
            "rain_norm":   rain_norm[i],
            # Lag features
            "lst_lag1":    lst_ts[i-1]   if i>0  else lst_ts[i],
            "lst_lag3":    lst_ts[i-3]   if i>2  else lst_ts[i],
            "lst_lag12":   lst_ts[i-12]  if i>11 else lst_ts[i],
            "rain_lag1":   rain_ts[i-1]  if i>0  else rain_ts[i],
            "rain_lag3":   rain_ts[i-3]  if i>2  else rain_ts[i],
            "rain_lag12":  rain_ts[i-12] if i>11 else rain_ts[i],
            "t2m_lag1":    t2m_ts[i-1]   if i>0  else t2m_ts[i],
            # Rolling means
            "lst_roll3":   float(np.mean(lst_ts[max(0,i-2):i+1])),
            "lst_roll6":   float(np.mean(lst_ts[max(0,i-5):i+1])),
            "rain_roll3":  float(np.mean(rain_ts[max(0,i-2):i+1])),
            "rain_roll6":  float(np.mean(rain_ts[max(0,i-5):i+1])),
            # Cyclical month encoding
            "month_sin":   np.sin(2*np.pi*date.month/12),
            "month_cos":   np.cos(2*np.pi*date.month/12),
            # Target
            "target_lst":  lst_ts[i+1] if i < len(times)-1 else lst_ts[i],
        })

df = pd.DataFrame(rows)
print(f"   Total samples : {len(df):,}")
print(f"   Features      : 20 (ERA5 + IMD combined)")
print(f"   Cities        : {df['city'].nunique()}")

FEATURES = [
    "lst","t2m","lst_norm",
    "rainfall","rain_norm",
    "lst_lag1","lst_lag3","lst_lag12",
    "rain_lag1","rain_lag3","rain_lag12",
    "t2m_lag1",
    "lst_roll3","lst_roll6",
    "rain_roll3","rain_roll6",
    "month_sin","month_cos",
]
TARGET = "target_lst"

# ── Train/Val/Test split ──────────────────────────────────────────────────────
train = df[df["year"] <= 2021]
val   = df[df["year"] == 2022]
test  = df[df["year"] >= 2023]

X_train = train[FEATURES].values; y_train = train[TARGET].values
X_val   = val[FEATURES].values;   y_val   = val[TARGET].values
X_test  = test[FEATURES].values;  y_test  = test[TARGET].values

print(f"\n   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\n[4/4] Training XGBoost on ERA5 + IMD combined data...")
model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1,
    reg_lambda=1.0, random_state=SEED,
    n_jobs=-1, early_stopping_rounds=30,
    eval_metric="rmse", verbosity=0,
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)], verbose=False)

def evaluate(X, y, name):
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"   {name}: RMSE={rmse:.4f}°C | MAE={mae:.4f} | R²={r2:.4f}")
    return preds, rmse, mae, r2

evaluate(X_train, y_train, "Train     ")
evaluate(X_val,   y_val,   "Validation")
test_preds, rmse, mae, r2 = evaluate(X_test, y_test, "TEST      ")

# Save model
joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_era5_imd_combined.pkl"))
print(f"\n✅ Combined model saved: xgboost_era5_imd_combined.pkl")

# ── Plots ─────────────────────────────────────────────────────────────────────
city = "Mumbai"
city_test  = test[test["city"]==city].copy()
city_preds = model.predict(city_test[FEATURES].values)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("REAL DATA — ERA5 + IMD Combined Model Results",
             fontsize=14, fontweight="bold", color="#1A3C6E")

# Prediction vs actual
axes[0].plot(city_test["date"].values, city_test[TARGET].values,
             color="#2e8b57", linewidth=2, label="Actual LST (ERA5)", alpha=0.9)
axes[0].plot(city_test["date"].values, city_preds,
             color="#D85A30", linewidth=2, linestyle="--",
             label="XGBoost prediction (ERA5+IMD)", alpha=0.85)
axes[0].set_title(f"LST Prediction vs Actual — {city} (Test 2023–2024)",
                  fontsize=12, color="#1A3C6E")
axes[0].set_ylabel("LST (°C)")
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_facecolor("#f9f9f9")

# Feature importance
fi = pd.DataFrame({"feature":FEATURES,
                   "importance":model.feature_importances_})
fi = fi.sort_values("importance", ascending=True)